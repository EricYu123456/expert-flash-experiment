import numpy as np
from math import ceil
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union, Optional, Any
import torch
import threading 
import config
from quantization_utils import *
from pipline import Pipline

@dataclass
class MatrixPosition:
    """表示矩阵在3D NAND中的位置"""
    layer: int
    plane: int
    block: int
    tsg: int
    bl: int  # bitline_index
    block_epochs: int  # block_group_index
    
    def copy(self):
        """创建位置对象的副本"""
        return MatrixPosition(
            layer=self.layer,
            plane=self.plane,
            block=self.block,
            tsg=self.tsg,
            bl=self.bl,
            block_epochs=self.block_epochs
        )   
    def update_after_computation(self, bl_size: int, hardware: 'NANDHardwareSimulator') -> 'MatrixPosition':
        """计算完成后更新位置"""
        plane_add, self.bl = divmod((self.bl + bl_size), hardware.bitlines_per_plane)
        tsg_add, self.plane = divmod((self.plane + plane_add), hardware.planes_per_die)
        layer_add, self.tsg = divmod((self.tsg + tsg_add), hardware.tsg_per_block)
        block_epochs_add, self.layer = divmod(
            (self.layer + layer_add), hardware.layer_per_die
        )
        self.block_epochs = self.block_epochs + block_epochs_add
        delta=[plane_add,tsg_add,layer_add,block_epochs_add]
        return self,delta

    
    @classmethod
    def from_dict(cls, position_dict: Dict[str, int]) -> 'MatrixPosition':
        """从字典创建位置对象"""
        return cls(
            layer=position_dict['layer'],
            plane=position_dict['plane'],
            block=position_dict['block'],
            tsg=position_dict['tsg'],
            bl=position_dict['bl'],
            block_epochs=position_dict['block_epochs']
        )
    
    def to_dict(self) -> Dict[str, int]:
        """转换为字典格式"""
        return {
            'layer': self.layer,
            'plane': self.plane,
            'block': self.block,
            'tsg': self.tsg,
            'bl': self.bl,
            'block_epochs': self.block_epochs
        }

class NANDHardwareSimulator:
    """
    3D NAND硬件模拟器，用于神经网络计算
    
    该模拟器模拟在3D NAND闪存架构上进行神经网络运算的过程，
    包括权重存储、量化处理、电流计算和时序模拟。
    
    主要组件:
    - 单元阵列: 模拟3D NAND闪存的存储单元
    - TSG (Transistor Select Gate): 晶体管选择门控制
    - WL (Word Line): 字线控制
    - BL (Bit Line): 位线读取
    
    主要功能:
    - 加载量化权重到模拟存储器
    - 处理输入向量
    - 执行模拟的并行计算
    - 追踪计算时间和功耗
    """
    
    def __init__(self, hidden_dimension, logger, device=None):
        """
        初始化3D NAND硬件模拟器
        
        参数:
        - hidden_dimension: 隐藏层维度
        - logger: 日志记录器对象
        - device: 计算设备 (默认: GPU如果可用，否则CPU)
        """
        # 设置计算设备
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 基础参数
        self.quantization_bits = config.QUANTIZATION_BITS  # 量化位数
        self.current_variation_mode = config.CURRENT_VARIATION_MODE  # 电流变化模式
        self.max_current = config.MAX_CURRENT  # nA，最大电流
        self.current_mean = config.MEAN  # 电流均值
        self.current_scale = config.SCALE  # 电流标准差
        self.logger = logger  # 日志记录器
        self.hidden_dimension = hidden_dimension  # 隐藏层维度
        
        # 操作参数
        self.max_current_sum = config.MAX_CURRENT_SUM  # 最大电流和
        self.blocks_per_operation = config.BLOCKS_PER_OPERATION  # 每次操作的块数
        
        # 硬件架构参数 - 初始化并计算衍生参数
        self._initialize_hardware_parameters()
        
        # 跟踪变量初始化
        self.matrix_locations = {}  # 存储矩阵位置信息
        self.total_space = self._calculate_total_space()
        self.rest_space = self.total_space  # 剩余空间
        
        # 创建TSG和WL实例
        self._initialize_instances()

        # 创建单元阵列
        self._create_cell_array()
        
        # 创建量化位数对应的权重激活和
        self._initialize_activation_weights()
        
        # 功耗相关参数初始化
        self._initialize_power_parameters()

        # 添加线程同步相关的属性
        self.timeline_lock = threading.Lock()  # 用于保护共享资源的锁
        self.current_hardware_state = {
            'block_group': 0,
            'layer': 0,
            'tsg': 0,
            'plane': 0,
            'bl': 0,
            'time': 0,
            'power': 0,
            'sense_power': 0
        }
        self.timeline_thread = None
        self.computation_thread = None
        self.current_matrix = None
        self.current_bl_voltages = None
        self.current_position = None
        self.result_dict = {}
    def _initialize_instances(self):
        """初始化TSG和WL实例"""
        # 创建多维列表存储TSG和WL实例
        self.instance_tsg = create_multidim_list(
            [self.tsg_per_block,self.planes_per_die], 
            None
        )
        self.instance_wl = create_multidim_list(
            [self.layer_per_die], 
            None
        )
        self.instance_block_group = create_multidim_list(
            [self.blocks_per_plane // self.blocks_per_operation], 
            None
        )
        
        # 初始化TSG实例
        for tsg_idx in range(self.tsg_per_block):
            for plane_idx in range(self.planes_per_die):
                self.instance_tsg[tsg_idx][plane_idx] = Pipline(
                    f'TSG{tsg_idx}_Plane{plane_idx}',
                    self.tsg_switch_time
                )
            
        # 初始化WL实例
        for layer_idx in range(self.layer_per_die):        
            self.instance_wl[layer_idx]= Pipline(
                f'Layer{layer_idx}',
                self.wl_switch_time
                )

        # 初始化Block组实例       
        for blocks_group_idx in range(self.blocks_per_plane // self.blocks_per_operation):
            self.instance_block_group[blocks_group_idx] = Pipline(
                f'Block组{blocks_group_idx}',
                self.wl_setup_time
                )
                
    def _initialize_hardware_parameters(self):
        """初始化硬件架构和时序参数"""
        # 硬件架构参数
        self.planes_per_die = config.PLANES_PER_DIE  # 每个die的平面数
        self.layer_per_die = config.LAYER_PER_DIE  # 每个die的层数
        self.blocks_per_plane = config.BLOCKS_PER_PLANE // self.blocks_per_operation * self.blocks_per_operation  # 跟随量化位调整有效Block数
        self.tsg_per_block = config.TSG_PER_BLOCK  # 每个块的TSG数
        self.Y_lost = config.Y_LOST  # Y方向损失
        
        # 位线计算
        self.bitlines_per_plane = self._calculate_bitlines_per_plane(config.BITLINES_PER_PLANE)
        
        # TIA相关计算
        self.TIA_num = config.TIA_NUM_BASE // config.BLOCKS_PER_OPERATION * config.BLOCKS_PER_OPERATION
        self.BL_Group_per_operation = self._calculate_bl_groups_per_operation()
        self.only_once_planecalcu_num = self._calculate_once_plane_calcu_num()
        
        # 硬件时序参数（微秒）
        self.wl_setup_time = config.WL_SETUP_TIME_BASE * self.blocks_per_operation
        self.wl_switch_time = config.WL_SWITCH_TIME
        self.tsg_switch_time = config.TSG_SWITCH_TIME
        self.bl_switch_time = config.BL_SWITCH_TIME
        self.tia_conversion_time = config.TIA_CONVERSION_TIME
        self.adc_conversion_time = config.ADC_CONVERSION_TIME_BASE * config.NUM_TIA_TIMES_ADC
        self.write_time = config.WRITE_TIME
        
        # Block分组计算
        self.num_block_epochs = self._calculate_block_epochs()

    def _calculate_bitlines_per_plane(self, initial_bitlines):
        """计算有效的每个平面的位线数"""
        y_num=(initial_bitlines-self.max_current_sum)//(self.max_current_sum+self.Y_lost)
        BL_num=y_num*self.max_current_sum+self.max_current_sum
        return BL_num


    def _calculate_bl_groups_per_operation(self):
        """计算每次操作的BL组数"""
        groups = self.TIA_num // config.BLOCKS_PER_OPERATION
        return groups 

    def _calculate_once_plane_calcu_num(self):
        """计算每次计算时平面的计算数"""
        return self.BL_Group_per_operation * self.max_current_sum // self.bitlines_per_plane

    def _calculate_block_epochs(self):
        """计算Block分组"""
        base_epochs = np.array([self.blocks_per_operation]).repeat(
            self.blocks_per_plane // self.blocks_per_operation
        )
        
        # 如果有余数，添加一个包含余数的块
        if self.blocks_per_plane % self.blocks_per_operation != 0:
            base_epochs = np.append(
                base_epochs, 
                (self.blocks_per_plane % self.blocks_per_operation)
            )
        return base_epochs

    def _calculate_total_space(self):
        """计算总存储空间"""
        return (self.layer_per_die * self.planes_per_die * self.blocks_per_plane * 
                self.tsg_per_block * self.bitlines_per_plane)

    def _create_cell_array(self):
        """创建模拟的存储单元阵列"""
        # 根据是否模拟电流变化选择合适的数据类型
        dtype = torch.int8 if not self.current_variation_mode else torch.float16
        
        self.cell_array = torch.zeros(
            (   
                self.planes_per_die,
                self.layer_per_die,
                self.blocks_per_plane,
                self.tsg_per_block,
                self.bitlines_per_plane 
            ),
            dtype=dtype,
            device=self.device
        )

    def _initialize_activation_weights(self):
        """初始化量化位数对应的权重激活和"""
        # 创建符号列表
        sign_list = np.array([1] * (self.quantization_bits - 1) + [-1])
        
        # 创建权重和激活位的值
        weight_bits = np.array([2**n * sign_list[n] for n in range(self.quantization_bits)])
        activation_bits = np.array([2**m * sign_list[m] for m in range(self.quantization_bits)])
        
        # 计算激活和权重的外积
        self.activation_sums_weight = np.outer(activation_bits, weight_bits).reshape(
            self.quantization_bits, self.quantization_bits
        )

    def _initialize_power_parameters(self):
        """初始化功耗相关参数"""
        # 背景电流和控制电流
        self.background_current = config.BG_CURRENT * (1+self.blocks_per_operation*0.01)  # A
        self.tsg_wl_current = config.TSG_WL_CURRENT # A
        self.bl_current = config.BL_CURRENT # A
        # 电源电压
        self.Vcc = config.VCC  # V
        # 背景功耗
        self.bg_power=self.background_current*self.Vcc # W
        # Block组建立功耗
        self.block_group_power= self.tsg_wl_current * self.Vcc * self.wl_setup_time * self.planes_per_die *self.blocks_per_operation # uJ
        # Bl建立功耗
        self.single_bl_power=self.bl_current*self.Vcc*self.bl_switch_time #uJ
        # TIA建立功耗+ADC建立功耗
        self.TIA_ADC_power_single=config.TIA_ADC_POWER_BASE*(self.tia_conversion_time+self.adc_conversion_time) #uJ
        # 切换WL功耗
        self.layer_change_power=self.tsg_wl_current*self.Vcc*self.wl_switch_time*self.planes_per_die *self.blocks_per_operation #uJ
    def load_weights(self, weight_matrix, location=None, matrix_name=None):
        """
        加载权重矩阵到硬件模拟器
        
        参数:
        - weight_matrix: 要加载的权重矩阵
        - location: 起始位置字典(可选)
        - matrix_name: 矩阵标识名称(可选)
        
        返回:
        - scale: 量化缩放因子
        - end: 结束位置
        - quantized_weight: 量化后的权重
        """
        # 确定起始位置
        start_position = self._determine_start_position(location, matrix_name)
        
        # 量化权重
        quantized_weight, scale, zero_point=quantize_per_tensor(
            weight_matrix,
            bits=self.quantization_bits,
            signed=True,
            per_channel=True,
            channel_dim=1
        )
        # 位展开和格式转换
        expanded_weight_final = self._expand_weight_bits(quantized_weight,weight_matrix)
        
        # 写入权重到存储
        end_position = self._write_weights_to_memory(expanded_weight_final, start_position)
        
        # 更新存储状态
        self._update_storage_status(expanded_weight_final.numel())
        
        # 保存矩阵位置信息
        if matrix_name:
            self.matrix_locations[matrix_name] = {
                'start': start_position.to_dict(),
                'end': end_position.to_dict()
            }
        
        # 检查存储限制
        if end_position.block_epochs >= len(self.num_block_epochs):
            end_position = MatrixPosition(
                layer=self.layer_per_die - 1, 
                plane=self.planes_per_die - 1,
                block=self.blocks_per_plane - 1, 
                tsg=self.tsg_per_block - 1,
                bl=self.bitlines_per_plane - 1,
                block_epochs=len(self.num_block_epochs) - 1
            )
            raise ValueError("不足够的空间存储所有权重。")

        
        return scale, end_position.to_dict(), quantized_weight

    def _determine_start_position(self, location, matrix_name):
        """确定权重加载的起始位置"""
        if matrix_name in self.matrix_locations and 'end' in self.matrix_locations[matrix_name]:
            # 使用之前矩阵的结束位置作为起始位置
            start_dict = self.matrix_locations[matrix_name]['end']
            return MatrixPosition.from_dict(start_dict)
        elif location is not None:
            # 使用指定的位置
            return MatrixPosition.from_dict(location)
        else:
            # 默认从第一个可用位置开始
            return MatrixPosition(
                layer=0, plane=0, block=0, tsg=0, bl=0, block_epochs=0
            )

    def _expand_weight_bits(self, quantized_weight,weight_matrix):
        """展开权重的位表示"""
        # 转换为无符号整数类型
        if self.quantization_bits <= 8:
            quantized_weight_unsigned = quantized_weight.astype(np.uint8)
        else:
            quantized_weight_unsigned = quantized_weight.astype(np.uint16)
            
        # 手动拆分并展开位
        shape = quantized_weight_unsigned.shape
        expanded_weight = np.zeros((*shape, self.quantization_bits), dtype=np.int8)
        
        # 逐位提取
        for bit in range(self.quantization_bits):
            # 使用位操作提取每一位
            bit_val = ((quantized_weight_unsigned & (1 << bit)) > 0).astype(np.int8)
            # little endian顺序存储
            expanded_weight[:, :, bit] = bit_val
            
        # 计算存储布局
        already_bl_group, rest_bl = divmod(weight_matrix.shape[0], self.max_current_sum)
        
        # 如果需要，填充到完整BL段
        if rest_bl != 0:
            expanded_weight = pad_to_target_shape(
                expanded_weight, 
                (already_bl_group + 1) * self.max_current_sum, 
                0
            )
        
        # 重塑权重以适应存储格式
        expanded_weight_final = torch.tensor(
            expanded_weight.reshape(
                expanded_weight.shape[0] // self.max_current_sum, 
                -1, 
                expanded_weight.shape[1], 
                expanded_weight.shape[2]
            ).repeat(self.quantization_bits, axis=0).reshape(
                -1, 
                expanded_weight.shape[1], 
                expanded_weight.shape[2]
            ),
            device=self.device
        )
        
        return expanded_weight_final

    def _write_weights_to_memory(self, expanded_weight_final, position):
        """将展开的权重写入到存储中"""  
        need_block_group_num = expanded_weight_final.shape[1]
        
        # 逐块写入权重
        for block_group_index in range(need_block_group_num):
            if self.bitlines_per_plane - position.bl >= expanded_weight_final.shape[0]:
                # 当前位面有足够空间容纳整个权重块
                self._write_weight_block_to_plane(
                    expanded_weight_final,
                    block_group_index,
                    position,
                    expanded_weight_final.shape[0]
                )
            else:
                # 需要跨位面存储
                self._write_weight_block_across_planes(
                    expanded_weight_final,
                    block_group_index,
                    position
                )
                
            # 更新位置索引用于下一组权重
            position = self._update_position_after_write(position, expanded_weight_final.shape[0])
            
        return position

    def _write_weight_block_to_plane(self, expanded_weight, block_index, position, size):
        """将权重块写入到单个位面"""
        # 计算块索引范围
        block_slice = slice(
            position.block_epochs * self.blocks_per_operation + position.block, 
            position.block_epochs * self.blocks_per_operation + position.block + self.quantization_bits
        )
        
        # 提取权重数据
        weight_data = expanded_weight[:size, block_index, :].T
        
        # 应用电流变化（如果启用）
        if self.current_variation_mode:
            # 生成随机噪声
            noise = torch.normal(
                mean=self.current_mean, 
                std=self.current_scale, 
                size=weight_data.shape
            ).to(expanded_weight.device)
            
            # 应用噪声
            weight_data = weight_data * noise
        
        # 写入到存储阵列
        self.cell_array[
            position.plane, position.layer, block_slice, position.tsg, 
            position.bl:position.bl+size
        ] = weight_data

    def _write_weight_block_across_planes(self, expanded_weight, block_index, position):
        """将权重块写入到多个位面（处理跨平面情况）"""
        # 计算第一个平面的剩余空间
        rest_bl_perplane = self.bitlines_per_plane - position.bl
        
        # 写入第一部分到当前平面
        self._write_weight_block_to_plane(
            expanded_weight, 
            block_index, 
            position, 
            rest_bl_perplane
        )
        
        # 计算还需要多少完整的平面
        complete_plane_num, last_rest_bl = divmod(
            (expanded_weight.shape[0] - rest_bl_perplane),
            self.bitlines_per_plane
        )
        
        # 当前位置
        temp_position = MatrixPosition(
            layer=position.layer,
            plane=position.plane,
            block=position.block,
            tsg=position.tsg,
            bl=position.bl,
            block_epochs=position.block_epochs
        )
        
        # 写入完整平面部分
        for i in range(complete_plane_num):
            # 更新到下一个平面
            temp_position = self._move_to_next_plane(temp_position)
            
            # 写入完整平面
            block_slice = slice(
                temp_position.block_epochs * self.blocks_per_operation + temp_position.block, 
                temp_position.block_epochs * self.blocks_per_operation + temp_position.block + self.quantization_bits
            )
            
            weight_data = expanded_weight[
                self.bitlines_per_plane * i + rest_bl_perplane:self.bitlines_per_plane * (i + 1) + rest_bl_perplane, 
                block_index, 
                :
            ].T
            
            if self.current_variation_mode:
                noise = torch.normal(
                    mean=self.current_mean, 
                    std=self.current_scale, 
                    size=weight_data.shape
                ).to(expanded_weight.device)
                weight_data = weight_data * noise
                
            self.cell_array[
                temp_position.plane, temp_position.layer, block_slice, temp_position.tsg, 
                :
            ] = weight_data
        
        # 写入最后剩余部分（如果有）
        if last_rest_bl > 0:
            # 更新到下一个平面
            temp_position = self._move_to_next_plane(temp_position)
            
            # 写入剩余部分
            block_slice = slice(
                temp_position.block_epochs * self.blocks_per_operation + temp_position.block, 
                temp_position.block_epochs * self.blocks_per_operation + temp_position.block + self.quantization_bits
            )
            
            weight_data = expanded_weight[-last_rest_bl:, block_index, :].T
            
            if self.current_variation_mode:
                noise = torch.normal(
                    mean=self.current_mean, 
                    std=self.current_scale, 
                    size=weight_data.shape
                ).to(expanded_weight.device)
                weight_data = weight_data * noise
                
            self.cell_array[
                temp_position.plane, temp_position.layer, block_slice, temp_position.tsg, 
                :last_rest_bl
            ] = weight_data

    def _move_to_next_plane(self, position):
        """移动到下一个平面，处理平面、TSG、层和块组的进位"""
        tsg_add, next_plane = divmod((position.plane + 1), self.planes_per_die)
        layer_add, next_tsg = divmod((position.tsg + tsg_add), self.tsg_per_block)
        block_epochs_add, next_layer = divmod(
            (position.layer + layer_add), 
            self.layer_per_die
        )
        
        next_block = position.block
        if layer_add:
            next_block = 0
            
        next_block_epochs = position.block_epochs + block_epochs_add
        
        return MatrixPosition(
            layer=next_layer,
            plane=next_plane,
            block=next_block,
            tsg=next_tsg,
            bl=0,  # 新平面从0开始
            block_epochs=next_block_epochs
        )

    def _update_position_after_write(self, position, size):
        """更新写入后的位置"""
        # 计算新的块位置
        bl_add, next_block = divmod(
            (position.block + self.quantization_bits), 
            self.num_block_epochs[position.block_epochs]
        )
        
        # 计算新的位线位置
        plane_add, next_bl = divmod((position.bl + bl_add * size), self.bitlines_per_plane)
        
        # 计算新的平面位置
        tsg_add, next_plane = divmod((position.plane + plane_add), self.planes_per_die)
        
        # 计算新的TSG位置
        layer_add, next_tsg = divmod((position.tsg + tsg_add), self.tsg_per_block)
        
        # 计算新的层位置和块组位置
        block_epochs_add, next_layer = divmod(
            (position.layer + layer_add), 
            self.layer_per_die
        )
        next_block_epochs = position.block_epochs + block_epochs_add
        
        return MatrixPosition(
            layer=next_layer,
            plane=next_plane,
            block=next_block,
            tsg=next_tsg,
            bl=next_bl,
            block_epochs=next_block_epochs
        )


    def _update_storage_status(self, elements_count):
        """更新存储状态"""
        self.rest_space -= elements_count

    def set_input(self, input_vector, sign_flag=True, symmetric=True):
        """
        处理单个输入向量
        
        参数:
        - input_vector: 输入向量
        - sign_flag: 是否有符号量化
        - symmetric: 是否使用对称量化
        
        返回:
        - expanded_input_final: 展开的输入向量
        - scale_input: 量化缩放因子
        - zero_point_input: 量化零点（仅在非对称量化时返回）
        """
        # 量化输入
        quantized_input, scale_input, zero_point_input = quantize_per_tensor(
            input_vector,
            bits=self.quantization_bits,
            signed=sign_flag,
            per_channel=True,
            channel_dim=0,  # 输入向量使用逐张量量化
            symmetric=symmetric
        )
        
        # 转换为无符号整数保持位模式
        quantized_input_unsigned = self._convert_to_unsigned(quantized_input)
        
        # 检查是否需要填充
        padded_input = self._pad_input_if_needed(quantized_input_unsigned)
        
        # 位展开
        expanded_input = self._expand_input_bits(padded_input)
        
        # 重塑输入以适应硬件格式
        expanded_input_final = self._reshape_input_for_hardware(expanded_input)
        
        # 返回结果
        if symmetric:
            return expanded_input_final, scale_input, None
        else:
            return expanded_input_final, scale_input, zero_point_input

    def _convert_to_unsigned(self, quantized_input):
        """将量化输入转换为无符号整数"""
        if self.quantization_bits <= 8:
            return quantized_input.astype(np.uint8)
        else:
            return quantized_input.astype(np.uint16)
            
    def _pad_input_if_needed(self, quantized_input):
        """如果需要，对输入进行填充使其符合硬件要求"""
        # 检查是否需要填充
        already_bl_group, rest_bl = divmod(quantized_input.shape[1], self.max_current_sum)
        
        if rest_bl != 0:
            self.logger.debug(f'提示：输入的维度无法填充完整单个或多个SL区块,'
                              f'补0直到{already_bl_group+1}个bl_per_operation')
            return pad_to_target_shape(
                quantized_input, 
                (already_bl_group+1)*self.max_current_sum, 
                1
            )
        else:
            return quantized_input

    def _expand_input_bits(self, padded_input):
        """展开输入的位表示"""
        shape = padded_input.shape
        expanded_input = np.zeros((*shape, self.quantization_bits), dtype=np.int8) 
        
        # 逐位提取
        for bit in range(self.quantization_bits):
            # 使用位操作提取每一位
            bit_val = ((padded_input & (1 << bit)) > 0).astype(np.int8)
            # little endian顺序存储
            expanded_input[:, :, bit] = bit_val
            
        return expanded_input

    def _reshape_input_for_hardware(self, expanded_input):
        """重塑输入以适应硬件格式"""
        return expanded_input.reshape(
            expanded_input.shape[0],
            expanded_input.shape[1]//self.max_current_sum,
            -1, 
            self.quantization_bits
        ).transpose(0, 1, 3, 2).reshape(
            expanded_input.shape[0],
            expanded_input.shape[1]//self.max_current_sum,
            -1
        ).reshape(expanded_input.shape[0], -1)

    def compute(self, bl_voltages, matrix_name, total_compute_time):
        """
        执行硬件计算
        
        参数:
        - bl_voltages: 输入电压值，形状为 [sequency_length, features]
        - matrix_name: 要计算的矩阵名称，可以是单个字符串或字符串列表
        - total_compute_time: 当前累计计算时间
        
        返回:
        - combined_results: 计算结果字典，键为矩阵名称，值为计算结果
        - total_compute_time: 更新后的总计算时间
        - total_power: 总功耗
        - total_sensing_power: 总感应功耗
        """

        # 确保 matrix_name 是列表
        matrix_name = [matrix_name] if not isinstance(matrix_name, list) else matrix_name
        
        # 初始化结果和功耗
        matrix_results = {matrix: [] for matrix in matrix_name}
        total_sensing_power = 0.0
        total_power = 0.0
        
        # 处理每个批次的输入
        for i in range(bl_voltages.shape[0]):
            # 计算单个输入结果
            result, compute_time, power, sensing_power = self._compute_single(
                bl_voltages[i], matrix_name, total_compute_time
            )
            
            # 更新总计算时间和功耗
            total_compute_time = compute_time #时间是连续的
            total_power += power
            total_sensing_power += sensing_power
            
            # 跳过无效结果
            if result is None:
                continue
            
            # 收集矩阵结果
            for matrix in matrix_name:
                if matrix in result:
                    matrix_results[matrix].append(result[matrix].reshape(1, -1))

        # 合并结果
        combined_results = {}
        for matrix in matrix_name:
            # 只有存在结果时才进行合并
            if matrix_results[matrix]:
                combined_results[matrix] = np.concatenate(matrix_results[matrix], axis=0)
            else:
                combined_results[matrix] = np.array([])
        
        return combined_results, total_compute_time, total_power, total_sensing_power
    def _compute_single(self, bl_voltages, matrix_name, total_compute_time):
        """
        执行单个输入的硬件计算 - 修改为启动两个并行线程
        """
        # 重塑输入向量
        bl_voltages = bl_voltages.reshape(-1)
        self.barrier = threading.Barrier(2)
        # 初始化计算所需的共享变量
        self.current_matrix = matrix_name if isinstance(matrix_name, list) else [matrix_name]
        self.current_bl_voltages = bl_voltages
        self.current_position = self._get_matrix_position(self.current_matrix[0])
        self.result_dict = {}
        
        # 启动timeline跟踪线程
        self.timeline_thread = threading.Thread(
            target=self._timeline,
            args=(total_compute_time, self.current_position)
        )
        self.timeline_thread.daemon = True
        # 启动计算线程
        self.computation_thread = threading.Thread(
            target=self._computation,
            args=(self.current_position,)
        )
        self.computation_thread.daemon = True
        
        # 启动两个线程
        self.timeline_thread.start()
        self.computation_thread.start()
        
        # 等待计算完成
        self.computation_thread.join()
        # 停止timeline线程
        self.timeline_thread.join()
        
        # 获取结果
        with self.timeline_lock:
            total_compute_time = self.current_hardware_state['time']
            total_power = self.current_hardware_state['power']
            total_sensing_power = self.current_hardware_state['sense_power']
        
        return self.result_dict, total_compute_time, total_power, total_sensing_power


    def _computation(self,position):
        """计算线程逻辑"""
        # 处理每个需要计算的矩阵
        self.barrier.wait()
        for matrix in self.current_matrix:
            self.logger.debug(f"计算{matrix}")  
            # 获取矩阵位置
            stop_flag = True
            # 执行计算直到完成
            while stop_flag:
                # 等待计算完成事件
                # 执行计算
                self.result_dict = self._process_computation(
                    matrix, self.current_bl_voltages, position, self.result_dict
                )
                
                # 更新位置
                position.update_after_computation(self.current_bl_voltages.shape[0], self)
                
                # 检查是否完成计算
                stop_flag = self._check_computation_completed(matrix, position)
    def _timeline(self, start_time, position):
        """
        硬件时序跟踪函数，在独立线程中运行（修改版）
        """  
        time_position=position.copy()
        self.barrier.wait() #保证position没有被_computation修改
        current_time=start_time
        single_transform_bl_group = int(self.BL_Group_per_operation / config.NUM_TIA_TIMES_ADC)
        power = 0
        count=0
        sense_power = 0        
        # 初始化当前硬件状态
        self._check(self.instance_wl[time_position.layer],current_time)#跟随block组同步建立，不需要增加时间
        self._check(self.instance_tsg[time_position.tsg][:],current_time)#跟随block组同步建立，不需要增加时间
        current_time,ready_flag=self._check(self.instance_block_group[time_position.block_epochs],current_time)
        if not ready_flag:
            power += self.block_group_power
        current_time=current_time+self.bl_switch_time#首个BL建立时间
        for matrix in self.current_matrix:
            stop_flag = True
            while stop_flag:
                current_time,_=self._check(self.instance_tsg[time_position.tsg][time_position.plane],current_time)#观察当前TSG Plane是否被建立完成
                # 继续时序推进（每0.25us）
                current_time += self.tia_conversion_time  # us 后续        
                # 更新BL组位置
                count+=1
                time_position,delta_position = time_position.update_after_computation(single_transform_bl_group*self.max_current_sum,self)
                power += self.single_bl_power*single_transform_bl_group*self.max_current_sum #bl功耗
                power += self.TIA_ADC_power_single*single_transform_bl_group*self.blocks_per_operation  # TIA+ADC功耗
                sense_power += self.TIA_ADC_power_single*single_transform_bl_group*self.blocks_per_operation  # TIA+ADC功耗属于感测功耗，单独累加
                self.current_hardware_state['bl_group'] = time_position.bl//self.max_current_sum
                self.current_hardware_state['power'] = power
                self.current_hardware_state['sense_power'] = sense_power
                self.current_hardware_state['time'] = current_time     
                # 检查是否需要建立新的bl组，layer和tsg
                if delta_position[3] != 0:
                    self.logger.debug(f'切换block组，消耗功耗{self.block_group_power}，消耗时间{self.wl_setup_time}')
                    self.instance_block_group[time_position.block_epochs-1].reset()
                    self._check(self.instance_wl[time_position.layer],current_time) #跟随block组同步建立，不需要增加时间
                    self._check(self.instance_tsg[time_position.tsg][:],current_time) #跟随block组同步建立，不需要增加时间
                    current_time,ready_flag=self._check(self.instance_block_group[time_position.block_epochs],current_time)
                    if not ready_flag:
                        current_time+=self.bl_switch_time #Block组切换后，添加首个BL建立时间
                        power += self.block_group_power
                    self.current_hardware_state['power'] = power
                    self.current_hardware_state['sense_power'] = sense_power
                    self.current_hardware_state['time'] = current_time
                    self.logger.debug(f'当前位置索引：Block组{time_position.block_epochs} layer{time_position.layer} tsg{time_position.tsg} plane{time_position.plane}')
                elif delta_position[2] != 0:
                    self.logger.debug(f'切换layer，消耗功耗分布在背景电流中不做统计，消耗时间{self.wl_switch_time}')
                    self.instance_wl[time_position.layer-1].reset()
                    self._check(self.instance_tsg[time_position.tsg][:],current_time)#跟随layer同步建立，不需要增加时间
                    current_time,ready_flag=self._check(self.instance_wl[time_position.layer],current_time)
                    if not ready_flag:
                        current_time=current_time+self.bl_switch_time #WL切换后，添加首个BL建立时间
                    self.current_hardware_state['power'] = power
                    self.current_hardware_state['sense_power'] = sense_power
                    self.current_hardware_state['time'] = current_time
                    self.logger.debug(f'当前位置索引：Block组{time_position.block_epochs} layer{time_position.layer} tsg{time_position.tsg} plane{time_position.plane}')
                elif delta_position[1] != 0: #正常的话plane切换逻辑会覆盖tsg建立逻辑
                    self.logger.debug(f'切换tsg，消耗功耗忽略不计，消耗时间{self.tsg_switch_time}')
                    self.instance_tsg[time_position.tsg-1][self.planes_per_die-1].reset()
                    current_time,_=self._check(self.instance_tsg[time_position.tsg][0],current_time)
                    self.current_hardware_state['power'] = power
                    self.current_hardware_state['sense_power'] = sense_power
                    self.current_hardware_state['time'] = current_time
                elif delta_position[0] != 0:
                    self.logger.debug(f'切换plane，重置Tsg{time_position.tsg} Plane{time_position.plane-1}，建立Tsg{time_position.tsg+1} Plane{time_position.plane-1}')
                    self.instance_tsg[time_position.tsg][time_position.plane-1].reset()
                    if time_position.tsg+1<self.tsg_per_block:
                        _=self._check(self.instance_tsg[time_position.tsg+1][time_position.plane-1],current_time)#提前建立，不计入前向时间
                    self.current_hardware_state['power'] = power
                    self.current_hardware_state['sense_power'] = sense_power
                    self.current_hardware_state['time'] = current_time
                    self.logger.debug(f'当前位置索引：Block组{time_position.block_epochs} layer{time_position.layer} tsg{time_position.tsg} plane{time_position.plane}')
                if count%24==0 and count!=0:
                    self.logger.debug(f'计算完成，当前功耗{power}，当前时间{current_time}')
                    self.logger.debug(f'------------------------------------------------')
                # 检查是否完成计算
                stop_flag = self._check_computation_completed_time(matrix, time_position)
    def _check(self,instance_list, compute_time):
        """检查并激活""" 
        ready_flag=False
        # 检查TSG的状态
        if not isinstance(instance_list,list):
            instance_list=[instance_list]
        for instance in instance_list:
            activated = instance.activate_state
            if activated:
                # 已激活
                if not instance.state_check(compute_time):
                    # 未准备好，等待完成
                    wait_time = instance.end
                    self.logger.debug(
                        f"等待{instance.instance_name}建立完成，执行时间{compute_time}，完成时间{wait_time}"
                    )
                else:
                    wait_time=compute_time
                    self.logger.debug(
                        f"{instance.instance_name}在时间{compute_time}处已准备完毕"
                    )
                    ready_flag=True
            else:
                # 未激活
                instance.activate(compute_time)
                wait_time = instance.end
                if instance.instance_name[:3]=='TSG':
                    self.logger.debug(
                        f"提前建立{instance.instance_name}，执行时间{compute_time}，完成时间{wait_time}"
                    )
                else:
                    self.logger.debug(
                        f"{instance.instance_name}未开始建立，执行时间{compute_time}，完成时间{wait_time}，需要等待建立完成"
                    )
            

        return wait_time,ready_flag#因为是同一时间建立，消耗时间也相同，所以只需返回一个值

    def _get_matrix_position(self, matrix):
        """获取矩阵的起始位置"""
        if matrix not in self.matrix_locations:
            raise ValueError(f"矩阵 '{matrix}' 未加载到硬件模拟器")
            
        start_dict = self.matrix_locations[matrix]['start']
        return MatrixPosition.from_dict(start_dict)

    def _process_computation(self, matrix, bl_voltages, position, result_dict):
        """处理计算逻辑"""
        # 检查是否可以在单个平面内完成计算
        if self.bitlines_per_plane - position.bl >= bl_voltages.shape[0]:
            # 在单个平面内计算
            result_dict = self.send2register(
                matrix, bl_voltages, position.plane, position.layer,
                position.block, position.tsg, position.bl, position.block_epochs,
                result_dict
            )
            return result_dict
        else:
            # 需要跨平面计算
            # 计算当前平面可以容纳的数据量
            rest_bl_perplane = self.bitlines_per_plane - position.bl
            
            # 计算需要多少完整平面和剩余部分
            complete_plane_num, last_rest_bl = divmod(
                (bl_voltages.shape[0] - rest_bl_perplane), 
                self.bitlines_per_plane
            )
            
            # 在当前平面计算第一部分
            part_sum = self.send2register(
                matrix, bl_voltages[:rest_bl_perplane],
                position.plane, position.layer, position.block,
                position.tsg, position.bl, position.block_epochs,
                result_dict, part_flag=True
            )
            
            # 在完整平面上计算
            temp_position = MatrixPosition(
                layer=position.layer,
                plane=position.plane,
                block=position.block,
                tsg=position.tsg,
                bl=position.bl,
                block_epochs=position.block_epochs
            )
            
            for i in range(complete_plane_num):
                # 移动到下一个平面
                temp_position = self._move_to_next_plane(temp_position)
                
                # 计算该平面上的部分
                part_sum = self.send2register(
                    matrix,
                    bl_voltages[rest_bl_perplane + self.bitlines_per_plane*i:
                                rest_bl_perplane + self.bitlines_per_plane*(i+1)],
                    temp_position.plane, temp_position.layer, temp_position.block,
                    temp_position.tsg, 0, temp_position.block_epochs,
                    result_dict, part_flag=True, extra_sum=part_sum
                )
            
            # 计算最后剩余部分(如果有)
            if last_rest_bl > 0:
                # 移动到下一个平面
                temp_position = self._move_to_next_plane(temp_position)
                
                # 计算最后剩余部分
                result_dict = self.send2register(
                    matrix,
                    bl_voltages[-last_rest_bl:],
                    temp_position.plane, temp_position.layer, temp_position.block,
                    temp_position.tsg, 0, temp_position.block_epochs,
                    result_dict, extra_sum=part_sum
                )
            

            return result_dict
    def _check_computation_completed(self, matrix, position):
        """检查计算是否已完成"""
        end_pos = MatrixPosition.from_dict(self.matrix_locations[matrix]['end'])
        # 比较当前位置和结束位置
        return not end_pos==position
    def _check_computation_completed_time(self, matrix, position):
        """检查计算是否已完成(时序)"""
        end_pos = MatrixPosition.from_dict(self.matrix_locations[matrix]['end'])
        
        # 比较当前位置和结束位置
        position_tuple = (position.layer, position.block_epochs, position.plane, 
                          position.block, position.tsg, position.bl)
        end_pos_tuple = (end_pos.layer, end_pos.block_epochs, end_pos.plane, 
                         end_pos.block, end_pos.tsg, end_pos.bl)
        
        # 如果position_tuple大于或等于end_pos_tuple，则返回False
        return position_tuple < end_pos_tuple

    def send2register(self, matrix, factor, plane, layer, block, tsg, bl, block_epochs, 
                     result_dict, part_flag=False, extra_sum=0):
        """
        向寄存器发送计算结果
        
        参数:
        - matrix: 矩阵名称
        - factor: 输入因子
        - plane, layer, block, tsg, bl, block_epochs: 位置参数
        - result_dict: 结果字典
        - part_flag: 是否为部分计算标志
        - extra_sum: 额外的累加和
        
        返回:
        - result_dict 或 part_sum: 根据part_flag返回结果字典或部分和
        """
        # 计算块的起始和结束索引
        block_start = block_epochs * self.blocks_per_operation
        block_end = block_start + self.blocks_per_operation
        
        # 确保电流和因子是张量
        max_current = self._ensure_tensor(self.max_current)
        factor = self._ensure_tensor(factor)
        
        # 执行乘法计算
        if self.current_variation_mode:
            # 带电流变化的乘法
            multiplication_result = self.cell_array[
                plane, layer, block_start:block_end, tsg, bl:bl+factor.shape[0]
            ] * factor * max_current
        else:
            # 标准乘法
            multiplication_result = self.cell_array[
                plane, layer, block_start:block_end, tsg, bl:bl+factor.shape[0]
            ] * factor
        
        # 计算激活和
        if self.current_variation_mode:
            # 带电流变化的激活和计算
            activation_sums = multiplication_result.reshape(
                multiplication_result.shape[0], -1, self.max_current_sum
            )
            activation_sums = torch.round(
                activation_sums.sum(axis=2) / max_current
            ).to(device='cpu').numpy().astype(np.uint8)
        else:
            # 标准激活和计算
            activation_sums = multiplication_result.reshape(
                multiplication_result.shape[0], -1, self.max_current_sum
            ).sum(axis=2).to(device='cpu').numpy().astype(np.uint8)
        
        # 计算最终激活和
        sum_weight = np.repeat(
            self.activation_sums_weight, 
            factor.shape[0] // self.max_current_sum // self.quantization_bits, 
            axis=0
        )
        
        final_activation_sum = torch.tensor(
            activation_sums.reshape(self.blocks_per_operation // self.quantization_bits, -1),
            dtype=torch.int64,
            device=self.device
        ) * torch.tensor(sum_weight.reshape(-1), dtype=torch.int64, device=self.device)
        
        part_sum = np.array([final_activation_sum.sum(axis=1).to(device='cpu').numpy()]) + extra_sum
        
        if not part_flag:
            # 更新结果字典
            if matrix not in result_dict:
                result_dict[matrix] = part_sum
            else:
                result_dict[matrix] = np.concatenate((result_dict[matrix], part_sum), axis=1)
            return result_dict
        else:
            # 返回部分和
            return part_sum

    def _ensure_tensor(self, value):
        """确保值是张量"""
        if not isinstance(value, torch.Tensor):
            return torch.tensor(value, device=self.device)
        return value
