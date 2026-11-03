import yaml
import matplotlib.pyplot as plt
import numpy as np
import re
import torch
import matplotlib
import os  # 添加os模块用于目录操作
import pandas as pd  # 添加pandas用于Excel操作
matplotlib.use('Agg')  # 使用非交互式后端，适合在多线程环境下使用
# 配置matplotlib
matplotlib.rcParams['font.family'] = 'Arial'  # 设置Arial字体
matplotlib.rcParams['font.weight'] = 'bold'   # 设置字体加粗
matplotlib.rcParams['font.size'] = 10         # 设置字体大小为10
matplotlib.rcParams['axes.unicode_minus'] = False  # 保持负号显示正常

# 添加坐标轴设置
matplotlib.rcParams['axes.spines.left'] = True          # 显示左侧坐标轴线
matplotlib.rcParams['axes.spines.bottom'] = True        # 显示底部坐标轴线
matplotlib.rcParams['axes.spines.top'] = True          # 显示顶部坐标轴线
matplotlib.rcParams['axes.spines.right'] = True        # 显示右侧坐标轴线
matplotlib.rcParams['axes.linewidth'] = 2.0
# 添加刻度线设置
matplotlib.rcParams['xtick.direction'] = 'in'  # x轴刻度线向内
matplotlib.rcParams['ytick.direction'] = 'in'  # y轴刻度线向内
matplotlib.rcParams['xtick.minor.visible'] = True  # 显示x轴次刻度线
matplotlib.rcParams['ytick.minor.visible'] = True  # 显示y轴次刻度线
matplotlib.rcParams['xtick.top'] = True               # 是否在顶部显示x轴刻度线
matplotlib.rcParams['ytick.right'] = True             # 是否在右侧显示y轴刻度线
# 设置图形保存分辨率
matplotlib.rcParams['savefig.dpi'] = 600
# 设置图形大小和分辨率
matplotlib.rcParams['figure.figsize'] = [5,5]     # 图形大小，单位英寸
matplotlib.rcParams['figure.dpi'] = 600                # 图形分辨率
def parse_metrics_file(file_path):
    """解析metrics文件，处理可能的编码问题"""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        # 尝试移除BOM标记和其他可能的特殊字符
        content = re.sub(r'^\ufeff|\ufeff', '', content)
        # 解析YAML格式数据
        try:
            metrics = yaml.safe_load(content)
            return metrics
        except Exception as e:
            print(f"解析文件失败: {e}")
            return None

def compare_metrics(file1_path, file2_path, target_layers, metrics_to_compare):
    """比较两个文件中指定层的指定指标"""
    # 解析两个文件
    metrics1 = parse_metrics_file(file1_path)
    metrics2 = parse_metrics_file(file2_path)
    
    if not metrics1 or not metrics2:
        print("文件解析失败")
        return None
    
    # 获取数据的第一个键（通常是"0"）
    key1 = list(metrics1.keys())[0]
    key2 = list(metrics2.keys())[0]
    
    # 存储比较结果
    comparison = {}
    
    for layer in target_layers:
        if layer in metrics1[key1] and layer in metrics2[key2]:
            layer_comparison = {}
            for metric in metrics_to_compare:
                if metric in metrics1[key1][layer] and metric in metrics2[key2][layer]:
                    value1 = metrics1[key1][layer][metric]
                    value2 = metrics2[key2][layer][metric]
                    layer_comparison[metric] = {
                        "文件1": value1,
                        "文件2": value2,
                        "差值": value2 - value1,
                        "相对差异(%)": ((value2 - value1) / value1 * 100) if value1 != 0 else float('nan')
                    }
            comparison[layer] = layer_comparison
    
    return comparison

def save_data_to_excel(comparison, save_dir):
    """将比较结果保存为Excel文件"""
    # 创建一个ExcelWriter对象
    excel_path = os.path.join(save_dir, "metrics_comparison.xlsx")
    with pd.ExcelWriter(excel_path) as writer:
        # 对每个指标创建单独的工作表
        metrics_set = set()
        for layer_data in comparison.values():
            for metric in layer_data.keys():
                metrics_set.add(metric)
        
        for metric in metrics_set:
            # 为当前指标创建数据
            data = []
            for layer, layer_data in comparison.items():
                if metric in layer_data:
                    row = {"层": layer}
                    row.update(layer_data[metric])
                    data.append(row)
            
            # 创建DataFrame并写入Excel
            if data:
                df = pd.DataFrame(data)
                df.set_index("层", inplace=True)
                df.to_excel(writer, sheet_name=metric)
    
    print(f"比较数据已保存到: {excel_path}")

def visualize_comparison(comparison, metrics_to_compare, save_dir):
    """使用柱形图可视化比较结果"""
    layers = list(comparison.keys())
    
    for metric in metrics_to_compare:
        plt.figure(figsize=(12, 6))
        
        file1_values = []
        file2_values = []
        
        for layer in layers:
            if metric in comparison[layer]:
                file1_values.append(comparison[layer][metric]["文件1"])
                file2_values.append(comparison[layer][metric]["文件2"])
            else:
                file1_values.append(0)
                file2_values.append(0)
        
        x = np.arange(len(layers))
        width = 0.35
        
        plt.bar(x - width/2, file1_values, width, label='文件1')
        plt.bar(x + width/2, file2_values, width, label='文件2')
        
        plt.xlabel('层')
        plt.ylabel(metric)
        plt.title(f'{metric} 对比')
        plt.xticks(x, layers, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        # 保存图表到error_analyze文件夹中
        save_path = os.path.join(save_dir, f"{metric}_comparison.png")
        plt.savefig(save_path)
        print(f"已保存图表: {save_path}")
        plt.close()

if __name__ == "__main__":
    # 直接在代码中指定文件路径和要比较的层、指标
    file1_path = r"E:\coda\GPT2-Hardware\Output\20250610_154337\metrics_summary.txt"  # 更改为第一个文件的实际路径
    file2_path = r"E:\coda\GPT2-Hardware\Output\20250609_154140\metrics_summary.txt" # 更改为第二个文件的实际路径
    
    # 指定要比较的层
    target_layers = [
        "layer_"+str(i)+"_mlp_fc2" for i in range(11)   # 比较第0层的fc2
    ]
    
    # 指定要比较的指标
    metrics_to_compare = ["mae", "rmse", "relative_error"]
    
    # 创建error_analyze文件夹(如果不存在)
    save_dir = "error_analyze"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"创建目录: {save_dir}")
    
    # 执行比较
    comparison = compare_metrics(file1_path, file2_path, target_layers, metrics_to_compare)
    
    if comparison:
        # 打印比较结果
        for layer, metrics in comparison.items():
            print(f"\n层: {layer}")
            for metric, values in metrics.items():
                print(f"  {metric}:")
                for key, value in values.items():
                    print(f"    {key}: {value}")
        
        # 保存数据到Excel
        save_data_to_excel(comparison, save_dir)
        
        # 可视化比较结果
        visualize_comparison(comparison, metrics_to_compare, save_dir)