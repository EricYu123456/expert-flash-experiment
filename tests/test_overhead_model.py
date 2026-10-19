import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import aif_model as f
import overhead_model as o


class OverheadTests(unittest.TestCase):
    def test_shared_channel_is_a_bottleneck(self):
        v=[0]*16; v[0]=100; v[8]=100
        self.assertEqual(o.transfer_ns(v),100)
        v[8]=0; v[1]=100
        self.assertEqual(o.transfer_ns(v),50)

    def test_common_input_reuse_preserves_output_bytes(self):
        shared=o.phase_traffic(list(range(8)),[(512,1024)],16,True)
        separate=o.phase_traffic(list(range(8)),[(512,1024)],16,False)
        self.assertEqual(sum(shared[0]),16*1024)
        self.assertEqual(sum(separate[0]),8*16*1024)
        self.assertEqual(sum(shared[1]),8*512*4)
        self.assertEqual(shared[1],separate[1])

    def test_controller_fanout_charges_one_host_input(self):
        demands=f.phase_demand([0,1],[(512,1024)],16)
        vi,vo,streams=o.phase_traffic([0,1],[(512,1024)],16,True)
        self.assertLess(o.transfer_ns(vi,1024),o.transfer_ns(vi))
        self.assertEqual(len(o.encode_v2(demands,vi,vo,streams,250,2000,1024,sum(vo))),800)

    def test_block_split_creates_additional_chip_commands(self):
        _,_,streams=o.phase_traffic([0],[(193*4,16384)],1,True)
        self.assertEqual(sum(streams),2)

    def test_wire_size_and_additive_cost(self):
        demands=f.phase_demand([0,1],[(512,1024)],4)
        vi,vo,streams=o.phase_traffic([0,1],[(512,1024)],4,True)
        payload=o.encode_v2(demands,vi,vo,streams,250,2000)
        self.assertEqual(len(payload),800)
        events=[dict(prompt='fixture',token_index=0,layer=0,expert_ids=[0,1])]
        dims={0:dict(gate=(512,1024),up=(512,1024),down=(1024,512))}
        result=o.evaluate(events,dims,4)
        self.assertEqual(result['model_ns'],sum(result[k] for k in ('flash_ns','control_ns','host_ns','input_ns','output_ns')))
        self.assertEqual(result['useful_bytes'],2*3*512*1024)
        alternative=o.sweep_cost(result,1000,10000)
        direct=o.evaluate(events,dims,4,control_ns=1000,host_ns=10000)
        self.assertEqual(alternative['model_ns'],direct['model_ns'])


if __name__=='__main__':
    unittest.main()
