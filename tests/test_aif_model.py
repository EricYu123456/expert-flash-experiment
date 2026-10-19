"""Conservation, rounding and schedule sanity checks for experiment validity."""
import importlib.util
from pathlib import Path
import sys
import unittest

spec = importlib.util.spec_from_file_location('aif_model', Path(__file__).parents[1]/'scripts/aif_model.py')
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


class ModelTests(unittest.TestCase):
    def test_row_partition_conserves_weights_with_remainder(self):
        for g in (1,2,4,8,16):
            d=m.phase_demand([2,7],[(19,257)],g)
            self.assertEqual(sum(x.useful for x in d),2*19*257)

    def test_tiny_stream_touches_all_chips_but_underfills_planes(self):
        d=m.phase_demand([0],[(512,1024)],16)
        self.assertEqual(sum(x.useful>0 for x in d),16)
        self.assertEqual(sum(x.pages for x in d),32)
        self.assertEqual(sum(x.rounds for x in d),16)
        self.assertEqual(max(x.service_ns for x in d),28000)

    def test_physical_run_bundles_precharge_and_final_discharge(self):
        d=m.phase_demand([0],[(192,1024)],1)[0]
        self.assertEqual(d.rounds,3)
        self.assertEqual(d.cold,1)
        self.assertEqual(d.service_ns,28000+2*9700)
        d=m.Demand(); d.add_stream(193*4*16384,192,1)
        self.assertEqual(d.cold,2)

    def test_conservative_two_cold_sensitivity_and_block_boundaries(self):
        d=m.Demand(); d.add_stream(3*4*16384,192,2)
        self.assertEqual(d.cold,2)
        self.assertEqual(d.service_ns,3*9700+2*18300)
        d=m.Demand(); d.add_stream(193*4*16384,192,2)
        self.assertEqual(d.cold,3)

    def test_static_group_collisions_matter(self):
        balanced=m.phase_demand([0,1,2,3],[(1024,2048)],4)
        collided=m.phase_demand([0,4,8,12],[(1024,2048)],4)
        self.assertEqual(sum(x.useful for x in balanced),sum(x.useful for x in collided))
        self.assertEqual(max(x.service_ns for x in collided),4*max(x.service_ns for x in balanced))

    def test_wire_size(self):
        self.assertEqual(len(m.encode_command([m.Demand() for _ in range(16)])),400)

    def test_gate_up_stacking_conserves_math_and_amortizes_read_start(self):
        events=[dict(prompt='fixture',token_index=0,layer=0,expert_ids=list(range(8)))]
        dims={0:dict(gate=(512,1024),up=(512,1024),down=(1024,512))}
        original=m.evaluate(events,dims,16)
        stacked=m.evaluate(events,dims,16,stack_gate_up=True)
        self.assertEqual(original['useful_bytes'],stacked['useful_bytes'])
        self.assertEqual(stacked['commands'],2)
        self.assertEqual(original['commands'],3)
        self.assertEqual(stacked['plane_slot_fill'],0.75)
        self.assertEqual(stacked['model_ns']*3,original['model_ns']*2)

    def test_bandwidth_bound(self):
        d=m.Demand(); d.add_stream(64*1024*192*100,192,1)
        self.assertGreaterEqual(d.service_ns,d.useful*1e9/m.CHIP_BW)


if __name__=='__main__':
    unittest.main()
