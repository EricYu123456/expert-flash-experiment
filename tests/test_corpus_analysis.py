"""Check split isolation and accelerated timing against the scalar model."""
from collections import Counter
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from build_prompt_corpus import build_records
import corpus_analysis as c
import overhead_model as o
from export_corpus_replays import export


class CorpusTests(unittest.TestCase):
    def test_stratified_disjoint_prompt_split(self):
        rows=build_records()
        self.assertEqual(Counter(r['split'] for r in rows),{'train':200,'test':100})
        self.assertEqual(Counter(r['task'] for r in rows),{t:50 for t in
            ('qa','programming','mathematics','summarization','translation','long_context')})
        self.assertEqual(len({r['prompt_sha256'] for r in rows}),300)
        self.assertEqual(len({r['scenario_id'] for r in rows}),300)
        for s in ('train','test'):
            self.assertEqual(len({(r['task'],r['length_bucket']) for r in rows if r['split']==s}),18)
        self.assertEqual(rows,build_records())

    def test_all_timing_components_match_scalar_with_shard_remainders(self):
        rng=np.random.default_rng(100)
        for hidden,intermediate in ((1024,512),(2048,768),(4096,14336),(259,515)):
            dims={3:dict(gate=(intermediate,hidden),up=(intermediate,hidden),down=(hidden,intermediate))}
            routes=np.stack([rng.choice(32,8,replace=False) for _ in range(17)])
            placement=rng.permutation(32).tolist()
            events=[dict(prompt='fixture',token_index=i,layer=3,expert_ids=r.tolist()) for i,r in enumerate(routes)]
            for g in c.GROUPS:
                for stacked in (False,True):
                    fast=c.layer_costs(routes,placement,g,c.profiles(dims[3],g,stacked))
                    exact=o.evaluate(events,dims,g,stack_gate_up=stacked,mappings={3:placement})
                    for k in c.COMPONENTS:
                        self.assertEqual(int(fast[k].sum()),exact[k],(hidden,intermediate,g,stacked,k))
                    for i,event in enumerate(events):
                        scalar=o.evaluate([event],dims,g,stack_gate_up=stacked,mappings={3:placement})
                        self.assertEqual(sum(int(v[i]) for v in fast.values()),scalar['model_ns'])

    def test_weighted_swaps_preserve_capacity_and_never_worsen_training_peak(self):
        routes=np.array([[0,4,8,12]]*17+[[1,5,9,13]]*5+[[2,6,10,14]]*3)
        placement,stats=c.optimize_weighted(routes,16,4,512,7)
        self.assertEqual(sorted(placement),list(range(16)))
        self.assertEqual(Counter(x%4 for x in placement),{0:4,1:4,2:4,3:4})
        self.assertLess(stats['final_peak_sum'],stats['initial_peak_sum'])
        self.assertEqual((placement,stats),c.optimize_weighted(routes,16,4,512,7))

    def test_prompt_level_interval_keeps_zero_gain_zero(self):
        baseline=np.arange(100,dtype=float)+100
        self.assertEqual(c.paired_interval(baseline,baseline),[0.,0.])

    def test_cached_replay_preserves_every_descriptor_and_counter(self):
        rng=np.random.default_rng(12)
        routes=np.stack([rng.choice(16,6,replace=False) for _ in range(12)]).reshape(2,3,2,6)
        layers=[1,3]
        dimensions={layer:dict(gate=(515,259),up=(515,259),down=(259,515)) for layer in layers}
        mappings={str(layer):rng.permutation(16).tolist() for layer in layers}
        events=[dict(prompt=f'p{i}',token_index=32+t,layer=layer,expert_ids=routes[i,t,li].tolist())
                for i in range(2) for t in range(3) for li,layer in enumerate(layers)]
        with tempfile.TemporaryDirectory() as temporary:
            for g in (2,4,8,16):
                for stacked in (False,True):
                    actual=Path(temporary)/'cached.bin';reference=Path(temporary)/'scalar.bin'
                    cached=export(routes,['p0','p1'],layers,dimensions,g,stacked,mappings,actual,32)
                    scalar=o.evaluate(events,dimensions,g,stack_gate_up=stacked,
                                      mappings={int(k):v for k,v in mappings.items()},binary_path=reference)
                    self.assertEqual(actual.read_bytes(),reference.read_bytes())
                    for k in ('commands','model_ns','useful_bytes','read_rounds','cold_reads',
                              'chip_bytes','chip_service_ns_by_chip',*c.COMPONENTS):
                        self.assertEqual(cached[k],scalar[k],(g,stacked,k))


if __name__=='__main__':
    unittest.main()
