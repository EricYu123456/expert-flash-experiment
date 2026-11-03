#!/usr/bin/env python3
"""Read GGUF headers only; audit routing precision versus timing assumptions.

No tensor payload, routing cache, test data, or large weight download is read.
Official HF API/config snapshots are small externally fetched provenance inputs.
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import pathlib
import struct

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODELS = pathlib.Path('/home/nfs_client/ericyu/expert_flash_experiment/models')
OUT = ROOT / 'results/research-2026-11-02/round2/precision-scope'
QTYPE = {0: ('F32', 1, 4), 8: ('Q8_0', 32, 34),
         12: ('Q4_K', 256, 144), 14: ('Q6_K', 256, 210)}
SCALAR = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i',
          6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_gguf_header(path):
    with path.open('rb') as f:
        def read(fmt):
            size = struct.calcsize('<' + fmt)
            raw = f.read(size)
            assert len(raw) == size
            return struct.unpack('<' + fmt, raw)[0]

        def string():
            return f.read(read('Q')).decode('utf-8')

        def value(dtype):
            if dtype in SCALAR:
                return read(SCALAR[dtype])
            if dtype == 8:
                return string()
            assert dtype == 9, dtype
            item_type, count = read('I'), read('Q')
            return [value(item_type) for _ in range(count)]

        assert f.read(4) == b'GGUF'
        version, ntensors, nfields = read('I'), read('Q'), read('Q')
        assert version in (2, 3)
        metadata = {}
        for _ in range(nfields):
            key = string()
            metadata[key] = value(read('I'))
        tensors = []
        for _ in range(ntensors):
            name = string()
            shape = [read('Q') for _ in range(read('I'))]
            dtype, offset = read('I'), read('Q')
            label, grain, size = QTYPE[dtype]
            count = math.prod(shape)
            assert count % grain == 0
            tensors.append(dict(name=name, shape_ggml=shape, ggml_type=label,
                                n_bytes=count // grain * size,
                                data_offset=offset))
        header_bytes = f.tell()
        alignment = metadata.get('general.alignment', 32)
        payload_start = (header_bytes + alignment - 1) // alignment * alignment
        for t in tensors:
            t['relative_payload_offset'] = t['data_offset']
            t['data_offset'] += payload_start
        f.seek(0)
        header_sha = hashlib.sha256(f.read(header_bytes)).hexdigest()
    return metadata, tensors, dict(version=version, header_bytes=header_bytes,
        header_sha256=header_sha, bytes_read=2 * header_bytes,
        payload_start_offset=payload_start,
        tensor_payload_read=False, full_file_sha_recomputed=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = {}

    def record(path):
        sources[str(path)] = sha(path)

    records = []
    for model in ('granite', 'olmoe', 'deepseek', 'qwen3', 'mixtral', 'granite4tiny'):
        meta_path = MODELS / (model + '-metadata.json')
        record(meta_path)
        previous = json.loads(meta_path.read_text())
        path = pathlib.Path(previous['model_path'])
        metadata, tensors, read_audit = read_gguf_header(path)
        reference = {t['name']: t for t in previous['tensors']}
        assert len(tensors) == len(reference)
        for t in tensors:
            for key in ('shape_ggml', 'ggml_type', 'n_bytes', 'data_offset'):
                assert t[key] == reference[t['name']][key], (model, t['name'], key)
        assert path.stat().st_size == previous['model_bytes']
        projections = {}
        for projection in ('gate', 'up', 'down'):
            selected = [t for t in tensors if reference[t['name']].get('projection') == projection
                        and reference[t['name']].get('expert_count', 0) > 1
                        and '_exps.' in t['name']]
            elements = sum(math.prod(t['shape_ggml']) for t in selected)
            stored = sum(t['n_bytes'] for t in selected)
            projections[projection] = dict(tensor_type_counts=dict(collections.Counter(
                t['ggml_type'] for t in selected)), tensor_count=len(selected),
                logical_weights=elements, actual_gguf_tensor_bytes=stored,
                modeled_raw_i8_bytes=elements, actual_bytes_over_modeled_i8=stored / elements,
                example=selected[0])
        router = [t for t in tensors if '.ffn_gate_inp.weight' in t['name']]
        source_path = MODELS / (model + '-source-manifest.json')
        download_path = MODELS / (model + '-download-manifest.json')
        provenance = {}
        for p in (source_path, download_path):
            if p.exists():
                record(p)
                d = json.loads(p.read_text())
                provenance[p.name] = {k: v for k, v in d.items() if k not in ('weight_files', 'config')}
                if 'config' in d:
                    provenance['source_config_dtype'] = d['config'].get('torch_dtype')
                    provenance['source_config_quantization'] = d['config'].get('quantization_config')
                if d.get('weight_files'):
                    provenance['source_cached_revision'] = d['weight_files'][0].get('download_metadata', [None])[0]
        records.append(dict(model=model, path=str(path), model_bytes=path.stat().st_size,
            recorded_full_sha256=previous['sha256'], header_audit=read_audit,
            header_exactly_matches_existing_metadata=True,
            gguf_file_type=metadata.get('general.file_type'),
            gguf_architecture=metadata.get('general.architecture'),
            gguf_name=metadata.get('general.name'),
            all_tensor_type_counts=dict(collections.Counter(t['ggml_type'] for t in tensors)),
            routed_projections=projections,
            routed_router_type_counts=dict(collections.Counter(t['ggml_type'] for t in router)),
            provenance=provenance, independently_validated_native_aif_w8a8_routing=False))

    official = []
    for p in sorted(OUT.glob('*.api.json')):
        record(p)
        wrapper = json.loads(p.read_text())
        d = wrapper['response']
        selected = [s for s in d.get('siblings', []) if any(x in s['rfilename'].lower()
                    for x in ('q8_0', 'bf16.gguf', '-f16.gguf'))]
        official.append(dict(repo=d['id'], request_url=wrapper['request_url'],
            revision=d['sha'], safetensors_parameter_dtype_counts=d.get('safetensors'),
            relevant_gguf_files=selected))
    for p in sorted(OUT.glob('*.config.json')):
        record(p)
    for relative in ('kernel/aif.c', 'kernel/aif.h', 'scripts/aif_model.py',
                     'scripts/joint_plane_search.py', 'scripts/llama_corpus_trace.cpp',
                     'scripts/validate_converted_weights.py', 'paper/aif.txt', 'README.md'):
        record(ROOT / relative)
    llama = pathlib.Path('/home/nfs_client/ericyu/expert_flash_experiment/sources/llama.cpp-927dada6c')
    for relative in ('ggml/src/ggml-common.h', 'ggml/src/ggml-cpu/ggml-cpu.c'):
        record(llama / relative)
    report = dict(scope='Documentation audit only; no method, rerouting, test-data read, or timing-result modification.',
        model_records=records, official_metadata_snapshots=official,
        gguf_q8_0_contract=dict(block_weights=32, signed_quant_bytes=32,
            fp16_scale_bytes=2, packed_bytes=34, scale_byte_overhead_vs_raw_i8=.0625,
            cpu_dot='sum_b scaleW[row,b]*scaleX[b]*integer_dot(qW[row,b],qX[b])',
            caveat='CPU implementation floating reduction/rounding order still matters for bit identity.'),
        modeled_contract=dict(weights='one byte per logical weight; no explicit scale bytes',
            inputs='one byte per vector component in timing metadata, not exported activation values',
            outputs='four bytes per output element in modeled transfers; not evidence of native dtype',
            backend='CPU llama.cpp generates routes, separate metadata-only AiF/NVMeVirt timing replay',
            measured_native_end_to_end_numeric_equivalence=False),
        native_published_contract=dict(proven=['INT8 multipliers and adder tree',
            'dense GEMV input dimension and matrix start/block/offset/page addressing',
            'sequential output FIFO', 'host output buffer address',
            'host/SSD partitioning and aggregation of projection submatrices'],
            unspecified=['accumulator width and overflow', 'FIFO element dtype/width',
                'block/per-row scale storage and arithmetic', 'rounding/requantization order'],
            local_api='kernel/aif.h explicitly says metadata only: no weights or GEMV outputs; output_bytes is a count.'),
        existing_operator_composition=dict(status='Dense column tiles plus host partial sums use existing mathematical GEMV; no new PE operator follows merely from this composition.',
            exact_condition='If a native 512-wide signed-I8 dot is returned as its unscaled integer in exact I32 or FP32, abs(dot)<=512*128*128=2^23; FP32 holds every such integer exactly. Host I64 summation then one specified rounding can reproduce a full raw-I8 dot.',
            warning='Published output semantics do not establish this condition. Scaled/requantized per-tile outputs need a different proof. Host partial sums or FP reordering were not prohibited by the human user.'),
        honest_claims=['Trace-driven physical-layout comparison under an explicit INT8 architectural cost model.',
            'Routes come from the named real, fixed, quantized llama.cpp checkpoints, not from timing dummy outputs.',
            'Complete-row relocation preserves each abstract GEMV dot without changing its reduction; numerical AiF execution of the routing checkpoint was not demonstrated.',
            'Do not claim same-checkpoint native W8A8 end-to-end accuracy, hardware timing, or quality preservation from metadata-only replay.',
            'Q8_0 revalidation tests routing precision sensitivity, not raw-I8/native AiF numerical equivalence.'],
        minimum_followup_not_run=[
            'First request or locate published/native output dtype, accumulation, scale, rounding and I/O contract; do not infer it from four-byte modeled outputs.',
            'For a cheap routing sensitivity control, freeze existing G4 layouts and pinned IBM Q8_0 revision/hash; collect the same 32 existing training prompts for 64 decode steps plus a prospectively disjoint public set whose IDs and scoring are frozen before reading outputs.',
            'Use independent greedy generation for each precision and a teacher-forced common token sequence where supported to separate token divergence from routing sensitivity. Record checkpoint-specific traced/untraced token equality and top-k overlap.',
            'Replay all frozen layouts on both route sets without fitting on public prompts; report robustness rather than AiF-matched accuracy.',
            'To claim exact native execution, implement an operator-level numerical reference matching the proven native contract and chosen checkpoint scales, then a full-model matched compute path with quality evaluation. Q8_0 alone does not supply that contract; split-by-32 host scaling is possible composition only after all extra operations and rounding are accounted for.'
        ],
        input_sha256=sources, script_sha256=sha(pathlib.Path(__file__)))
    (OUT / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    for r in records:
        print(r['model'], r['all_tensor_type_counts'],
              {k: v['tensor_type_counts'] for k, v in r['routed_projections'].items()},
              'header_bytes', r['header_audit']['header_bytes'])


if __name__ == '__main__':
    main()
