#!/usr/bin/env python3
"""Config-only arithmetic screen; no weights, route synthesis or quality claim."""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHIPS, PAGE, PLANES = 16, 16384, 4


def ceildiv(a, b):
    return (a + b - 1) // b


def phase(rows, columns, active, experts):
    result = []
    for g in range(1, CHIPS + 1):
        shard_rows = [rows // g + (c < rows % g) for c in range(g)]
        pages = [ceildiv(r * columns, PAGE) for r in shard_rows]
        waves = [ceildiv(p, PLANES) for p in pages]
        one = len(set(waves)) == 1
        result.append(dict(g=g, power_of_two=(g & (g - 1)) == 0,
            rows_per_shard=shard_rows, pages_per_expert=sum(pages), waves_per_expert=sum(waves),
            useful_wave_equivalent=rows * columns / (PAGE * PLANES),
            per_request_total_work_lb=ceildiv(active * sum(waves), CHIPS),
            identical_job_lattice_lb=ceildiv(active * g, CHIPS) * waves[0] if one else None,
            equal_shard_wave_length=waves[0] if one else None,
            equal_capacity_assignment_count_integral=(experts * g) % CHIPS == 0))
    return dict(rows=rows, columns=columns, plans=result,
        scope='Equal or floor/ceil-balanced whole-output-row shards. Bounds allow different owners per request and are not a feasible fixed plan or predicted token latency.')


def main():
    records = json.loads((HERE / 'model_configs.json').read_text())
    out = []
    for rec in records:
        cfg = json.loads(Path(rec['config_path']).read_text())
        cfg = cfg.get('text_config', cfg)
        hidden = cfg['hidden_size']
        width = cfg.get('moe_latent_size') or hidden
        middle = cfg.get('moe_intermediate_size', cfg.get('intermediate_size'))
        experts = cfg.get('num_local_experts', cfg.get('num_experts', cfg.get('n_routed_experts')))
        active = cfg.get('num_experts_per_tok', cfg.get('num_experts_per_token'))
        if not experts or not active:
            out.append(dict(model=rec['model'], status='Skipped: explicit routed E/K not found; do not infer.'))
            continue
        # NVIDIA's released NemotronH experts use one relu2 up matrix, not SwiGLU.
        factor = 1 if cfg.get('mlp_hidden_act') == 'relu2' else 2
        out.append(dict(model=rec['model'], source_revision=rec['revision'], source_url=rec['config_url'],
            config_sha256=rec['config_sha256'], hidden=hidden, expert_input_width=width,
            intermediate=middle, experts=experts, active=active, up_matrix_count=factor,
            latent_projection_not_counted=(width != hidden),
            up=phase(factor * middle, width, active, experts),
            down=phase(width, middle, active, experts)))
    result = dict(scope='Existing experiment geometry only: fixed 1-byte weights, ordinary 4-plane 16KiB reads, 16 chips. No new precision method is proposed.',
        important_limits=['Published checkpoint storage need not be INT8; this arithmetic does not establish that its native format can execute on AiF.',
            'No new routes/masks are generated here. Config dimensions alone cannot predict static-layout benefits.',
            'Input/output traffic, PE/ECC throughput, dense/shared/latent projections and launch overhead are excluded.',
            'Fluid byte bounds can be unattainable due to job granularity. The identical-job lattice bound is reported separately.'],
        models=out)
    (HERE / 'geometry_bounds.json').write_text(json.dumps(result, indent=2) + '\n')
    for m in out:
        if 'up' not in m:
            print(m['model'], m['status'])
            continue
        descriptions = []
        for name in ('up', 'down'):
            plans=m[name]['plans']
            valid=[p for p in plans if p['identical_job_lattice_lb'] is not None]
            p2=min((p for p in valid if p['power_of_two']), key=lambda p:p['identical_job_lattice_lb'])
            anyg=min(valid, key=lambda p:p['identical_job_lattice_lb'])
            descriptions.append(f'{name}: bestP2 oracle g{p2["g"]}/{p2["identical_job_lattice_lb"]}w, anyg{anyg["g"]}/{anyg["identical_job_lattice_lb"]}w')
        print(m['model'], '; '.join(descriptions))


if __name__ == '__main__':
    main()
