#!/usr/bin/env python3
"""Build immutable, official-split WikiText-2 inputs for CPU shared-GU research."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import urllib.request

REVISION = 'b08601e04326c79dfdd32d625aee71d232d685c3'
SEED = '20261108-shared-features-v1'
COUNTS = {'train': 600, 'validation': 200, 'test': 1000}
ROOT = Path('/home/nfs_client/ericyu/expert_flash_experiment/research-2026-11-08-shared-features')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f'Refusing to overwrite different content: {path}')
    else:
        path.write_bytes(data)


def build(out):
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file('/home/nfs_client/ymtuan2/models/DeepSeek-V2-Lite-Chat/tokenizer.json')
    sources, records, seen_text, seen_prefix = [], [], set(), set()
    for split, count in COUNTS.items():
        url = f'https://huggingface.co/datasets/Salesforce/wikitext/resolve/{REVISION}/wikitext-2-raw-v1/{split}-00000-of-00001.parquet?download=true'
        raw = out / 'raw' / f'{split}.parquet'
        if not raw.exists():
            with urllib.request.urlopen(url, timeout=90) as response:
                write_new(raw, response.read())
        payload = raw.read_bytes()
        if split == 'test':
            assert digest(payload) == '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'
        rows = pq.read_table(raw)['text'].to_pylist()
        eligible, article, title = [], -1, ''
        for row_index, text in enumerate(rows):
            stripped = text.strip()
            if re.fullmatch(r'= [^=].*[^=] =', stripped):
                article += 1
                title = stripped
            if len(stripped.split()) < 32 or re.fullmatch(r'=+.*=+', stripped):
                continue
            encoded = tokenizer.encode(stripped, add_special_tokens=False).ids
            if len(encoded) < 33:
                continue
            text_hash = digest(stripped.encode())
            prefix_hash = digest(json.dumps(encoded[:15], separators=(',', ':')).encode())
            # Deduplication is input-only. No activations, targets or fit results
            # enter corpus construction. It also protects the 16-token context.
            if text_hash in seen_text or prefix_hash in seen_prefix:
                continue
            eligible.append(dict(source_row=row_index, article_id=f'{split}:{article}',
                                 article_title=title, text=stripped, prompt_sha256=text_hash,
                                 prefix15_sha256=prefix_hash,
                                 rank=digest(f'{SEED}\0{split}\0{row_index}'.encode())))
        eligible.sort(key=lambda r: (r['rank'], r['source_row']))
        chosen = []
        for row in eligible:
            if row['prompt_sha256'] in seen_text or row['prefix15_sha256'] in seen_prefix:
                continue
            seen_text.add(row['prompt_sha256'])
            seen_prefix.add(row['prefix15_sha256'])
            chosen.append(row)
            if len(chosen) == count:
                break
        if len(chosen) != count:
            raise RuntimeError(f'Only {len(chosen)} eligible paragraphs in {split}; need {count}')
        split_records = []
        for i, row in enumerate(chosen):
            prompt_id = f'wikitext2_{split}_{i:04d}'
            prompt_path = out / 'prompts' / f'{prompt_id}.txt'
            text = row.pop('text')
            write_new(prompt_path, text.encode())
            split_records.append(dict(id=prompt_id, split=split, path=str(prompt_path), **row))
        write_new(out / f'{split}.jobs.tsv', ''.join(f"{r['path']}\t{r['id']}\n" for r in split_records).encode())
        for shard in range(4):
            write_new(out / f'{split}.shard{shard}.jobs.tsv', ''.join(f"{r['path']}\t{r['id']}\n" for r in split_records[shard::4]).encode())
        records.extend(split_records)
        sources.append(dict(split=split, url=url, path=str(raw), sha256=digest(payload),
                            source_rows=len(rows), eligible_paragraphs=len(eligible), selected=count))
    manifest = dict(schema_version=1, seed=SEED, dataset='Salesforce/wikitext', revision=REVISION,
                    config='wikitext-2-raw-v1', sources=sources, counts=COUNTS,
                    articles_by_split={s: len({r['article_id'] for r in records if r['split'] == s}) for s in COUNTS},
                    records=records, prompt_disjoint=True, prefix15_disjoint=True,
                    selection='SHA256(seed,NUL,official split,NUL,row index); first requested count after input-only eligibility/deduplication',
                    context='Runner retains first16 native tokens incl BOS; target token17 is reserved for NLL. Full paragraph files retained. Short-context mechanism screen, not full benchmark PPL.',
                    limitations=['English encyclopedic paragraphs, not a multitask instruction accuracy benchmark',
                                 'Paragraphs within an article are correlated; cluster quality intervals by source article',
                                 'Test split has appeared in prior project experiments; no checkpoint or readout training on it in this run'],
                    source_code_sha256=digest(Path(__file__).read_bytes()))
    write_new(out / 'manifest.json', (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode())
    print(json.dumps({k: v for k, v in manifest.items() if k != 'records'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'corpus')
    args = parser.parse_args()
    build(args.output)
