#!/usr/bin/env python3
"""Fresh article-disjoint WikiText103 quality VALID200/TEST1000 inputs."""
from pathlib import Path
import hashlib
import json
import re
import urllib.request

ROOT=Path('/home/nfs_client/ericyu/expert_flash_experiment')
OLD=ROOT/'research-2026-11-08-shared-features'
OUT=ROOT/'research-2026-11-09-shared-features/corpus103'
REV='b08601e04326c79dfdd32d625aee71d232d685c3'
SEED='20261109-new-article-quality-v1'


def digest(v):return hashlib.sha256(v).hexdigest()


def write(p, b):
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():assert p.read_bytes()==b,f'Preserve {p}'
    else:p.write_bytes(b)


def title(text):
    t=text.strip()
    return t if re.fullmatch(r'= [^=].*[^=] =',t) else None


def main():
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'manifest.json').exists(),'Corpus already immutable'
    excluded=set()
    old_sources=[]
    for split in ('train','validation','test'):
        p=OLD/f'corpus/raw/{split}.parquet'
        for t in pq.read_table(p,columns=['text'])['text'].to_pylist():
            tt=title(t)
            if tt:excluded.add(tt)
        old_sources.append({'path':str(p),'sha256':digest(p.read_bytes())})
    tok=Tokenizer.from_file('/home/nfs_client/ymtuan2/models/DeepSeek-V2-Lite-Chat/tokenizer.json')
    old=json.loads((OLD/'corpus/manifest.json').read_text())
    oldprefix={tuple(tok.encode(Path(r['path']).read_text(),add_special_tokens=False).ids[:63]) for r in old['records']}
    best={}; sources=[]; article=None; rows_seen=0
    for shard in range(2):
        file=f'train-{shard:05d}-of-00002.parquet'
        url=f'https://huggingface.co/datasets/Salesforce/wikitext/resolve/{REV}/wikitext-103-raw-v1/{file}?download=true'
        raw=OUT/'raw'/file
        if not raw.exists():
            raw.parent.mkdir(parents=True,exist_ok=True)
            temporary=raw.with_suffix('.download')
            with urllib.request.urlopen(url,timeout=120) as response,temporary.open('wb') as f:
                while True:
                    block=response.read(8<<20)
                    if not block:break
                    f.write(block)
            temporary.replace(raw)
        h=hashlib.sha256()
        with raw.open('rb') as f:
            for block in iter(lambda:f.read(8<<20),b''):h.update(block)
        sources.append({'path':str(raw),'url':url,'sha256':h.hexdigest(),'bytes':raw.stat().st_size})
        rownum=0
        for batch in pq.ParquetFile(raw).iter_batches(batch_size=16384,columns=['text']):
            for text in batch.column(0).to_pylist():
                thisrow=rownum;rownum+=1;rows_seen+=1
                tt=title(text)
                if tt:article=tt;continue
                if article is None or article in excluded:continue
                stripped=text.strip()
                if len(stripped.split())<96:continue
                rank=digest(f'{SEED}\0paragraph\0{article}\0{shard}\0{thisrow}'.encode())
                if article in best and rank>=best[article]['paragraph_rank']:continue
                best[article]={'article_title':article,'source_shard':shard,'source_row':thisrow,
                               'paragraph_rank':rank,'text':stripped,
                               'article_rank':digest(f'{SEED}\0article\0{article}'.encode())}
        print(json.dumps({'parsed_shard':shard,'rows':rownum,'new_eligible_articles':len(best)}),flush=True)
    ranked=sorted(best.values(),key=lambda r:(r['article_rank'],r['article_title']))
    chosen=[];prefixes=set(oldprefix)
    for row in ranked:
        ids=tok.encode(row['text'],add_special_tokens=False).ids
        prefix=tuple(ids[:63])
        if len(ids)<65 or prefix in prefixes:continue
        prefixes.add(prefix);row['prefix63_sha256']=digest(json.dumps(list(prefix),separators=(',',':')).encode())
        chosen.append(row)
        if len(chosen)==1200:break
    assert len(chosen)==1200
    records=[]
    for i,row in enumerate(chosen):
        split='quality_validation' if i<200 else 'test'
        index=i if i<200 else i-200
        id=f'fresh103_{split}_{index:04d}'
        text=row.pop('text');p=OUT/'prompts'/f'{id}.txt';write(p,text.encode())
        records.append({'id':id,'split':split,'path':str(p),'prompt_sha256':digest(text.encode()),
                        'article_id':digest(row['article_title'].encode()),**row})
    for split,n in [('quality_validation',200),('test',1000)]:
        rs=[r for r in records if r['split']==split];assert len(rs)==n
        write(OUT/f'{split}.jobs.tsv',''.join(f"{r['path']}\t{r['id']}\n" for r in rs).encode())
        for shard in range(4):
            write(OUT/f'{split}.shard{shard}.jobs.tsv',''.join(f"{r['path']}\t{r['id']}\n" for r in rs[shard::4]).encode())
    assert len({r['article_id'] for r in records})==1200
    manifest={'schema':1,'seed':SEED,'dataset':'Salesforce/wikitext','config':'wikitext-103-raw-v1',
              'revision':REV,'publisher_split':'train','experimental_split':'held-out article-disjoint quality-validation200/test1000',
              'sources':sources,'old_sources_excluded':old_sources,'excluded_old_article_count':len(excluded),
              'eligible_new_articles':len(best),'publisher_rows_parsed':rows_seen,
              'counts':{'quality_validation':200,'test':1000},'records':records,
              'input_contract':'First64 native tokens incl BOS, score63 adjacent targets; up to16 greedy continuation tokens with EOG stop.',
              'selection':'Input-only SHA256 article ranking; one ranked >=96-word paragraph per new article; no inference outputs used.',
              'no_article_overlap_with_any_old_WikiText2_split':True,
              'all_1200_articles_distinct':True,'no_old1800_prefix63_overlap':True,
              'test_targets_forbidden_until_model_selection_and_eligibility_frozen':True,
              'source_sha256':digest(Path(__file__).read_bytes())}
    write(OUT/'manifest.json',(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n').encode())
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('records','sources')},indent=2),flush=True)


if __name__=='__main__':main()
