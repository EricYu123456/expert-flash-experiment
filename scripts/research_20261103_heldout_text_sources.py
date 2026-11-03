#!/usr/bin/env python3
"""Freeze then acquire bounded publisher-pinned text, without model fitting.

The two commands are deliberately separate. `freeze` never opens target
Parquet payloads. `acquire` requires the exact frozen source/config hashes.
Prompts and answers are never printed by this script.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import urllib.parse
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'results/research-2026-11-02/round2/fresh-format-heldout'
PLACEMENT=ROOT/'results/research-2026-11-02/round2/geometry-study/granite4tiny.frozen.json'
PLACEMENT_SHA='a55f17010b7b48c6c396dee92bb1d127a541e28a93b94aee2e42aac9f3dc2538'
NUMERIC=ROOT/'results/research-2026-11-02/round2/full-ffn-integer-reference/frozen_protocol.json'
NUMERIC_SHA='3163341827c25b210c42126ae7465c1fe8be9827aab93d314a89ef8a54975f40'
MODEL=Path('/home/nfs_client/ericyu/expert_flash_experiment/models/granite4tiny-q4_k_m.gguf')
MODEL_SHA='5a38b08c441ae1adbafb1d2b8a7167e0d48734d83af68b268cefea1eec553dcd'
MODEL_MANIFEST=MODEL.with_name('granite4tiny-source-manifest.json')
SEED=20261103
SOURCES=[
    dict(key='wikitext2_raw_test',repo='Salesforce/wikitext',
         revision='b08601e04326c79dfdd32d625aee71d232d685c3',config='wikitext-2-raw-v1',split='test',kind='lm',
         file='wikitext-2-raw-v1/test-00000-of-00001.parquet',bytes=732610,
         expected_sha256='5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91',
         license_as_publisher_card=['cc-by-sa-3.0','gfdl']),
    dict(key='arc_challenge_validation',repo='allenai/ai2_arc',
         revision='210d026faf9955653af8916fad021475a3f00453',config='ARC-Challenge',split='validation',kind='arc',
         file='ARC-Challenge/validation-00000-of-00001.parquet',bytes=55743,
         expected_sha256='395a5c88d1580d69855fbaee9450270578df1ad5af6259771cd0a42c20e99f05',
         license_as_publisher_card=['cc-by-sa-4.0'])]


def now():return datetime.now(timezone.utc).isoformat()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def textsha(text):return hashlib.sha256(text.encode('utf-8')).hexdigest()


def write_new(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:f.write(data)


def json_new(path,data):
    write_new(path,(json.dumps(data,ensure_ascii=False,indent=2)+'\n').encode())


def get(url,limit):
    req=urllib.request.Request(url,headers={'User-Agent':'expert-flash-research-source-audit/1.0'})
    with urllib.request.urlopen(req,timeout=60) as r:
        data=r.read(limit+1)
        assert len(data)<=limit,('download exceeds frozen bound',url)
        return data,dict(url=url,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),
                         retrieved_utc=now(),etag=r.headers.get('ETag'),
                         final_url_without_query=urllib.parse.urlunsplit(urllib.parse.urlsplit(r.url)._replace(query='',fragment='')))


def rank(key,source_id):
    return hashlib.sha256(f'{SEED}|{key}|{source_id}'.encode()).hexdigest()


def freeze():
    assert not (DEST/'acquisition_protocol.frozen.json').exists()
    assert sha(PLACEMENT)==PLACEMENT_SHA and sha(NUMERIC)==NUMERIC_SHA and sha(MODEL)==MODEL_SHA
    source_manifest=json.loads(MODEL_MANIFEST.read_text())
    assert source_manifest['sha256']==MODEL_SHA
    receipts=[]
    for source in SOURCES:
        repo,rev=source['repo'],source['revision']
        for role,url in [
            ('metadata',f'https://huggingface.co/api/datasets/{repo}/revision/{rev}'),
            ('tree',f'https://huggingface.co/api/datasets/{repo}/tree/{rev}/{source["config"]}')]:
            raw,receipt=get(url,2*1024*1024);doc=json.loads(raw)
            if role=='metadata':
                assert doc['sha']==rev
                assert doc.get('cardData',{}).get('license')==source['license_as_publisher_card']
            else:
                target=next(v for v in doc if v['path']==source['file'])
                assert target['size']==source['bytes'] and target['lfs']['oid']==source['expected_sha256']
            path=DEST/'metadata'/(source['key']+'.'+role+'.json');write_new(path,raw)
            receipts.append(dict(role=role,source=source['key'],path=str(path),**receipt))
    model_revisions={k:v['revision'] for k,v in source_manifest['source_records'].items()}
    protocol=dict(schema='expert_flash.fresh_format_heldout_protocol.v1',frozen_utc=now(),
        source_script_path=str(Path(__file__).resolve()),source_script_sha256=sha(__file__),seed=SEED,
        input_sha256={str(PLACEMENT):PLACEMENT_SHA,str(NUMERIC):NUMERIC_SHA,str(MODEL):MODEL_SHA,
                      str(MODEL_MANIFEST):sha(MODEL_MANIFEST)},
        checkpoint_revisions=model_revisions,
        checkpoint_revision_note='These are model/checkpoint repositories, distinct from dataset revisions below.',
        source_specs=SOURCES,prepayload_metadata_receipts=receipts,
        selection=dict(count_per_dataset=8,
            ranking='Ascending SHA256 of UTF8(decimal_seed + verticalbar + source_key + verticalbar + source_id). No answer-based ranking.',
            wikitext_source_id='Zero-based target Parquet row index as canonical decimal string.',
            wikitext_eligibility='Trimmed nonempty row; not a heading matching ^\\s*=+.*=+\\s*$; at least160 Python str.split whitespace words. Row is the paragraph unit; no joining across rows.',
            wikitext_prompt='Exactly first160 whitespace words joined by one ASCII space, followed by LF. No detokenization, instruction or chat template. Original full row retained.',
            arc_source_id='Publisher id string; duplicate ids are an integrity failure, not a reason to resample.',
            arc_eligibility='Question stem at least5 whitespace words; exactly4 nonempty choices with4 distinct nonempty labels; rendered full stem+choices at most96 whitespace words. No answerKey read for eligibility/ranking.',
            arc_prompt='question.strip() + LF + each choice in source order as label.strip()+dot+space+text.strip(), LF-separated, final LF. No answer/correct label/cue appended.',
            item_order='WikiText rank00..07, then ARC rank00..07. Source identity selected only by frozen ranking. No output- or tokenizer-driven replacement.',
            insufficient_eligible='Fail and retain failure; do not relax rules or resample silently.'),
        workload_contract=dict(purpose='Independent same-format placement/numerical validation with fresh project workloads; not a new quantization method.',
            language_model_quality='LM tokenizer is the existing GGUF tokenizer, add_special=true. First128 tokens, score positions1..127, no chat template; token counts are recorded after selection, not selection criteria.',
            generation='16 fixed prompt files; roughly128 prefill and16 decode tokens per item. Preserve complete ARC question/options; runner must document any differing token budget.',
            arc_answers='Separate answers/arc_answers.json, never included in prompt files. Free generation16/32tokens is not benchmark accuracy.',
            placement='Existing frozen Granite4 plan and numeric contract unchanged; no placement/weight calibration fit after new payload.',
            statistical_scope='Eight length-eligible examples per source, not a representative full benchmark; no model-pretraining contamination claim.'),
        preprotocol_exposure=dict(target_split_payload_opened=False,
            metadata_viewer_erratum='During publisher metadata verification, HF landing pages auto-displayed a few WikiText103raw train and ARC-Challenge train examples. Those are not the target WikiText2test/ARCvalidation payloads and do not enter eligibility or ranking. No target viewer preview is used.'),
        acquisition_limits=dict(payload_files=2,total_expected_payload_bytes=sum(v['bytes'] for v in SOURCES),
            maximum_each_payload_bytes=2*1024*1024,readme_limit_bytes=256*1024,no_models_downloaded=True,
            no_other_splits_or_tasks=True))
    path=DEST/'acquisition_protocol.frozen.json';json_new(path,protocol)
    print(json.dumps(dict(status='frozen before target payload',path=str(path),sha256=sha(path)),indent=2))


def acquire():
    import pyarrow.parquet as pq
    path=DEST/'acquisition_protocol.frozen.json';protocol=json.loads(path.read_text());protocol_sha=sha(path)
    assert protocol['source_script_sha256']==sha(__file__) and protocol['source_specs']==SOURCES
    for p,h in protocol['input_sha256'].items():assert sha(p)==h
    assert not (DEST/'manifest.json').exists()
    acquisition_started=now();receipts=[];items=[];selected_records=[];answers=[];source_reports=[]
    for spec in SOURCES:
        prefix=f'https://huggingface.co/datasets/{spec["repo"]}/resolve/{spec["revision"]}/'
        payload_url=prefix+spec['file']+'?download=true'
        raw,receipt=get(payload_url,protocol['acquisition_limits']['maximum_each_payload_bytes'])
        assert receipt['bytes']==spec['bytes'] and receipt['sha256']==spec['expected_sha256']
        local=DEST/'raw'/(spec['key']+'.parquet');write_new(local,raw)
        receipts.append(dict(source=spec['key'],role='target_split_payload',path=str(local),**receipt))
        rawcard,card_receipt=get(prefix+'README.md',protocol['acquisition_limits']['readme_limit_bytes'])
        card=DEST/'metadata'/(spec['key']+'.README.md');write_new(card,rawcard)
        receipts.append(dict(source=spec['key'],role='publisher_dataset_card',path=str(card),**card_receipt))
        eligible=[];article=None
        if spec['kind']=='lm':
            rows=pq.read_table(local,columns=['text']).to_pylist()
            for ix,row in enumerate(rows):
                text=row['text'];assert isinstance(text,str)
                top=re.fullmatch(r'=\s*([^=]+?)\s*=',text.strip())
                if top:article=top.group(1).strip()
                if not text.strip() or re.fullmatch(r'\s*=+.*=+\s*',text.strip(),flags=re.S) or len(text.split())<160:continue
                eligible.append(dict(row_index=ix,source_id=str(ix),text=text,article_title=article,
                                     ranking_sha256=rank(spec['key'],str(ix))))
        else:
            # Deliberately exclude answerKey until after deterministic selection.
            rows=pq.read_table(local,columns=['id','question','choices']).to_pylist()
            ids=[r['id'] for r in rows];assert len(ids)==len(set(ids))
            for ix,row in enumerate(rows):
                question=row['question'];labels=row['choices']['label'];texts=row['choices']['text']
                if not isinstance(question,str) or len(question.split())<5:continue
                if len(labels)!=4 or len(texts)!=4 or len(set(labels))!=4:continue
                if not all(isinstance(v,str) and v.strip() for v in labels+texts):continue
                prompt=question.strip()+'\n'+'\n'.join(l.strip()+'. '+t.strip() for l,t in zip(labels,texts))+'\n'
                if len(prompt.split())>96:continue
                eligible.append(dict(row_index=ix,source_id=row['id'],question=question,
                    choices=row['choices'],prompt=prompt,ranking_sha256=rank(spec['key'],row['id'])))
        assert len(eligible)>=8
        selected=sorted(eligible,key=lambda r:(r['ranking_sha256'],r['source_id']))[:8]
        assert len({r['ranking_sha256'] for r in selected})==8
        answer_column=pq.read_table(local,columns=['answerKey']).column('answerKey').to_pylist() if spec['kind']=='arc' else None
        source_reports.append(dict(key=spec['key'],source_rows=len(rows),eligible_rows=len(eligible),selected_rows=8,
            selected_source_ids=[r['source_id'] for r in selected],selected_source_row_indices=[r['row_index'] for r in selected],
            dataset_revision=spec['revision'],dataset_license_as_publisher=spec['license_as_publisher_card'],
            raw_source_sha256=sha(local),publisher_card_sha256=sha(card)))
        for i,source in enumerate(selected):
            item_id=f'fresh20261103_{"wikitext2" if spec["kind"]=="lm" else "arc_challenge"}_{i:02d}'
            if spec['kind']=='lm':prompt=' '.join(source['text'].split()[:160])+'\n'
            else:prompt=source['prompt']
            target=DEST/'prompts'/(item_id+'.txt');write_new(target,prompt.encode())
            selected_record=dict(source_key=spec['key'],source_id=source['source_id'],row_index=source['row_index'],
                ranking_sha256=source['ranking_sha256'],**{k:v for k,v in source.items() if k in ['text','article_title','question','choices']})
            if spec['kind']=='lm' and source['article_title']:
                selected_record['wikipedia_article_url']='https://en.wikipedia.org/wiki/'+urllib.parse.quote(source['article_title'].replace(' ','_'))
                selected_record['wikipedia_history_url']=selected_record['wikipedia_article_url']+'?action=history'
            selected_records.append(selected_record)
            item=dict(id=item_id,dataset=spec['repo'],dataset_revision=spec['revision'],dataset_config=spec['config'],
                split=spec['split'],kind=spec['kind'],source_id=source['source_id'],source_row_index=source['row_index'],
                selection_rank=i,ranking_sha256=source['ranking_sha256'],prompt_path=str(target),prompt_sha256=sha(target),
                prompt_utf8_bytes=len(prompt.encode()),prompt_unicode_characters=len(prompt),prompt_whitespace_words=len(prompt.split()),
                source_payload_path=str(local),source_payload_sha256=sha(local),license_as_publisher=spec['license_as_publisher_card'])
            if spec['kind']=='lm':
                item.update(source_text_sha256=textsha(source['text']),source_text_whitespace_words=len(source['text'].split()),
                            article_title=source['article_title'])
                assert len(prompt.split())==160
            else:
                answer=answer_column[source['row_index']];assert answer in source['choices']['label']
                answer_index=source['choices']['label'].index(answer)
                answers.append(dict(id=item_id,source_id=source['source_id'],answer_label=answer,answer_index=answer_index,
                                    answer_text=source['choices']['text'][answer_index]))
                item['answers_sidecar_key']=item_id
                # Exact rendering equality protects against accidental answer append.
                assert prompt==source['question'].strip()+'\n'+'\n'.join(l.strip()+'. '+t.strip() for l,t in zip(source['choices']['label'],source['choices']['text']))+'\n'
            items.append(item)
    records_path=DEST/'selected_source_records.json';json_new(records_path,selected_records)
    answers_path=DEST/'answers/arc_answers.json';json_new(answers_path,answers)
    assert len(items)==16 and [i['kind'] for i in items]==['lm']*8+['arc']*8
    assert len({i['prompt_sha256'] for i in items})==16
    for p,h in protocol['input_sha256'].items():assert sha(p)==h
    assert sha(path)==protocol_sha
    manifest=dict(schema='expert_flash.fresh_format_heldout_manifest.v1',created_utc=now(),
        acquisition_protocol_path=str(path),acquisition_protocol_sha256=protocol_sha,
        acquisition_started_utc=acquisition_started,items=items,source_reports=source_reports,
        selected_source_records_path=str(records_path),selected_source_records_sha256=sha(records_path),
        answers_path=str(answers_path),answers_sha256=sha(answers_path),source_receipts=receipts,
        all_original_inputs_unchanged=True,model_checkpoint_sha256=MODEL_SHA,
        placement_plan_sha256=PLACEMENT_SHA,numeric_protocol_sha256=NUMERIC_SHA,
        placement_fitting_performed=False,model_downloaded=False,answers_in_prompts=False,
        no_benchmark_accuracy_claim=True,
        provenance_note='Publisher-pinned heldout source splits are new project workloads. This does not assert absence from model pretraining. The protocol records metadata-page train previews separately.')
    manifest_path=DEST/'manifest.json';json_new(manifest_path,manifest)
    print(json.dumps(dict(status='acquired frozen16 text items',path=str(manifest_path),sha256=sha(manifest_path),
        items=16,lm=8,arc=8,target_payload_bytes=sum(r['bytes'] for r in receipts if r['role']=='target_split_payload'),
        source_reports=source_reports),indent=2))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','acquire']);args=ap.parse_args()
    freeze() if args.stage=='freeze' else acquire()


if __name__=='__main__':main()
