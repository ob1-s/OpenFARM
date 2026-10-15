"""Fail-closed, source-group-aware integrity audit for prepared OpenFARM datasets.

Run: python scripts/audit_openfarm_integrity.py --dataset horse --output audit.json
Optional: --max-rows 50 (exploratory ONLY; never certifies a split).
Requires: pip install datasets huggingface_hub
No weights, model inference, or modification of Hub datasets.
"""
from __future__ import annotations
import argparse
import hashlib
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

# (Hub repository, identity field, grouping field, primary label field, splits)
SOURCES = {
    'horse': ('oliveirabruno01/openfarm-horse-grimace-region','source_sha256','horse_id','pain_score',['train_raw','test_raw','train','test']),
    'sheep': ('oliveirabruno01/sheep-facial-expression-benchmark','__image_bytes_sha256',None,'label',['train_raw','test_raw','train','test']),
    'dog': ('oliveirabruno01/openfarm-dog-pain','source_row_index','source_subject_id','aww_category',['train_raw','test_raw','train','test']),
    'cat': ('oliveirabruno01/openfarm-catmeows','audio_sha256','cat_id','context',['train_raw','test_raw','train','test']),
    'hen': ('oliveirabruno01/openfarm-laying-hen-stress','audio_sha256','experimental_day_group','stress_context',['train_raw','test_raw','train','test']),
    'ungulate': ('oliveirabruno01/openfarm-ungulate-valence','audio_sha256','source_group_id','valence',['train_raw','test_raw','train','test']),
    'zoo': ('oliveirabruno01/openfarm-zoo-valence-arousal','video_sha256',None,'valence_arousal',['test']),
    'aww': ('oliveirabruno01/AWW-bench',None,None,'target_category',['train','test']),
    'soundwel': ('oliveirabruno01/soundwel-pig-vocalizations','Audio Filename',None,'Valence',None),
}


def _token(value: Any) -> str:
    return str(value) if value is not None else ''


def inspect_rows(splits: dict[str,list[dict]], key: str|None, group: str|None, label: str,
                 expected_splits: list[str]|None, *, complete: bool=True,
                 subset_key: str|None=None) -> dict:
    """Compare group/identity boundaries and raw-vs-balanced views; no green on missing evidence."""
    findings = []
    def note(level, code, detail):
        findings.append({'severity':level,'code':code,'detail':detail})

    if expected_splits:
        for name in expected_splits:
            if name not in splits: note('error','missing_split',name)

    present = {name: rows for name,rows in splits.items() if rows is not None}
    stats = {}
    for name,rows in present.items():
        labels=Counter(_token(r.get(label)) for r in rows)
        stats[name]={'rows':len(rows),'labels':dict(labels)}
        if not rows:
            note('error','empty_split',name)
        for field in (label,key,group):
            if field and any(_token(r.get(field)) == '' for r in rows):
                # Extra, non-headline heldout splits may lack animal identifiers.
                # This is an evidentiary limitation, not an automatic train/test leak.
                level = 'incomplete' if field == group and name not in {
                    'train','test','train_raw','test_raw'
                } else 'error'
                note(level,'missing_value',f'{name}: field {field} missing/empty')
        if key:
            keys=[_token(r.get(key)) for r in rows]
            if len(keys)!=len(set(keys)):
                note('warning','duplicate_identity_within_split',f'{name}: {len(keys)-len(set(keys))} duplicate IDs')
            seen=defaultdict(set)
            for r in rows:
                seen[_token(r.get(key))].add(_token(r.get(label)))
            contradictory=sum(len(values)>1 for values in seen.values())
            if contradictory:
                note('error','contradictory_labels',f'{name}: {contradictory} sample IDs have conflicting labels')

    # Balanced train/test are EXPECTED to be subsets of their raw counterparts.
    for small,big in (('train','train_raw'),('test','test_raw')):
        if small in present and big in present:
            effective_subset_key = subset_key or key
            if effective_subset_key:
                a={_token(r.get(effective_subset_key)) for r in present[small]}
                b={_token(r.get(effective_subset_key)) for r in present[big]}
                extra=a-b
                extra.discard('')
                if extra: note('error','view_not_subset',f'{small} has {len(extra)} IDs absent from {big}')
                if '' in a or '' in b:
                    note('incomplete','subset_key_missing',f'{small}/{big}: incomplete source identity fields')
            else: note('incomplete','subset_unverifiable',f'{small}/{big}: no stable sample ID')

    # Only compare source-disjoint splits, never train vs train_raw or test vs test_raw.
    tr='train_raw' if 'train_raw' in present else 'train'
    te='test_raw' if 'test_raw' in present else 'test'
    if tr in present and te in present:
        if tr==te:
            note('error','identical_split_names','Train/test resolved to identical split name')
        if key:
            train_keys={_token(r.get(key)) for r in present[tr]}
            test_keys={_token(r.get(key)) for r in present[te]}
            shared=train_keys & test_keys
            shared.discard('')
            if shared:
                # IDs are hashed/truncated: no raw media, clinical data or source text.
                examples = sorted(shared)[:5]
                diag=[]
                for sample_id in examples:
                    train_meta=[(str(r.get(group)) if group else None,str(r.get(label)))
                                for r in present[tr] if _token(r.get(key))==sample_id][:3]
                    test_meta=[(str(r.get(group)) if group else None,str(r.get(label)))
                               for r in present[te] if _token(r.get(key))==sample_id][:3]
                    diag.append({'id_prefix':sample_id[:12],
                                 'train_group_label':train_meta,'test_group_label':test_meta})
                note('error','cross_split_identity',
                     f'{tr}/{te}: {len(shared)} shared IDs; examples: {diag}')
        else: note('incomplete','identity_unverifiable','Cannot prove zero sample overlap without stable ID')
        if group:
            train_groups={_token(r.get(group)) for r in present[tr]}
            test_groups={_token(r.get(group)) for r in present[te]}
            shared=train_groups & test_groups
            shared.discard('')
            if shared: note('error','cross_split_group',f'{tr}/{te}: {len(shared)} shared groups; examples: {sorted(shared)[:5]}')
        else: note('incomplete','group_unverifiable','Cannot prove animal/session independence without group ID')
    elif len(present)>1:
        note('incomplete','train_test_unavailable','Source-disjoint train/test splits unavailable')

    if not complete: note('incomplete','partial_sample','A sample-limited run cannot certify whole-split integrity')
    if not key: note('incomplete','missing_stable_identity','No trustworthy sample key configured')
    if not group: note('incomplete','missing_group_identity','No trustworthy subject/session group configured')
    status=('FAIL' if any(f['severity']=='error' for f in findings) else
            'INCOMPLETE' if any(f['severity']=='incomplete' for f in findings) else
            'PASS_WITH_WARNINGS' if any(f['severity']=='warning' for f in findings) else
            'PASS_BOUNDED')
    return {'status':status,'stats':stats,'findings':findings,
            'notes':['A pass covers only listed exact-key/group assertions, NOT source-label semantics, near-duplicates, rights, or commercial generalization.']}


def hub_splits(dataset: str, revision: str, max_rows: int|None) -> dict[str,list[dict]]:
    from datasets import load_dataset, get_dataset_split_names, Audio, Image
    names=get_dataset_split_names(dataset,revision=revision)
    out={}
    for split in names:
        ds=load_dataset(dataset,split=split,revision=revision,streaming=True)
        for col,feature in ds.features.items():
            if isinstance(feature, Audio): ds=ds.cast_column(col,Audio(decode=False))
            elif isinstance(feature, Image): ds=ds.cast_column(col,Image(decode=False))
        rows=[]
        for i,row in enumerate(ds):
            if max_rows and i>=max_rows:break
            # Never write medical text, raw media bytes or source metadata into report.
            safe_row={k:v for k,v in row.items() if not isinstance(v,(dict,bytes,bytearray,list))}
            for media_col in ('image','audio'):
                payload=row.get(media_col)
                if isinstance(payload,dict) and isinstance(payload.get('bytes'),bytes):
                    safe_row[f'__{media_col}_bytes_sha256']=hashlib.sha256(
                        payload['bytes']).hexdigest()
            if dataset == SOURCES['sheep'][0] and row.get('source_filename') is not None:
                # Sheep filenames are reused *across label directories*. Their
                # (filename, label) tuple is a candidate provenance key, not
                # a cryptographic sample fingerprint.
                safe_row['__sheep_source_label_key'] = (
                    str(row['source_filename']) + '::' + str(row.get('label'))
                )
            rows.append(safe_row)
        out[split]=rows
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset',choices=sorted(SOURCES),required=True)
    ap.add_argument('--max-rows',type=int,default=None,help='Audit sample only; report cannot pass')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.max_rows is not None and args.max_rows<=0: ap.error('--max-rows must be positive')
    from huggingface_hub import HfApi
    dsid,key,group,label,expected=SOURCES[args.dataset]
    sha=HfApi().dataset_info(dsid).sha
    if not sha: raise RuntimeError(f'Cannot resolve immutable revision for {dsid}')
    splits=hub_splits(dsid,sha,args.max_rows)
    report=inspect_rows(
        splits,key,group,label,expected,complete=args.max_rows is None,
        subset_key='__sheep_source_label_key' if args.dataset=='sheep' else None
    )
    report['dataset']=dsid; report['revision']=sha; report['sample_only']=args.max_rows is not None
    # Verify whether published audio_sha256 claims describe the actual media bytes.
    media_checks={}
    for split,items in splits.items():
        both=[r for r in items if r.get('audio_sha256') and r.get('__audio_bytes_sha256')]
        if both:
            media_checks[split]={'checked':len(both),
                 'matching':sum(r['audio_sha256']==r['__audio_bytes_sha256'] for r in both)}
    if media_checks: report['audio_sha256_validation']=media_checks
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(f'{dsid}@{sha}: {report["status"]}; {len(report["findings"])} findings → {args.output}')
    if report['status']=='FAIL':raise SystemExit(2)
    if report['status']=='INCOMPLETE':raise SystemExit(3)

if __name__=='__main__': main()
