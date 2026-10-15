"""Triage semantics/provenance of OpenFARM published datasets (no media download).

Usage: python scripts/triage_decision_sources.py --dataset soundwel --output audit/soundwel-triage.json
Read-only, pinned Hugging Face dataset revisions; no model inference.
This is a measured diagnostic, NOT proof of label causality or scientific validity.
"""
from __future__ import annotations
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path

CONFIGS={
 "soundwel":{"repo":"oliveirabruno01/soundwel-pig-vocalizations",
   "targets":["Type","Valence","Context Category","Age Category"],
   "groups":["Leakage_Group","Animal ID","Animal_ID","Pig_ID","Source"],
   "controls":["Context Category","Age Category","Sex","Type","Valence","Leakage_Group"]},
 "horse":{"repo":"oliveirabruno01/openfarm-horse-grimace-region",
   "targets":["pain_score","binary_pain_label","face_region"],
   "groups":["horse_id","source_file_id","source_sha256"],
   "controls":["face_region","horse_id","source_sha256"]},
 "cat":{"repo":"oliveirabruno01/openfarm-catmeows",
   "targets":["context"],"groups":["cat_id","owner_id"],
   "controls":["breed","sex_status","cat_id","owner_id"]},
 "hen":{"repo":"oliveirabruno01/openfarm-laying-hen-stress",
   "targets":["stress_binary","stress_context"],
   "groups":["experimental_day_group","source_group_id","source_audio_id"],
   "controls":["phase","arm_code","experimental_day_group","stress_context"]},
 "dog":{"repo":"oliveirabruno01/openfarm-dog-pain",
   "targets":["aww_category"],"groups":["source_subject_id","source_row_index"],
   "controls":["pain_type","pain_category","diagnosis","behavior","case_type","source_subject_id"]},
 "ungulate":{"repo":"oliveirabruno01/openfarm-ungulate-valence",
   "targets":["valence"],"groups":["animal_id","source_group_id","audio_sha256"],
   "controls":["species","species_code","context","context_code","animal_id","source_group_id","valence"]},
}
VIEWS=[("train","test"),("train_raw","test_raw")]

def rows_from_hub(name):
    from datasets import load_dataset,get_dataset_split_names
    from huggingface_hub import HfApi
    config=CONFIGS[name]
    rev=HfApi().dataset_info(config["repo"]).sha
    names=get_dataset_split_names(config["repo"],revision=rev)
    rows={}
    required=set(config["targets"]+config["groups"]+config["controls"])
    schema={}
    for split in names:
        ds=load_dataset(config["repo"],split=split,revision=rev,streaming=True)
        schema[split]=list(ds.column_names)
        columns=[col for col in schema[split] if col in required]
        ds=ds.select_columns(columns)
        rows[split]=list(ds)
    return rev,schema,rows

def overlap(a,b,key):
    left={str(x.get(key)) for x in a if x.get(key) not in (None,"")}
    right={str(x.get(key)) for x in b if x.get(key) not in (None,"")}
    return {"train_distinct":len(left),"test_distinct":len(right),
            "shared":len(left&right)}

def conditional_purity(rows,label,col):
    table=defaultdict(Counter)
    for row in rows:
        k=row.get(col); y=row.get(label)
        if k is None or y is None:continue
        if isinstance(k,(dict,list)):continue
        table[str(k)][str(y)]+=1
    if not table:return None
    ct=sum(sum(v.values()) for v in table.values())
    purity=sum(max(v.values()) for v in table.values())/ct
    distinct=[len(v) for v in table.values()]
    return {"groups":len(table),"observations":ct,"weighted_majority_fraction":round(purity,4),
            "ambiguous_group_values":sum(n>1 for n in distinct)}

def look_up_predict(train,test,label,col):
    table=defaultdict(Counter)
    for row in train:
        k=row.get(col); y=row.get(label)
        if k is not None and y is not None:
            table[str(k)][str(y)]+=1
    table={key:val.most_common(1)[0][0] for key,val in table.items() if sum(val.values())>=2}
    ys=[]
    correct=0
    for row in test:
        v=row.get(col);y=row.get(label)
        if v is None or y is None:continue
        if str(v) in table:
            ys.append(str(y));correct+=str(y)==table[str(v)]
    return {"coverage":round(len(ys)/max(1,len(test)),3),
            "accuracy_on_covered":round(correct/max(1,len(ys)),3)}

def scan(name,revision,schema,splits):
    cfg=CONFIGS[name]
    report={"dataset":cfg["repo"],"revision":revision,"rows":{k:len(v) for k,v in splits.items()},
            "schema":schema,"checks":{}}
    for tr,te in VIEWS:
        if tr not in splits or te not in splits:continue
        train,test=splits[tr],splits[te]
        block={"groups":{},"label_distributions":{},"control_predictability":{}}
        for field in cfg["groups"]:
            if field in schema[tr] and field in schema[te]:
                block["groups"][field]=overlap(train,test,field)
        for label in cfg["targets"]:
            if label not in schema[tr] or label not in schema[te]:continue
            block["label_distributions"][label]={"train":dict(Counter(str(x.get(label)) for x in train)),
                                                 "test":dict(Counter(str(x.get(label)) for x in test))}
            ctr={}
            for col in cfg["controls"]:
                if col==label or col not in schema[tr] or col not in schema[te]:continue
                pure=conditional_purity(train,label,col)
                if pure is None:continue
                ctr[col]={"train_association":pure,
                          "test_lookup":look_up_predict(train,test,label,col)}
            block["control_predictability"][label]=ctr
        report["checks"][f"{tr}_vs_{te}"]=block
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",choices=sorted(CONFIGS),required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    revision,schema,rows=rows_from_hub(args.dataset)
    report=scan(args.dataset,revision,schema,rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print("SOURCE_TRIAGE_REPORT:"+json.dumps(report))
if __name__=="__main__":main()
