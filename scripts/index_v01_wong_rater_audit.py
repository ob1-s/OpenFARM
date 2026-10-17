#!/usr/bin/env python3
"""Source-grounded Wong multi-expert uncertainty audit; no case text leaves machine.

Without --source, uses 102-case test manifest in GitHub Actions secret. With
--source, reads original 340-case XLSX via Python ZIP/XML (no extra dependencies).
"""
import argparse,base64,collections,gzip,hashlib,itertools,json,math,os,statistics,zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
SHA="4d4498fff6108f883fd209f4c996599980453a181ba5c39b27ce3d07ec044027"
LABELS=["Immediate","15 mins","30-60 mins","120 mins","240 mins"]

def original_xlsx(path):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=SHA:raise ValueError("Original Wong workbook checksum mismatch")
    tag="{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(path) as f:
        shared=ET.fromstring(f.read("xl/sharedStrings.xml"))
        dictionary=["".join(x.text or "" for x in si.iter(tag+"t")) for si in shared.findall(tag+"si")]
        root=ET.fromstring(f.read("xl/worksheets/sheet1.xml"))
    vectors=[]
    for row in root.find(tag+"sheetData").findall(tag+"row")[1:]:
        columns={}
        for c in row.findall(tag+"c"):
            v=c.find(tag+"v")
            if v is None:continue
            col="".join(ch for ch in c.attrib["r"] if ch.isalpha())
            columns[col]=dictionary[int(v.text)] if c.attrib.get("t")=="s" else v.text
        vectors.append([columns.get(c) for c in ("C","D","E")])
    if len(vectors)!=340:raise ValueError("340 source cases required")
    return vectors,"full_original_340"

def heldout_secret():
    val=os.environ.get("WONG_TEST_CASES_B64")
    if not val:raise ValueError("WONG_TEST_CASES_B64 unavailable: clinical source not transferred")
    raw=json.loads(gzip.decompress(base64.b64decode(val,validate=True)))
    if raw.get("source_sha256")!=SHA or raw.get("split")!="test" or len(raw.get("cases",[]))!=102:
        raise ValueError("Wong secret source provenance mismatch")
    return [c["vet_votes"] for c in raw["cases"]],"test_case_102"

def report(vectors,scope):
    n=len(vectors)
    rows=[[LABELS.index(x) for x in vote] for vote in vectors]
    if any(len(v)!=3 for v in rows):raise ValueError("Each clinical case requires 3 separate vet ratings")
    dist=[]
    for i in range(3):
        dist.append({name:sum(v[i]==j for v in rows) for j,name in enumerate(LABELS)})
    pairs={}
    for a,b in itertools.combinations(range(3),2):
        less=sum(v[a]<v[b] for v in rows)
        more=sum(v[a]>v[b] for v in rows)
        ties=n-less-more
        pairs[f"vet_{a+1}_vet_{b+1}"]={
            "same":ties,"vet_a_more_urgent":less,"vet_b_more_urgent":more,
            "exact_agreement":ties/n,
            "mean_signed_ordinal_vet_a_minus_b":statistics.mean(v[a]-v[b] for v in rows)}
    brier_noise=statistics.mean(1-sum((v.count(k)/3)**2 for k in range(5)) for v in rows)
    d={
        "source_sha256":SHA,"scope":scope,"n_cases":n,"n_expert_votes":n*3,
        "three_way_unanimous":sum(len(set(v))==1 for v in rows),
        "three_distinct_ratings":sum(len(set(v))==3 for v in rows),
        "one_or_more_immediate_votes":sum(0 in v for v in rows),
        "majority_immediate":sum(sorted(v)[1]==0 for v in rows),
        "unanimous_immediate":sum(v==[0,0,0] for v in rows),
        "vet_by_index_distribution":dist,
        "vet_by_index_mean_ordinal_rank":[statistics.mean(v[i] for v in rows) for i in range(3)],
        "pairwise_reference_differences":pairs,
        "mean_pair_abs_ordinal_difference":statistics.mean(abs(a-b) for v in rows for a,b in itertools.combinations(v,2)),
        "median_agrees_with_individual_vote_fraction":sum(sum(x==sorted(v)[1] for x in v) for v in rows)/(3*n),
        "empirical_vote_noise_to_one_hot_brier_floor":brier_noise,
        "reference_explanation":"Brier_q uses 3-vet relative frequencies q per case. Expected one-hot Brier for a randomly sampled vet = Brier_q + empirical_vote_noise_to_one_hot_brier_floor.",
        "independence_warning":"Patients IDs unavailable; case-level frequencies not proof of independent animals or external hospitals.",
        "expert_agreement_warning":"Severity-index asymmetry is descriptive; does not establish that any rater is right or wrong."
    }
    return d

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",type=Path,default=None)
    a=ap.parse_args()
    values,scope=original_xlsx(a.source) if a.source else heldout_secret()
    result=report(values,scope)
    output=Path("reports/decision-index")/(scope+"_wong_expert_agreement.json")
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+"\n")
    print("WONG_RATER_FORENSICS",json.dumps(result,sort_keys=True),flush=True)

if __name__=="__main__":main()
