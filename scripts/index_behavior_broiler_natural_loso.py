#!/usr/bin/env python3
"""All-92-video Big Broiler cross-camera L1 summary without positive-video repair.

Derives aggregate *leave-one-camera-setup-out* (LOSO) heldout statistics from
the already implemented source-audited geometry model pipeline. Each heldout
video is scored once by a model trained on OTHER setups. This is not a native
image/video model, natural farm prevalence or L2 injury outcome.
"""
import argparse,json,random,statistics
from pathlib import Path
import pandas as pd
def rate(tp,fp,fn):
    p=tp/max(1,tp+fp);r=tp/max(1,tp+fn)
    return {"tp":int(tp),"fp":int(fp),"fn":int(fn),
            "precision":p,"recall":r,"f1":2*p*r/max(1e-12,p+r)}
def summarize(report,manifest,boots=400):
    loso=report["loso_family_blocked"]
    if set(loso)!={"1","2","3"}:raise ValueError("Must have all three independent heldout setups")
    all_stems=set()
    models={}
    for model in ("logistic_l2","extra_trees_55"):
        episodes=[0,0,0];window=[0,0,0];per_video=[]
        dur=0.0;positive=0;scorable=0;by_setup={}
        for setup in sorted(loso):
            fam=loso[setup]
            b=fam[model]; rows=b["per_video"]
            if len(rows)!=fam["held_videos"]:raise ValueError("LOSO incomplete video count")
            stems={x["stem"] for x in rows}
            if any(x["stem"] in all_stems for x in rows) and model=="logistic_l2":
                raise ValueError("Same source video heldout by two setup folds")
            if model=="logistic_l2":all_stems.update(stems)
            for x in rows:
                if str(x["setup"])!=setup:raise ValueError("source setup not held out")
                frames=manifest[x["stem"]]
                if frames<=0:raise ValueError("bad source manifest")
                hours=frames/25/3600
                dur+=hours
                ep=x["episodes"]
                episodes=[a+ep[key] for a,key in zip(episodes,("tp","fp","fn"))]
                window=[a+x[key] for a,key in zip(window,("window_tp","window_fp","window_fn"))]
                per_video.append({"setup":setup,"hours":hours,"episode_fp":ep["fp"],"ep_tp":ep["tp"],
                                  "ep_fn":ep["fn"],"w_tp":x["window_tp"],"w_fp":x["window_fp"],
                                  "w_fn":x["window_fn"]})
            scorable+=int(fam["held_scorable"]);positive+=int(fam["held_positives"])
            by_setup[setup]={"videos":len(rows),"positive_window_prevalence":int(fam["held_positives"])/max(1,int(fam["held_scorable"])),
                             "false_episodes_per_clip_hour":b["episode"]["false_episodes_per_stream_hour"],
                             "episode_counts":{k:b["episode"][k] for k in ("tp","fp","fn")}}
        if len(per_video)!=92:raise ValueError("Must cover all 92 non-IAA original source clips")
        ep=rate(*episodes);ws=rate(*window)
        ep["false_episodes_per_annotated_clip_hour"]=episodes[1]/dur
        ep["annotated_clip_hours"]=dur
        ws["held_scorable_positive_fraction"]=positive/max(1,scorable)
        rng=random.Random(20261018)
        bootstrap=[]
        # Stratify bootstrap by setup, preserve each setup's source-video count.
        setup_groups={s:[v for v in per_video if v["setup"]==s] for s in ("1","2","3")}
        for _ in range(boots):
            sampled=[rng.choice(g) for g in setup_groups.values() for _ in g]
            hh=sum(v["hours"] for v in sampled)
            bootstrap.append(sum(v["episode_fp"] for v in sampled)/hh)
        bootstrap.sort()
        models[model]={"episode":ep,"windows":ws,"by_setup":by_setup,
                       "video_cluster_bootstrap_95pct_fp_per_clip_hour":[
                         bootstrap[int(.025*(boots-1))],bootstrap[int(.975*(boots-1))]],
                       "threshold_policy":"LOSO benchmark: selected from TRAINING setups only; no heldout threshold tuning"}
    if len(all_stems)!=92:raise ValueError("Non-unique video population")
    return {"status":"DEVELOPMENTAL_CLIP_SET_NOT_NATURAL_FARM_PREVALENCE",
            "source":"Big Broiler 102 video archive, 92 non-IAA training/eval clips; all videos assigned once to heldout setup",
            "representation":"released annotated tracking geometry, NOT original pixels",
            "task_level":"L1 observed feather-pecking, NOT clinical injury/care",
            "source_sampling":"all 92 non-IAA clips without event-count selected evaluation subset; no label-informed video repair",
            "source_manifest_sha256":report["meta"]["manifest_sha256"],
            "video_independence":"heldout camera setup family for each LOSO specialist; 3 models trained on other two setups",
            "other_constraints":"source footage itself curated; per-stream farm false alarms and natural prevalence unmeasured",
            "n_source_heldout_videos":len(all_stems),"models":models}
def main():
    a=argparse.ArgumentParser()
    a.add_argument("--benchmark",type=Path,default=Path("reports/index_behavior_broiler_event_benchmark.json"))
    a.add_argument("--manifest",type=Path,default=Path("data/broiler/cache/manifest.csv"))
    a.add_argument("--out",type=Path,default=Path("reports/index_behavior_broiler_natural_clip_loso.json"))
    args=a.parse_args()
    r=json.loads(args.benchmark.read_text());d=pd.read_csv(args.manifest)
    manifest={str(row["stem"]):int(row["video_nb_frames"]) for _,row in d.iterrows()}
    ans=summarize(r,manifest)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(ans,indent=2)+"\n")
    print("BROILER_NATURAL_CLIP_LOSO_RESULT",json.dumps({"status":ans["status"],
       "videos":ans["n_source_heldout_videos"],
       "models":{k:{"episode":v["episode"],"windows":v["windows"],
                    "fp_bootstrap_ci":v["video_cluster_bootstrap_95pct_fp_per_clip_hour"]}
                 for k,v in ans["models"].items()}},sort_keys=True))
if __name__=="__main__":main()
