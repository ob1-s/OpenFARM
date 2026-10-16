"""OpenFARM v0.2: scientific observation-state -> multiple typed questions.

Source/label evidence is not a model input. Compiler refuses subject leakage,
temporal leakage, duplicate state identities and misleading "independent"
supervision counts. No pretense that deriving many binary questions from one
human Choice label provides independent ground truth.

Public reports aggregate counts; optional JSONL exports keep input, labels and
provenance in three *separate* files, for a blinded evaluator/harness.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from dataclasses import dataclass,field
from datetime import datetime
from pathlib import Path
from typing import Any
import hashlib,json,math

KINDS={"Noul","Choice","Score","Value"}
SPLITS={"train","calibration","test"}
RELATIONS={"contemporaneous","future"}
# Do not pass source hints / answers / experimental contexts through fit state.
FORBIDDEN={"label","labels","target","targets","answer","answers","subject_id",
           "animal_id","pig_id","cow_id","horse_id","site","farm","institute",
           "context","condition","conditions","treatment","split","partition",
           "group","group_id","media_id","video_id","track_id","feeder_event",
           "feed","behavior","behaviour","pain","raw_level","annotator","rater",
           "date","time","timestamp","source_minute_index","source_video"}

def safe_json(v:Any):
    if v is None or isinstance(v,(str,bool,int)):return v
    if isinstance(v,float):
        return v if math.isfinite(v) else None
    if isinstance(v,(tuple,list)):return [safe_json(x) for x in v]
    if isinstance(v,dict):return {str(k):safe_json(x) for k,x in v.items()}
    # Keep source objects (NumPy scalars) out of the contract.
    raise TypeError(f"Non-JSON contract value {type(v).__name__}")

@dataclass(frozen=True)
class Question:
    question_id: str
    text: str
    primitive: str
    reference_family: str
    evidence_origin: str
    value: Any=None
    valid: bool=True
    mask_reason: str|None=None
    # A derived proposition can be useful for analysis but must not create
    # a new independent reference family or be counted as new measured truth.
    derived_from: str|None=None
    relation: str="contemporaneous"
    reference_time: str|None=None
    reference_index: int|None=None
    readiness: str="candidate"
    options: tuple[str,...]=()
    def validate(self,cutoff:str|None,cutoff_index:int|None)->None:
        if not self.question_id or not self.reference_family or not self.text:
            raise ValueError("Question id, wording and reference family mandatory")
        if self.primitive not in KINDS:raise ValueError("Unknown primitive")
        if self.relation not in RELATIONS:raise ValueError("Unknown temporal relation")
        if self.valid and self.mask_reason:raise ValueError("valid label carries missingness")
        if not self.valid and (self.value is not None or not self.mask_reason):
            raise ValueError("Masked labels MUST carry a reason but no value")
        if self.valid and self.value is None:raise ValueError("Valid label is null")
        if self.valid and self.primitive=="Noul" and self.value not in (0,1,False,True):
            raise ValueError("Noul reference must be binary")
        if self.valid and self.primitive=="Choice":
            if len(self.options)<2 or self.value not in self.options:
                raise ValueError("Choice reference must match declared options")
        if self.valid and self.primitive in ("Score","Value"):
            if isinstance(self.value,bool) or not isinstance(self.value,(int,float)) or not math.isfinite(self.value):
                raise ValueError("Score/Value must be numeric finite")
        if self.relation=="future":
            has_clock=bool(cutoff and self.reference_time)
            has_index=cutoff_index is not None and self.reference_index is not None
            if has_clock==has_index:raise ValueError("Future target requires exactly one true temporal basis")
            if has_clock and datetime.fromisoformat(self.reference_time)<=datetime.fromisoformat(cutoff):
                raise ValueError("Noncausal future time label")
            if has_index and self.reference_index<=cutoff_index:
                raise ValueError("Noncausal future ordinal index")
    def public(self):
        return safe_json({"id":self.question_id,"question":self.text,"primitive":self.primitive,
          "options":list(self.options),"reference_family":self.reference_family,
          "evidence_origin":self.evidence_origin,"derived_from":self.derived_from,
          "relation":self.relation,"readiness":self.readiness})
    def private(self):
        return {"id":self.question_id,"valid":self.valid,"value":safe_json(self.value),
                "mask_reason":self.mask_reason,"reference_family":self.reference_family,
                "derived_from":self.derived_from,
                "reference_time":self.reference_time,"reference_index":self.reference_index}

@dataclass
class Observation:
    source: str
    state_id: str
    split: str
    group_scope: str
    group_id: str
    modalities: tuple[str,...]
    features: dict[str,Any]
    questions: list[Question]
    cutoff: str|None=None
    cutoff_index: int|None=None
    audit: dict[str,Any]=field(default_factory=dict)
    def validate(self):
        if not self.source or not self.state_id or self.split not in SPLITS:
            raise ValueError("Invalid state provenance/split")
        if not self.group_scope or not self.group_id:
            raise ValueError("Independent source group scope/id required")
        if not self.modalities or len(set(self.modalities))!=len(self.modalities):
            raise ValueError("Modalities must be explicit and unique")
        if not self.features or not self.questions:raise ValueError("State and questions required")
        for key in self.features:
            if key.lower() in FORBIDDEN or key.lower().startswith(("future_","post_","answer_","target_")):
                raise ValueError(f"Model-visible leakage feature: {key}")
        for v in self.features.values():
            safe_json(v)
        if len({q.question_id for q in self.questions})!=len(self.questions):
            raise ValueError("Duplicate question in same state")
        for q in self.questions:q.validate(self.cutoff,self.cutoff_index)

class Compiler:
    def __init__(self,local_dir:Path|None=None):
        self.n_states=0
        self.states_by_source=Counter()
        self.splits_by_source=defaultdict(Counter)
        self.q_possible=Counter()
        self.q_valid=Counter()
        self.q_masked=Counter()
        self.mask_reasons=Counter()
        self.answer_counts=defaultdict(Counter)
        self.reference_families=defaultdict(set)
        self.derived_questions=set()
        self.family_stats=defaultdict(Counter)
        self.by_split=defaultdict(lambda:defaultdict(Counter))
        self.independent_reference_per_state=Counter()
        self.valid_question_per_state=Counter()
        self.independent_questions_per_state=Counter()
        self.composite_states=0
        self.group_seen={}
        self.state_seen=set()
        self.local_dir=Path(local_dir) if local_dir else None
        self.outputs={}
        self.sources=set()
    def add(self,obs:Observation):
        obs.validate()
        ident=(obs.source,obs.state_id)
        if ident in self.state_seen:raise ValueError("Repeated observation state")
        self.state_seen.add(ident)
        grp=(obs.source,obs.group_scope,obs.group_id)
        existing=self.group_seen.get(grp)
        if existing is not None and existing!=obs.split:
            raise ValueError(f"Leakage: {grp} in {existing} and {obs.split}")
        self.group_seen[grp]=obs.split
        if len(obs.modalities)>1:self.composite_states+=1
        self.n_states+=1;self.sources.add(obs.source)
        self.states_by_source[obs.source]+=1
        self.splits_by_source[obs.source][obs.split]+=1
        valid=[q for q in obs.questions if q.valid]
        self.valid_question_per_state[len(valid)]+=1
        # Each human ethogram or instrument/hormone assay is one reference
        # *family* even if it yields 15 one-vs-rest questions or 3 horizons.
        families={q.reference_family for q in valid if not q.derived_from}
        primary_count=sum(not q.derived_from for q in valid)
        self.independent_reference_per_state[len(families)]+=1
        self.independent_questions_per_state[primary_count]+=1
        for q in obs.questions:
            self.q_possible[q.question_id]+=1
            self.reference_families[q.question_id].add(q.reference_family)
            if q.derived_from:self.derived_questions.add(q.question_id)
            if not q.valid:
                self.q_masked[q.question_id]+=1
                self.mask_reasons[q.mask_reason]+=1
                continue
            self.q_valid[q.question_id]+=1
            self.family_stats[(obs.source,q.reference_family)]["question_labels"]+=1
            self.answer_counts[q.question_id][str(q.value)]+=1
            self.by_split[obs.source][obs.split][q.question_id]+=1
        if self.local_dir:self.write(obs)
    def write(self,obs):
        # Four files are isolated: model input, labels, sensitive grouping,
        # and one static catalog of questions (labels never appear in prompt).
        d=self.local_dir/obs.source/obs.split
        d.mkdir(parents=True,exist_ok=True)
        # Opaque stable key: source state IDs often encode pig/site/condition,
        # and temporal cutoffs can reveal fixed experiment schedules. Neither
        # is safe for learner input. Put full identifiers/time in provenance.
        row_id=hashlib.sha256((obs.source+"|"+obs.state_id).encode()).hexdigest()[:24]
        records={
         "states":safe_json({"state_id":row_id,"modalities":list(obs.modalities),
                             "features":obs.features,
                             "question_ids":[q.question_id for q in obs.questions]}),
         "references":{"state_id":row_id,"answers":[q.private() for q in obs.questions]},
         "provenance":{"state_id":row_id,"original_source_key":obs.state_id,
                       "source":obs.source,"group_scope":obs.group_scope,
                       "group_id":obs.group_id,"split":obs.split,
                       "cutoff":obs.cutoff,"cutoff_index":obs.cutoff_index,
                       "audit":obs.audit}
        }
        for name,record in records.items():
            fp=d/(name+".jsonl")
            with fp.open("a",encoding="utf-8") as f:
                f.write(json.dumps(record,sort_keys=True,ensure_ascii=False)+"\n")
        for q in obs.questions:
            self.outputs[q.question_id]=q.public()
    def report(self):
        n=self.n_states
        if not n:raise ValueError("No states compiled")
        fam_count=sum(k*v for k,v in self.independent_reference_per_state.items())
        valid_count=sum(k*v for k,v in self.valid_question_per_state.items())
        independent_q_count=sum(k*v for k,v in self.independent_questions_per_state.items())
        return {"compiler_version":"v0.2","state_count":n,"source_count":len(self.sources),
            "multi_modality_states":self.composite_states,
            "observed_valid_question_labels":sum(self.q_valid.values()),
            "available_question_slots":sum(self.q_possible.values()),
            "masked_question_slots":sum(self.q_masked.values()),
            "masked_by_reason":dict(self.mask_reasons),
            "mean_valid_question_labels_per_state":round(valid_count/n,5),
            "mean_non_derived_question_labels_per_state":round(independent_q_count/n,5),
            "mean_independently_measured_reference_families_per_state":round(fam_count/n,5),
            "independent_reference_families_histogram":dict(sorted(self.independent_reference_per_state.items())),
            "valid_question_labels_histogram":dict(sorted(self.valid_question_per_state.items())),
            "source_states":dict(self.states_by_source),
            "source_splits":{k:dict(v) for k,v in self.splits_by_source.items()},
            "question_slots":dict(self.q_possible),"valid_by_question":dict(self.q_valid),
            "masked_by_question":dict(self.q_masked),
            "derived_question_ids":sorted(self.derived_questions),
            "source_reference_family_label_counts":{
                f"{source}:{family}":d["question_labels"]
                for (source,family),d in self.family_stats.items()},
            "source_valid_label_counts_by_split":{
                src:{sp:dict(cnt) for sp,cnt in data.items()}
                for src,data in self.by_split.items()},
            "group_count_by_source":dict(Counter(k[0] for k in self.group_seen)),
            "readiness":"Source task status is separate from compiler validity: valid mask does not approve clinical/welfare inference",
            "warnings":[
                "Available binary slices of a Choice label are derived and do NOT increase independently measured supervision.",
                "Current and future feeder labels from one physical instrument share one reference family.",
                "Different source assays (e.g. cortisol/oxytocin) can share state but not be mislabeled as welfare/emotion GT.",
                "Labels and group identifiers are physically separated from model-visible state exports."
            ]}
    def finish(self,report_path:Path):
        report_path.parent.mkdir(parents=True,exist_ok=True)
        report=self.report()
        report_path.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
        if self.local_dir:
            (self.local_dir/"question_catalog.json").write_text(
                json.dumps(self.outputs,indent=2,sort_keys=True)+"\n")
        return report
