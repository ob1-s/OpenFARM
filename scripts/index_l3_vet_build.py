#!/usr/bin/env python3
"""Fail-closed builder for independently vet-rated case triage datasets.

This is an ingestion tool, NOT a benchmark or a clinical triage predictor. Original
sources must first be obtained lawfully and mapped using an explicit JSON file.
Only pre-triage input columns are permitted into model-visible state.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

CATEGORIES = ("red", "orange", "yellow", "green", "blue")
ALIASES = {
    "red": "red", "immediate": "red", "0": "red", "0min": "red",
    "orange": "orange", "15": "orange", "15min": "orange",
    "yellow": "yellow", "30": "yellow", "30-60": "yellow", "30–60": "yellow",
    "green": "green", "120": "green", "120min": "green",
    "blue": "blue", "240": "blue", "240min": "blue",
}
BLOCKLIST = re.compile(r"(?i)(diagnos|discharg|outcome|mortality|death|treat|medicat|response|\bgpt\b|model[_ -]?predict|triage|urgency|rating|vote|category|wait[_ -]?time|referral[_ -]?decision)")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def normalize_vote(value: object) -> str | None:
    s = str(value).strip().lower().replace(' ', '') if value is not None else ''
    if s in ('', 'nan', 'none', 'na', 'null', '.', 'notrateable'):
        return None
    if s not in ALIASES:
        raise ValueError(f'unmapped clinical triage category: {value!r}')
    return ALIASES[s]


def read_rows(path: Path) -> list[dict]:
    if path.suffix.lower() == '.csv':
        with path.open(encoding='utf-8-sig', newline='') as f:
            return list(csv.DictReader(f))
    if path.suffix.lower() == '.json':
        raw = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(raw, dict):
            raw = raw.get('cases')
        if not isinstance(raw, list) or not all(isinstance(r, dict) for r in raw):
            raise ValueError('JSON source must be array of cases or {"cases":[...]}')
        return raw
    raise ValueError('source must be CSV or JSON; convert original XLSX/DOCX with a separately audited provenance step')


def build(raw: list[dict], mapping: dict) -> tuple[list[dict], dict]:
    """Source-to-task transformer. Mapping MUST be authored after seeing real columns.

    mapping: case_id, state_fields [..], vet_votes [..], optional nurse_votes [..],
    source_doi, source_sha256, expected_cases. No field auto-discovery.
    """
    required = {'case_id', 'state_fields', 'vet_votes', 'source_doi', 'source_sha256'}
    if not required.issubset(mapping):
        raise ValueError(f'mapping missing required keys: {sorted(required - set(mapping))}')
    state_fields, vets = mapping['state_fields'], mapping['vet_votes']
    nurses = mapping.get('nurse_votes', [])
    if not isinstance(state_fields, list) or not state_fields or not isinstance(vets, list) or len(vets) < 2:
        raise ValueError('must explicitly name state fields and at least 2 distinct veterinarian vote columns')
    all_cols = [mapping['case_id'], *state_fields, *vets, *nurses]
    if len(all_cols) != len(set(all_cols)):
        raise ValueError('source roles overlap or duplicate columns')
    if any(BLOCKLIST.search(field) for field in state_fields):
        raise ValueError('state fields look post-triage or clinician-label-derived; verify against primary-source timeline')
    if not isinstance(mapping['source_doi'], str) or not mapping['source_doi']:
        raise ValueError('missing source DOI')
    if not re.fullmatch(r'[0-9a-f]{64}', str(mapping['source_sha256'])):
        raise ValueError('invalid source SHA256')
    if 'expected_cases' in mapping and len(raw) != mapping['expected_cases']:
        raise ValueError(f'case count mismatch: found {len(raw)}, expected {mapping["expected_cases"]}')
    if not raw:
        raise ValueError('source has no real cases')
    for col in all_cols:
        if any(col not in row for row in raw):
            raise ValueError(f'missing source column {col!r}')
    used_ids, used_text = set(), set()
    results = []
    vote_counts = Counter()
    missing_votes = 0
    for row in raw:
        cid = str(row[mapping['case_id']]).strip()
        if not cid or cid in used_ids:
            raise ValueError(f'duplicate or absent case_id {cid!r}')
        used_ids.add(cid)
        state = '\n'.join(str(row[f]).strip() for f in state_fields)
        if not state.strip():
            raise ValueError(f'empty triage-time state for case {cid}')
        text_fingerprint = hashlib.sha256(' '.join(state.split()).lower().encode()).hexdigest()
        if text_fingerprint in used_text:
            raise ValueError(f'possible duplicate case text under different ID: {cid}')
        used_text.add(text_fingerprint)
        vs = [normalize_vote(row[v]) for v in vets]
        ns = [normalize_vote(row[n]) for n in nurses]
        present = [v for v in vs if v is not None]
        missing_votes += len(vs) - len(present)
        for v in present:
            vote_counts[v] += 1
        c = Counter(present)
        denom = len(present)
        probabilities = {k: c[k] / denom for k in CATEGORIES} if denom else {k: None for k in CATEGORIES}
        strongest = max(c.values(), default=0)
        tied = [k for k in CATEGORIES if c[k] == strongest] if c else []
        consensus = tied[0] if len(tied) == 1 else None
        results.append({
            'case_id': cid, 'state': state,
            'veterinarian_votes': vs, 'nurse_votes': ns,
            'vote_distribution': probabilities,
            'consensus_mode': consensus,
            'consensus_tied_classes': tied if len(tied) > 1 else [],
            'n_vet_votes_present': denom,
            'agreement_fraction': strongest / denom if denom else None,
            'reference_status': 'complete' if denom == len(vets) else 'partial' if denom else 'unmeasured',
        })
    audit = {
        'source_doi': mapping['source_doi'],
        'source_sha256': mapping['source_sha256'],
        'cases': len(results), 'unique_patient_cases': len(used_ids),
        'veterinarian_columns': len(vets), 'nurse_columns': len(nurses),
        'missing_vet_votes': missing_votes,
        'cases_with_unanimous_vet_votes': sum(x['agreement_fraction'] == 1 for x in results),
        'cases_with_tied_reference_modes': sum(bool(x['consensus_tied_classes']) for x in results),
        'vet_class_vote_counts': dict(vote_counts),
        'approval': 'SOURCE_PARSE_ONLY_NOT_BENCHMARK_ADMISSION',
        'note': 'Case-level split and clinician/reference adjudication need separate dataset-specific review.',
    }
    return results, audit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', required=True, type=Path)
    ap.add_argument('--mapping', required=True, type=Path)
    ap.add_argument('--out-dir', required=True, type=Path)
    args = ap.parse_args()
    mapping = json.loads(args.mapping.read_text())
    sha = digest(args.source)
    if sha != mapping.get('source_sha256'):
        raise SystemExit(f'source checksum mismatch: {sha}')
    task, audit = build(read_rows(args.source), mapping)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir/'cases.json').write_text(json.dumps(task, ensure_ascii=False, indent=2)+'\n')
    (args.out_dir/'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
