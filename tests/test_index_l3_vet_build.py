"""Offline synthetic-contract tests; NOT empirical veterinary benchmark examples."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).parents[1]/'scripts'/'index_l3_vet_build.py'
spec = importlib.util.spec_from_file_location('index_l3_vet_build', SCRIPT)
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)
SHA = 'a'*64
MAPPING = dict(case_id='id', state_fields=['presenting_signs','history'],
    vet_votes=['vet_1','vet_2','vet_3'], nurse_votes=['nurse_1'],
    source_doi='10.1234/example', source_sha256=SHA, expected_cases=2)
ROWS = [
    dict(id='case-a', presenting_signs='simulated breathing issue', history='simulated recent illness', vet_1='red',vet_2='orange',vet_3='red',nurse_1='red',diagnosis='SIMULATED FUTURE LABEL'),
    dict(id='case-b', presenting_signs='simulated minor concern',history='simulated no distress', vet_1='green',vet_2='blue',vet_3='yellow',nurse_1='',diagnosis='SIMULATED FUTURE LABEL')]

class CaseBuilder(unittest.TestCase):
    def test_group_by_patient_and_preserve_vote_distribution(self):
        task,audit=B.build(ROWS,MAPPING)
        self.assertEqual(audit['cases'],2)
        self.assertEqual(task[0]['vote_distribution']['red'],2/3)
        self.assertEqual(task[0]['consensus_mode'],'red')
        self.assertIsNone(task[1]['consensus_mode'])
        self.assertEqual(task[1]['consensus_tied_classes'],['yellow','green','blue'])
        self.assertNotIn('SIMULATED FUTURE LABEL',task[0]['state'])
    def test_diagnosis_field_is_blocked(self):
        m={**MAPPING,'state_fields':['presenting_signs','diagnosis']}
        with self.assertRaisesRegex(ValueError,'post-triage'):
            B.build(ROWS,m)
    def test_clinician_vote_cannot_enter_state(self):
        m={**MAPPING,'state_fields':['presenting_signs','vet_1']}
        with self.assertRaisesRegex(ValueError,'duplicate'):
            B.build(ROWS,m)
    def test_duplicate_case_rejected(self):
        with self.assertRaisesRegex(ValueError,'duplicate'):
            B.build([ROWS[0],ROWS[0]],MAPPING)
    def test_duplicate_text_is_rejected_even_with_different_id(self):
        clone={**ROWS[0],'id':'case-b'}
        with self.assertRaisesRegex(ValueError,'duplicate case text'):
            B.build([ROWS[0],clone],MAPPING)
    def test_unknown_vote_fails_closed(self):
        bad={**ROWS[0],'vet_3':'purple'}
        with self.assertRaisesRegex(ValueError,'unmapped'):
            B.build([bad,ROWS[1]],MAPPING)
    def test_missing_reference_never_negative(self):
        blank={**ROWS[1],'vet_1':'','vet_2':'','vet_3':''}
        task,audit=B.build([ROWS[0],blank],MAPPING)
        self.assertEqual(task[1]['reference_status'],'unmeasured')
        self.assertIsNone(task[1]['consensus_mode'])
        self.assertEqual(audit['missing_vet_votes'],3)
    def test_expected_case_count_guard(self):
        with self.assertRaisesRegex(ValueError,'case count mismatch'):
            B.build([ROWS[0]],MAPPING)
    def test_source_hash_integrity(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'source.json'
            p.write_text(json.dumps(ROWS))
            sha=B.digest(p)
            self.assertEqual(len(sha),64)
            self.assertNotEqual(sha,'0'*64)
    def test_normalized_minutes(self):
        self.assertEqual(B.normalize_vote('30–60'),'yellow')
        self.assertEqual(B.normalize_vote('240 min'),'blue')
        self.assertIsNone(B.normalize_vote(''))

if __name__=='__main__':unittest.main()
