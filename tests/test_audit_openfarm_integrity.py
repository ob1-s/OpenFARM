"""Offline fixtures validating the fail-closed source/split checker.

Run from repository root:
    python -m unittest discover -s tests -p 'test_audit_openfarm_integrity.py' -v
"""
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
sp=importlib.util.spec_from_file_location('audit_openfarm_integrity',ROOT/'scripts'/'audit_openfarm_integrity.py')
m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
A={'id':'a','animal':'A','label':'yes'}
B={'id':'b','animal':'A','label':'no'}
C={'id':'c','animal':'C','label':'yes'}

def check(raw_train=None,raw_test=None,train=None,test=None,**kwargs):
    splits={'train_raw':raw_train if raw_train is not None else [A,B],
            'test_raw':raw_test if raw_test is not None else [C],
            'train':train if train is not None else [A],
            'test':test if test is not None else [C]}
    return m.inspect_rows(splits,'id','animal','label',list(splits),**kwargs)

class IntegrityTests(unittest.TestCase):
    def test_clean_pass(self): self.assertEqual(check()['status'],'PASS_BOUNDED')
    def test_subject_leak(self):
        r=check(raw_test=[{'id':'c','animal':'A','label':'yes'}])
        self.assertEqual(r['status'],'FAIL')
        self.assertIn('cross_split_group',[x['code'] for x in r['findings']])
    def test_exact_id_leak(self):
        self.assertIn('cross_split_identity',[x['code'] for x in check(raw_test=[A])['findings']])
    def test_balanced_view_not_subset(self):
        self.assertIn('view_not_subset',[x['code'] for x in check(train=[C])['findings']])
    def test_conflicting_labels(self):
        self.assertIn('contradictory_labels',[x['code'] for x in check(raw_train=[A,dict(A,label='no')])['findings']])
    def test_missing_subject_id(self):
        r=m.inspect_rows({'train':[A],'test':[C]},'id',None,'label',['train','test'])
        self.assertEqual(r['status'],'INCOMPLETE')
    def test_partial_cannot_certify(self): self.assertEqual(check(complete=False)['status'],'INCOMPLETE')
    def test_missing_split(self):
        r=m.inspect_rows({'train':[A]},'id','animal','label',['train','test'])
        self.assertEqual(r['status'],'FAIL')
        self.assertIn('missing_split',[x['code'] for x in r['findings']])
    def test_empty_required_fields(self):
        r=check(raw_train=[dict(A,animal=''),B])
        self.assertIn('missing_value',[x['code'] for x in r['findings']])

if __name__=='__main__': unittest.main()
