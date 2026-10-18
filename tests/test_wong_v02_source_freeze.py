"""OpenFARM next-phase reliability tests; synthetic data only."""
import unittest,hashlib
from scripts.wong_benchmark import CATS,clean_case,splits
class SourceFrozenTests(unittest.TestCase):
    def test_original_categories_order(self):
        self.assertEqual(CATS,['Immediate','15 mins','30-60 mins','120 mins','240 mins'])
    def test_date_metadata_removed_without_dropping_case_context(self):
        value='Presenting complaint: '+('coughing for two days and difficulty breathing; '*4)+'\nDate: 04/06/2021\nHR: 160'
        s=clean_case(value)
        self.assertNotIn('04/06/2021',s);self.assertIn('HR: 160',s)
    def test_immutable_hash_split_is_case_exclusive(self):
        rows=[{'id':hashlib.sha256(('synthetic'+str(i)).encode()).hexdigest()[:24],
          'median':CATS[i%5]} for i in range(340)]
        a,b=splits(rows)
        self.assertEqual(a,splits(list(reversed(rows)))[0])
        self.assertEqual(set(a.values()),{'train','cal','test'})
        self.assertEqual(len(a),len(rows))
        self.assertEqual(sum(x['test'] for x in b.values()),100)
    def test_cannot_train_on_unapproved_original(self):
        from scripts.wong_benchmark import parse
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.xlsx') as temp:
            temp.write(b'Fake source, not a veterinary workbook');temp.flush()
            with self.assertRaises(ValueError):parse(temp.name)
if __name__=='__main__':unittest.main()
