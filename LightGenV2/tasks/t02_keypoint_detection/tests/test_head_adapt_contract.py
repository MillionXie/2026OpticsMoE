import unittest
from types import SimpleNamespace
from LightGenV2.tasks.t02_keypoint_detection.lab_acquire import validate_fields
from LightGenV2.tasks.t02_keypoint_detection.lab_head_adapt import select_train

class ContractTests(unittest.TestCase):
    def test_test_still_requires_full1000(self):
        with self.assertRaises(ValueError):validate_fields({'fields':[{'key':'test_00000'}]},1)
    def test_train_explicit_identity(self):
        release={'split':'train','samples':2,'fields':[{'key':'train_00000'},{'key':'train_00001'}],
                 'train_test_disjoint':True,'targets_sha256':'abc'}
        self.assertEqual(len(validate_fields(release,2)),2)
        release['train_test_disjoint']=False
        with self.assertRaises(ValueError):validate_fields(release,2)
    def test_fixed_train_subset_and_overlap(self):
        rows=[SimpleNamespace(sample_id=str(i)) for i in range(12)]
        self.assertEqual(select_train(rows[:10],rows[10:],4,1009),select_train(rows[:10],rows[10:],4,1009))
        with self.assertRaises(ValueError):select_train(rows[:10],rows[9:],4,1009)

if __name__=='__main__':unittest.main()
