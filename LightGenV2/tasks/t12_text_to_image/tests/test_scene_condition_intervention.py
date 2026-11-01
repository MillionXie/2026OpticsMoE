import unittest
from ..scene_condition_intervention import CASES,selected_train_rows


def cohort():
    return [dict(split='train',input_key=f'{cat}-{ref}-{cond}',category=cat,
        reference_id=f'{cat}-{ref}',condition_id=str(cond))
        for cat in ['lamp','pillow']for ref in range(5)for cond in [10,2,0]]


class InterventionContract(unittest.TestCase):
    def test_fixed_cohort_is_order_independent_and_numeric(self):
        rows=cohort();keys={r['input_key']for r in rows}
        a=selected_train_rows(rows,keys);b=selected_train_rows(list(reversed(rows)),keys)
        self.assertEqual(a,b);self.assertEqual(len(a),16)
        self.assertEqual({r['condition_id']for r in a},{'0','2'})
        self.assertFalse(any(r['reference_id'].endswith('-4')for r in a))

    def test_held_out_paths_and_duplicate_train_identity_refused(self):
        rows=cohort();keys={r['input_key']for r in rows}
        with self.assertRaises(ValueError):selected_train_rows(rows,keys|{'test-only'})
        with self.assertRaises(ValueError):selected_train_rows(rows+[rows[0]],keys)
        changed=[dict(r,split='test')if r['input_key']==rows[0]['input_key']else r for r in rows]
        with self.assertRaises(ValueError):selected_train_rows(changed,keys)

    def test_all_six_pairs_keep_a_frozen_control(self):
        cases={name:(model,condition,steps)for name,model,condition,steps in CASES}
        self.assertEqual(len(cases),6)
        self.assertEqual(cases['E_E_30'],('E','E',30))
        self.assertEqual(cases['E_LG_30'],('E','LG',30))
        self.assertEqual(cases['E_E_8'],('E','E',8))
        self.assertEqual(cases['E_LG_8'],('E','LG',8))
        self.assertEqual(cases['LG_E_8'],('LG','E',8))
        self.assertEqual(cases['LG_LG_8'],('LG','LG',8))


class TensorInterventionContract(unittest.TestCase):
    def test_finite_matched_latents_and_identity(self):
        import torch
        from ..scene_condition_intervention import relative_error
        value=torch.tensor([[[[1.,2.]]]])
        self.assertEqual(relative_error(value,value),0.)
        with self.assertRaises(ValueError):relative_error(value,torch.zeros(2))
        with self.assertRaises(ValueError):relative_error(value*float('nan'),value)


if __name__=='__main__':unittest.main()
