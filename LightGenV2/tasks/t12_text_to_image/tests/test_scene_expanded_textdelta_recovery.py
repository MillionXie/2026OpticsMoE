import unittest
from ..scene_expanded_textdelta_recovery import groups_for,reference_representatives,validate_recipe


class ActualReferenceGroups(unittest.TestCase):
    def test_joint_recipe_is_explicit_and_old_defaults_preserved(self):
        for mode in ['text_delta','image_prefix']:validate_recipe(mode,800,5e-5)
        validate_recipe('joint_conditions',3200,5e-5)
        for args in [('text_delta',3200,5e-5),('joint_conditions',800,5e-5),('joint_conditions',3200,1e-4),('unknown',800,5e-5)]:
            with self.assertRaises(ValueError):validate_recipe(*args)
    def test_singleton_not_fabricated_and_full_real_contexts_covered(self):
        rows=[dict(split='train',reference_id=ref,input_key=ref+str(i),condition_id=str(i),prompt='request'+str(i))for ref,n in [('one',1),('two',2),('many',8)]for i in range(n)]
        groups=groups_for(rows)
        self.assertEqual({r['input_key']for g in groups for r in g},{r['input_key']for r in rows})
        for group in groups:
            self.assertEqual(len({r['reference_id']for r in group}),1)
            self.assertEqual(len(group),len({r['input_key']for r in group}))
        self.assertEqual(sum(len(g)==1 for g in groups),1)

    def test_held_out_and_conflicting_inputs_rejected(self):
        row=dict(split='test',reference_id='x',input_key='k',condition_id='0',prompt='a')
        with self.assertRaises(ValueError):groups_for([row])
        row['split']='train'
        with self.assertRaises(ValueError):groups_for([row,dict(row,prompt='different')])

    def test_reference_only_training_does_not_count_prompt_duplicates(self):
        rows=[dict(split='train',reference_id=ref,input_key=ref+str(i),condition_id=str(i),prompt='request'+str(i))for ref,n in [('one',1),('many',8)]for i in range(n)]
        representatives=reference_representatives(rows)
        self.assertEqual(len(representatives),2)
        self.assertEqual({r['reference_id']for r in representatives},{'one','many'})
        self.assertTrue(all(r in rows for r in representatives))
        with self.assertRaises(ValueError):reference_representatives([dict(rows[0],split='test')])


if __name__=='__main__':unittest.main()
