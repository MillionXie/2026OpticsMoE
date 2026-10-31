import unittest
from ..scene_expanded_textdelta_recovery import groups_for


class ActualReferenceGroups(unittest.TestCase):
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


if __name__=='__main__':unittest.main()
