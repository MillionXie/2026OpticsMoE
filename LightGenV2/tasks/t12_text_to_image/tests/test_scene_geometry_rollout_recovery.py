import copy,unittest
from ..scene_geometry_rollout_recovery import training_grid,with_training_grid


class TrainingGrid(unittest.TestCase):
    def inputs(self):
        teacher=dict(teacher_sha256='frozen_E30',full_TRAIN_teacher_paths_sha256='full_path_SHA')
        geometry=dict(complete=True,CPU_only=True,DEV_or_TEST_forwarded=False,TRAIN_contexts=192,
            teacher_sha256='frozen_E30',teacher_path_sha256='full_path_SHA',teacher_timesteps=list(range(958,-2,-32)),
            suggestions={'6':{'edges':[0,3,9,16,22,27,30]}})
        return geometry,teacher
    def test_training_grid_is_exact_subset_of_the_bound_teacher(self):
        g,t=self.inputs();times,edges=training_grid(g,t,6)
        self.assertEqual(times,[g['teacher_timesteps'][i]for i in [0,3,9,16,22,27]])
        self.assertEqual(edges[-1],30)
    def test_reject_different_teacher_or_any_heldout_geometry(self):
        g,t=self.inputs()
        for key,value in [('teacher_path_sha256','wrong'),('teacher_sha256','wrong'),('DEV_or_TEST_forwarded',True),('TRAIN_contexts',128)]:
            broken=dict(g);broken[key]=value
            with self.assertRaises(ValueError):training_grid(broken,t,6)
    def test_untrained_old_checkpoint_contract_is_never_mutated(self):
        state=dict(sampling_contract=dict(steps=6,CFG=7.5,timesteps=[958,793,628,463,298,133]),head={'frozen':1})
        old=copy.deepcopy(state);new=with_training_grid(state,[958,859,661,430,232,67],'geometry_SHA')
        self.assertEqual(state,old);self.assertNotEqual(new['sampling_contract'],old['sampling_contract'])
        self.assertIs(new['head'],state['head']);self.assertTrue(new['TRAIN_grid_recovery']['new_grid_is_for_training_not_zero_shot_use'])


if __name__=='__main__':unittest.main()
