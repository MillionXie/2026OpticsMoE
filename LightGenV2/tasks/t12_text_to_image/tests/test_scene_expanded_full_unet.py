import copy,unittest
from ..scene_expanded_full_unet import restored_state
from ..scene_depth_recovery import DEEP
from ..scene_expanded_middle import checked_profile
from .test_scene_expanded_rollout_recovery import fixture
from ..scene_expanded_rollout_recovery import checked_contract


class FullUNetRestoration(unittest.TestCase):
    def source(self):
        args=fixture();state=args[1];state['unet_lora']={name+'.'+str(i)+'.attn1.to_q.lora_A.scene_pair.weight':[i]for name in DEEP for i in range(10)}
        state['head']={'readout':[1,2,3]}
        return args

    def test_restore_changes_only_explicit_architecture_metadata(self):
        args=self.source();state=args[1];before=copy.deepcopy(state);restored=restored_state(state,args[2])
        self.assertEqual(state,before)
        for key in state:
            if key!='depth_compression':self.assertEqual(restored[key],state[key])
        profile=restored['depth_compression']['profile'];self.assertEqual(profile['UNet_deep_keep'],10)
        self.assertEqual(checked_profile(profile),profile)
        with self.assertRaises(ValueError):checked_profile(dict(profile,schema='scene_expanded_middle_v1'))

    def test_missing_original_block_rejected_instead_of_fabricated(self):
        args=self.source();args[1]['unet_lora'].pop(next(iter(args[1]['unet_lora'])))
        with self.assertRaises(ValueError):restored_state(args[1],args[2])

    def test_full_capacity_cannot_enter_old_recovery_recipe_silently(self):
        args=list(self.source());state=restored_state(args[1],args[2]);args[1]=state
        args[0]['middle_profile']=state['depth_compression']['profile']
        with self.assertRaises(ValueError):checked_contract(*args)
        args[0].update(schema='scene_expanded_cached_rollout_full_UNet_v3',training_batch_size=2,coverage_epochs=1)
        self.assertEqual(checked_contract(*args)['TRAIN_contexts'],2)


if __name__=='__main__':unittest.main()
