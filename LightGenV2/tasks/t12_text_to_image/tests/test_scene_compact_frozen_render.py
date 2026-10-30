import copy
import unittest
from ..scene_compact_frozen_render import verify_compact_freeze


def fixture():
    config=dict(UNet_deep_keep=9);description=dict(schema='scene_compact_recovered_v1',profile=config)
    sampling=dict(steps=6,CFG=7.5,timesteps=[958,793,628,463,298,133])
    state=dict(depth_compression=description,sampling_contract=sampling)
    ready=dict(complete=True,compression_training_performed=True,best_sha256='frozen')
    frozen=dict(schema='scene_frozen_evaluation_v1',development_selection_finished=True,TEST_used_for_model_selection=False,frozen_before_first_TEST_prediction=True,
        models={'light':dict(depth_compression=description,checkpoint_sha256='frozen',sampling_contract=sampling,sampling_steps=6)})
    review=dict(complete=True,all128_contexts_reviewed=True,all16_boards_reviewed=True,checkpoint_sha256='frozen',proxy_all12_mean_gate_pass=True,approved_as_final_lightweight_model=True,independent_TEST_opened=False)
    return state,ready,frozen,'light',review,config


class CompactFreezeTest(unittest.TestCase):
    def test_exact_development_approved_graph_is_accepted(self):
        args=fixture();self.assertEqual(verify_compact_freeze(*args),args[0]['depth_compression'])

    def test_visually_rejected_or_previously_test_tuned_weight_is_rejected(self):
        for key in ['approved_as_final_lightweight_model','all128_contexts_reviewed','proxy_all12_mean_gate_pass']:
            args=list(fixture());args[4][key]=False
            with self.assertRaisesRegex(ValueError,'development-approved'):verify_compact_freeze(*args)
        args=list(fixture());args[2]['TEST_used_for_model_selection']=True
        with self.assertRaises(ValueError):verify_compact_freeze(*args)

    def test_other_depth_or_timesteps_cannot_reuse_an_existing_freeze(self):
        for key,value in [('depth_compression',dict(schema='other')),('sampling_contract',dict(steps=6,CFG=1.))]:
            args=list(fixture());args[2]=copy.deepcopy(args[2]);args[2]['models']['light'][key]=value
            with self.assertRaises(ValueError):verify_compact_freeze(*args)


if __name__=='__main__':unittest.main()
