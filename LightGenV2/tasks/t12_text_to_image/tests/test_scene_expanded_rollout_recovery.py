import unittest
from ..scene_expanded_rollout_recovery import checked_contract
from ..scene_R2_middle_protocol import F6
from ..scene_paper_expanded_data import PROFILE


def fixture():
    profile=dict(schema='scene_expanded_middle_v1',prompt_profile=PROFILE,vision_keep=6,language_keep=4,
        UNet_deep_keep=9,student_steps=8,parent_generator_sha256='a'*64,parent_frontend_sha256=F6,
        electronic_teacher_sha256='b'*64,pairs_sha256='c'*64,TRAIN_contexts=2)
    contract=dict(schema='scene_expanded_cached_rollout_v1',middle_profile=profile,warm_checkpoint_sha256='d'*64,reuse_all_TRAIN_contexts=True)
    state=dict(depth_compression=dict(profile=profile),sampling_contract=dict(steps=8,CFG=7.5),
        no_validation_test_PT_selection=True,pair_sha256='c'*64,optical_frontend=dict(checkpoint_sha256=F6))
    parent=dict(best_sha256='d'*64,compression_training_performed=True)
    cache=dict(pair_sha256='c'*64,optical_frontend=dict(checkpoint_sha256=F6),records=[
        dict(input_key='lamp',split='train'),dict(input_key='pillow',split='train'),dict(input_key='held',split='test')])
    teacher=dict(all_TRAIN=True,paths=dict(lamp=None,pillow=None),pair_sha256='c'*64,teacher_steps=30,teacher_sha256='b'*64)
    source=dict(complete=True,teacher_sha256='b'*64,TRAIN_contexts=2,teacher_is_full_electronic_Qwen=True,DEV_or_TEST_targets_created=False)
    return contract,state,parent,cache,teacher,source


class ExpandedRolloutContract(unittest.TestCase):
    def test_exact_lineage_accepted(self):
        self.assertEqual(checked_contract(*fixture())['TRAIN_contexts'],2)

    def test_wrong_teacher_warm_model_or_optics_rejected(self):
        for side,key in [(2,'best_sha256'),(4,'teacher_sha256'),(5,'teacher_sha256')]:
            args=fixture();args[side][key]='f'*64
            with self.assertRaises(ValueError):checked_contract(*args)
        args=fixture();args[3]['optical_frontend']['checkpoint_sha256']='f'*64
        with self.assertRaises(ValueError):checked_contract(*args)

    def test_held_out_or_partial_coverage_cannot_enter_gradients(self):
        for replacement in [dict(lamp=None,held=None),dict(lamp=None)]:
            args=fixture();args[4]['paths']=replacement
            with self.assertRaises(ValueError):checked_contract(*args)
        args=fixture();args[5]['DEV_or_TEST_targets_created']=True
        with self.assertRaises(ValueError):checked_contract(*args)

    def test_architecture_or_pixel_contract_not_silently_changed(self):
        for key,value in [('steps',6),('CFG',1.)]:
            args=fixture();args[1]['sampling_contract'][key]=value
            with self.assertRaises(ValueError):checked_contract(*args)
        args=fixture();args[1]['joint_image_context']=True
        with self.assertRaises(ValueError):checked_contract(*args)
        args=fixture();args[0]['reuse_all_TRAIN_contexts']=False
        with self.assertRaises(ValueError):checked_contract(*args)


if __name__=='__main__':unittest.main()
