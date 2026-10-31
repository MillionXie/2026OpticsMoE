import unittest
from ..scene_R2_middle_protocol import checked_profile,checked_training,F6,ELECTRONIC,PAIRS


class MiddleProtocol(unittest.TestCase):
    def config(self):return dict(schema='scene_R2_middle_v1',vision_keep=6,language_keep=4,UNet_deep_keep=9,student_steps=8,
                                parent_generator_sha256='b'*64,parent_frontend_sha256=F6,electronic_teacher_sha256=ELECTRONIC,pairs_sha256=PAIRS,TRAIN_contexts=132)
    def valid(self):
        return dict(pair_sha256=PAIRS,optical_frontend=dict(checkpoint_sha256=F6)),dict(best_sha256='b'*64,initial_paired_checkpoint_sha256=ELECTRONIC,condition_teacher_sha256=ELECTRONIC),dict(teacher_sha256=ELECTRONIC,TRAIN_contexts=132,student_steps=8,teacher_steps=30,DEV_or_TEST_targets_created=False,teacher_is_full_electronic_Qwen=True,teacher_task_LoRA_fusion=dict(task_LoRA_fused=True))
    def test_exact_new_task(self):
        c=self.config();self.assertEqual(checked_profile(c),c);self.assertTrue(checked_training(c,*self.valid(),7.5,False))
    def test_old_task_and_changed_graph_rejected(self):
        for key,value in [('electronic_teacher_sha256','6'*64),('UNet_deep_keep',8),('student_steps',6),('TRAIN_contexts',192)]:
            c=self.config();c[key]=value
            with self.assertRaises(ValueError):checked_profile(c)
    def test_target_leak_and_wrong_parent_rejected(self):
        for key,value in [('DEV_or_TEST_targets_created',True),('teacher_sha256','6'*64),('TRAIN_contexts',192),('teacher_task_LoRA_fusion',None)]:
            s,p,t=self.valid();t[key]=value
            with self.assertRaises(ValueError):checked_training(self.config(),s,p,t,7.5,False)
        s,p,t=self.valid();p['initial_paired_checkpoint_sha256']='6'*64
        with self.assertRaises(ValueError):checked_training(self.config(),s,p,t,7.5,False)
    def test_guidance_and_head_must_be_preserved(self):
        for guidance,adapt in[(1.,False),(7.5,True)]:
            with self.assertRaises(ValueError):checked_training(self.config(),*self.valid(),guidance,adapt)


if __name__=='__main__':unittest.main()
