import copy
import unittest
from ..scene_paper_middle import checked_profile,checked_training,F6,PROFILE


class NewMiddle(unittest.TestCase):
    def config(self):return dict(schema='scene_paper_middle_v1',prompt_profile=PROFILE,vision_keep=6,language_keep=4,UNet_deep_keep=9,student_steps=8,parent_generator_sha256='a'*64,parent_frontend_sha256=F6,electronic_teacher_sha256='b'*64,pairs_sha256='c'*64,TRAIN_contexts=100)
    def test_keep_selected_graph(self):
        c=self.config();self.assertEqual(checked_profile(c),c)
        for key,v in [('UNet_deep_keep',8),('student_steps',6),('TRAIN_contexts',133),('parent_frontend_sha256','d'*64)]:
            bad=dict(c,**{key:v})
            with self.assertRaises(ValueError):checked_profile(bad)
    def test_no_old_teacher_or_dev_targets(self):
        c=self.config();state=dict(pair_sha256='c'*64,joint_image_context=False,optical_frontend=dict(checkpoint_sha256=F6));parent=dict(best_sha256='a'*64,initial_paired_checkpoint_sha256='b'*64,condition_teacher_sha256='b'*64);targets=dict(teacher_sha256='b'*64,TRAIN_contexts=100,student_steps=8,teacher_steps=30,DEV_or_TEST_targets_created=False,teacher_is_full_electronic_Qwen=True,teacher_task_LoRA_fusion=dict(task_LoRA_fused=True))
        self.assertTrue(checked_training(c,state,parent,targets,7.5,False))
        for key,v in [('teacher_sha256','d'*64),('DEV_or_TEST_targets_created',True),('TRAIN_contexts',99)]:
            with self.assertRaises(ValueError):checked_training(c,state,parent,dict(targets,**{key:v}),7.5,False)


if __name__=='__main__':unittest.main()
