import json
import random
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.high_alpha import augment


class PositionJitterTests(unittest.TestCase):
    def config(self):
        return dict(minimum_crop_side_fraction=1., rotation_degrees=0,
                    contain_jitter_min_scale=.9, contain_jitter_probability=.5,
                    horizontal_flip_probability=0., brightness_min=1., brightness_max=1.,
                    contrast_min=1., contrast_max=1., blur_probability=0., blur_radius=.3)

    def test_zero_probability_leaves_pixels(self):
        image=Image.fromarray(np.random.default_rng(2).integers(0,256,(224,224,3),dtype=np.uint8))
        cfg=self.config();cfg['contain_jitter_probability']=0.
        for seed in range(5):
            np.testing.assert_array_equal(np.array(image),np.array(augment(image,random.Random(seed),cfg)))

    def test_whole_content_remains_and_half_samples_keep_placement(self):
        image=Image.new('RGB',(224,224),'black');cfg=self.config()
        unchanged=0
        for seed in range(100):
            out=np.array(augment(image,random.Random(seed),cfg))
            rows,cols=np.where(out[:,:,0]<128)
            self.assertGreaterEqual(rows.max()-rows.min()+1,202)
            self.assertGreaterEqual(cols.max()-cols.min()+1,202)
            unchanged+=int(np.all(out==0))
        self.assertTrue(30<unchanged<70)

    def test_old_always_jitter_rng_is_preserved(self):
        cfg=self.config();cfg.pop('contain_jitter_probability')
        image=Image.new('RGB',(224,224),'black')
        for seed in range(10):
            rng=random.Random(seed);side=round(224*rng.uniform(.9,1.))
            left,top=rng.randint(0,224-side),rng.randint(0,224-side)
            expected=Image.new('RGB',(224,224),'white');expected.paste(image.resize((side,side)),(left,top))
            np.testing.assert_array_equal(np.array(augment(image,random.Random(seed),cfg)),np.array(expected))

    def test_reject_bad_probability_and_profile_contract(self):
        cfg=self.config();cfg['contain_jitter_probability']=float('nan')
        with self.assertRaises(ValueError):augment(Image.new('RGB',(224,224)),random.Random(1),cfg)
        path=Path(__file__).resolve().parents[1]/'standalone/domain_distillation.json'
        profile=json.loads(path.read_text())['profiles']['domain_distill_position_jitter']
        self.assertEqual(profile['relation_teacher_weight'],.3)
        self.assertEqual(profile['test_every'],1)
        self.assertNotIn('input_preprocessing',profile)
        self.assertNotIn('retrieval_head',profile)
        self.assertNotIn('teacher_feature_weight',profile)
        self.assertEqual(profile['augmentation']['minimum_crop_side_fraction'],1.)


if __name__=='__main__':unittest.main()
