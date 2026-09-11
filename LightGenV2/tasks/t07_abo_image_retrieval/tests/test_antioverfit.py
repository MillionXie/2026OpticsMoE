import random
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.data import Sample
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization import overlay_config, parameter_decay
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.high_alpha import augment
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.optics import phase_dropout, OpticalPath
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.learning_curves import clean_train_metrics, curve_rows, write_learning_curves


class AntiOverfitTests(unittest.TestCase):
    def test_clean_train_removes_entire_product(self):
        samples=[Sample(str(p)+'_'+str(v),str(p),p,str(p),'train',Path('unused')) for p in range(2) for v in range(12)]
        features=torch.eye(2).repeat_interleave(12,0)
        metrics=clean_train_metrics(features,samples)
        self.assertEqual(metrics['candidate_count'],1)
        self.assertEqual(metrics['hit_at_1'],0.)

    def test_clean_train_other_same_class_is_relevant(self):
        samples=[Sample(str(p)+'_'+str(v),str(p),p//2,str(p//2),'train',Path('unused')) for p in range(4) for v in range(12)]
        features=torch.eye(2).repeat_interleave(24,0)
        metrics=clean_train_metrics(features,samples)
        self.assertEqual(metrics['candidate_count'],3)
        self.assertEqual(metrics['hit_at_1'],1.)

    def test_phase_dropout_keeps_unit_amplitude_and_can_backprop(self):
        raw=torch.ones(16,16,requires_grad=True)
        modulation=torch.exp(1j*raw)
        dropped=phase_dropout(modulation,1.,True,3)
        self.assertEqual(dropped.shape,(3,16,16))
        self.assertTrue(torch.equal(dropped,torch.ones_like(dropped)))
        dropped.real.sum().backward()
        self.assertTrue(torch.equal(raw.grad,torch.zeros_like(raw)))
        torch.manual_seed(42)
        partly=phase_dropout(modulation,.5,True,3,4)
        self.assertTrue(torch.allclose(partly.abs(),torch.ones_like(partly.real)))
        self.assertFalse(torch.equal(partly[0],partly[1]))

    def test_phase_dropout_disabled_is_exact_and_consumes_no_rng(self):
        modulation=torch.exp(1j*torch.ones(16,16))
        state=torch.get_rng_state()
        self.assertIs(phase_dropout(modulation,.5,False,2),modulation)
        self.assertIs(phase_dropout(modulation,0.,True,2),modulation)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        optics=OpticalPath();optics.configure_phase_dropout(dict(expert_global_probability=.05,router_probability=.02))
        optics.train();optics.set_training_noise(False)
        self.assertTrue(optics.training and optics.router.training)
        self.assertFalse(optics.noise_enabled or optics.router.noise_enabled)
        self.assertEqual(optics.phase_dropout_probability,.05)

    def test_no_decay_on_phase_alpha_bias_or_norm(self):
        p=torch.ones(3,3)
        self.assertEqual(parameter_decay('vision.optics.global_phase',p,'phase',.01),0)
        self.assertEqual(parameter_decay('x.weight',p,'router',.01),0)
        self.assertEqual(parameter_decay('x.weight',torch.ones(3),'electronic',.01),0)
        self.assertEqual(parameter_decay('x.bias',p,'electronic',.01),0)
        self.assertEqual(parameter_decay('x.weight',p,'electronic',.01),.01)

    def test_controls_only_differ_in_phase_dropout(self):
        a=overlay_config({'adapt':{},'augmentation':{}},'regularized_control')
        b=overlay_config({'adapt':{},'augmentation':{}},'regularized_phase05')
        self.assertNotEqual(a.pop('phase_dropout'),b.pop('phase_dropout'))
        self.assertEqual(a,b)

    def test_augmentation_keeps_object_and_does_not_mutate_original(self):
        cfg=overlay_config({'adapt':{},'augmentation':{}},'regularized_control')['augmentation']
        cfg.update(brightness_min=1.,brightness_max=1.,contrast_min=1.,contrast_max=1.,color_min=1.,color_max=1.,blur_probability=0)
        raw=np.full((224,224,3),255,dtype=np.uint8);raw[:12,90:130]=[255,0,0];raw[-12:,90:130]=[0,0,255]
        im=Image.fromarray(raw)
        for seed in range(8):
            value=np.asarray(augment(im,random.Random(seed),cfg))
            self.assertTrue(np.any(np.all(value==[255,0,0],-1)))
            self.assertTrue(np.any(np.all(value==[0,0,255],-1)))
        self.assertTrue(np.array_equal(raw,np.asarray(im)))

    def test_curves_do_not_invent_clean_values(self):
        history=[dict(epoch=1,losses={'train_gallery_hit1':.98,'correct':.99}),
                 dict(epoch=5,test={'hit_at_1':.7,'train_clean_leave_product_out':{'hit_at_1':.95}},
                      test_live={'hit_at_1':.69,'train_clean_leave_product_out':{'hit_at_1':.97}})]
        rows=curve_rows(history)
        self.assertIsNone(rows[0]['clean_train_hit1_live'])
        self.assertAlmostEqual(rows[1]['gap_ema_pp'],25.)
        self.assertAlmostEqual(rows[1]['gap_live_pp'],28.)
        with tempfile.TemporaryDirectory() as d:
            write_learning_curves(history,d)
            self.assertTrue((Path(d)/'learning_curves.png').is_file())
            self.assertTrue((Path(d)/'learning_curves.csv').is_file())


if __name__=='__main__':unittest.main()
