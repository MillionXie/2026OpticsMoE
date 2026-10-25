import json
from pathlib import Path
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from .fit_lab_terminal import evaluate, selection, validate


class TerminalHardwareFitTests(unittest.TestCase):
    def test_optimizer_scaling_is_folded_into_existing_bias_free_head(self):
        torch.manual_seed(73)
        x=torch.rand(40,784)
        scale=x.square().mean(0).sqrt().clamp_min(1e-3)
        parameter=torch.randn(10,784)
        torch.testing.assert_close(F.linear(x/scale,parameter), F.linear(x,parameter/scale),atol=3e-5,rtol=3e-5)
        self.assertEqual((parameter/scale).numel(),7840)

    def test_one_joint_selection_and_no_upper_score_clipping(self):
        def scores(a,b):
            return dict(RGB=dict(n=1000,correct=a,nll=1),SAR=dict(n=1000,correct=b,nll=1))
        self.assertGreater(selection(scores(819,680),'LG'),selection(scores(839,670),'LG'))
        self.assertGreater(selection(scores(900,800),'LG'),selection(scores(819,680),'LG'))
        self.assertEqual(selection(scores(748,541),'D2NN')[0],0)
        with self.assertRaises(ValueError):
            selection(dict(RGB=dict(n=999,correct=819),SAR=dict(n=1000,correct=680)),'LG')

    def test_original_evaluation_has_all_ids_and_confusion_counts(self):
        weight=torch.eye(10)
        x=torch.eye(10).repeat(2,1);y=torch.arange(10).repeat(2)
        r=evaluate(weight,x,y,np.repeat([0,1],10),np.arange(20))
        self.assertEqual(r['RGB']['correct'],10)
        self.assertEqual(r['SAR']['indices'],list(range(10,20)))
        self.assertEqual(np.asarray(r['SAR']['confusion_matrix']).trace(),10)

    def test_exact_bounded_profile_no_TEST_teacher(self):
        cfg=json.loads(Path(__file__).with_name('config_terminal_hardware_20261026.json').read_text())
        validate(cfg)
        self.assertFalse(cfg['TEST_teacher']);self.assertFalse(cfg['TEST_gradients'])
        self.assertEqual(cfg['trainable'],['shared_head.weight'])
        with self.assertRaises(ValueError):
            validate(dict(cfg,epochs=99999))


if __name__=='__main__':
    unittest.main()
