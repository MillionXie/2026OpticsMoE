"""Presentation selection cannot silently substitute another precision or batch."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from LightGenV2.tasks.t06_video_quality_assessment.reports import plot_strict_resource_20261020 as plot


class Selection(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.data=dict(status='all48_complete',accounting_protocol=plot.source.ACCOUNTING,
                       optical={'time_ms':10.,'energy_j':1.,'srcc':.8},values=[])
        for model in plot.DEPLOYED_MODELS:
            dtypes=(['torch.float16','torch.float32'] if model=='clip' else
                    ['torch.float32'] if model=='yolo' else ['torch.bfloat16'])
            for cap in (100,50):
                directory=self.root/f'{model}{cap}'
                directory.mkdir()
                (directory/'identity.json').write_text(json.dumps(dict(
                    parameter_dtypes=dtypes,visible_multiprocessors=108 if cap==100 else 54)))
                repeats=[dict(video_evaluations=8,forward_calls=8,elapsed_s=1.,energy_j=10.) for _ in range(3)]
                self.data['values'].append(dict(model=model,precision='native',method='mps',
                    condition=f'mps_cap{cap}',resource_cap=cap,batch=1,gpu_count=1,repeats=repeats,
                    checkpoint_sha256=model,accuracy_source_sha256='accuracy',
                    source=f'{model}{cap}/mps_cap{cap}/report.json',time16_ms=2000.,energy16_j=20.))
        self.patcher=patch.object(plot.source,'REPO',self.root)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop();self.temp.cleanup()

    def test_native_only_fixed_batch_and_readout(self):
        decoy=copy.deepcopy(self.data['values'][0]);decoy.update(precision='fp32',time16_ms=1.)
        self.data['values'].append(decoy)
        result=plot.selected_mps(self.data)
        self.assertEqual(len(result['values']),12)
        self.assertTrue(all(p['batch']==1 and p['precision']=='native' for p in result['values']))
        self.assertTrue(all(p['quality_readout_precision']=='FP32' for p in result['values']))
        self.assertTrue(all(p['time16_ms']==2000. for p in result['values']))

    def test_batch_change_rejected(self):
        self.data['values'][0]['batch']=2
        with self.assertRaises(AssertionError):plot.selected_mps(self.data)

    def test_precision_relabel_rejected(self):
        (self.root/'qwen100/identity.json').write_text(json.dumps(dict(
            parameter_dtypes=['torch.float32'],visible_multiprocessors=108)))
        with self.assertRaises(AssertionError):plot.selected_mps(self.data)

    def test_accuracy_substitution_rejected(self):
        self.data['values'][0]['accuracy_source_sha256']='different-cap-reference'
        with self.assertRaises(AssertionError):plot.selected_mps(self.data)

    def test_duplicate_or_missing_condition_rejected(self):
        self.data['values'].pop()
        with self.assertRaises(AssertionError):plot.selected_mps(self.data)


if __name__=='__main__':unittest.main()
