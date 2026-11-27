import contextlib
import csv
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scipy.stats import spearmanr


MODULES=[importlib.import_module(name+'.infer') for name in ('LightGen-II','LightGen-II-MLP')]


class ReferenceResultsTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.root=Path(self.temporary.name).resolve()
        self.reference=self.root/'reference_results'
        self.reference.mkdir()
        self.rows=[dict(sample_id='video_'+str(i),target_MOS=float(i),predicted_MOS=float(i)+.2)
                   for i in range(8)]
        with (self.reference/'01_predictions.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=['sample_id','target_MOS','predicted_MOS'])
            writer.writeheader();writer.writerows(self.rows)
        self.metrics=dict(videos=8,SRCC=float(spearmanr([r['target_MOS'] for r in self.rows],
                                                      [r['predicted_MOS'] for r in self.rows]).statistic))
        (self.reference/'02_metrics.json').write_text(json.dumps(self.metrics),encoding='utf-8')
        self.original={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in self.reference.iterdir()}

    def tearDown(self):
        for p in self.reference.iterdir():
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),self.original[p.name])
        self.temporary.cleanup()

    def test_order_independent_comparison_accepts_small_numerical_differences(self):
        observed=[dict(row) for row in reversed(self.rows)]
        observed[0]['predicted_MOS']+=.0001
        for module in MODULES:
            with self.subTest(module=module.__name__),contextlib.redirect_stdout(io.StringIO()) as terminal:
                result=module.compare_reference_results(observed,self.metrics,self.reference)
                self.assertTrue(result['passed'])
                self.assertIn('Reference comparison: PASS',terminal.getvalue())

    def test_material_prediction_difference_fails(self):
        observed=[dict(row) for row in self.rows]
        observed[0]['predicted_MOS']+=.01
        for module in MODULES:
            with self.subTest(module=module.__name__),contextlib.redirect_stdout(io.StringIO()),self.assertRaises(RuntimeError):
                module.compare_reference_results(observed,self.metrics,self.reference)

    def test_duplicate_identity_or_wrong_labels_fail(self):
        for module in MODULES:
            duplicate=[dict(row) for row in self.rows];duplicate[-1]=duplicate[0]
            wrong_labels=[dict(row) for row in self.rows];wrong_labels[0]['target_MOS']+=.1
            for rows in (duplicate,wrong_labels):
                with self.subTest(module=module.__name__),self.assertRaises(ValueError):
                    module.compare_reference_results(rows,self.metrics,self.reference)

    def test_inconsistent_reported_srcc_fails(self):
        for module in MODULES:
            with self.subTest(module=module.__name__),self.assertRaises(ValueError):
                module.compare_reference_results(self.rows,dict(videos=8,SRCC=.5),self.reference)

    def test_reference_directory_descendant_and_ancestor_are_rejected(self):
        for module in MODULES:
            for directory in (self.reference,self.reference/'child',self.root):
                with self.subTest(module=module.__name__,directory=str(directory)),self.assertRaises(ValueError):
                    module.check_output_directory(directory,self.reference)
            self.assertEqual(module.check_output_directory(self.root/'results',self.reference),self.root/'results')

    def test_hard_link_cannot_overwrite_reference_files(self):
        output=self.root/'results'
        output.mkdir()
        os.link(self.reference/'01_predictions.csv',output/'01_predictions.csv')
        for module in MODULES:
            with self.subTest(module=module.__name__),self.assertRaises(ValueError):
                module.check_output_directory(output,self.reference)

    def test_main_rejects_reference_output_before_loading_model(self):
        for module in MODULES:
            with self.subTest(module=module.__name__),patch.object(module,'load_lightgen2') as loader:
                with patch('sys.argv',['infer.py','--output',str(module.PROJECT/'reference_results')]),self.assertRaises(ValueError):
                    module.main()
                loader.assert_not_called()


if __name__=='__main__':unittest.main()
