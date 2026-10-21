from pathlib import Path
import ast
import importlib.util
import unittest

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'lab_caltech20_dvp_phase_export_review_20261021.py'
spec=importlib.util.spec_from_file_location('export_review',SOURCE)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class ExportReview(unittest.TestCase):
    def test_complete_8bit_complement_identity(self):
        for gray in range(256):self.assertEqual(module.device_gray_turns(gray/256),(255-gray)/256)
    def test_no_fit_or_device(self):
        text=SOURCE.read_text();ast.parse(text)
        for bad in ('.backward(','.step(','.Bench(','.capture(','.settings('):self.assertNotIn(bad,text)
        self.assertIn('optimizer_updates=0',text);self.assertIn('TEST_used=False',text)
        self.assertIn('geometry_LUT_exposure_unchanged=True',text)


if __name__=='__main__':unittest.main()
