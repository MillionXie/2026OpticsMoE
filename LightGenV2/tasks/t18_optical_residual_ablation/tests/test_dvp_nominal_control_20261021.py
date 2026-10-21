from pathlib import Path
import ast
import unittest

SOURCE=Path(__file__).resolve().parents[1]/'lab_caltech20_dvp_nominal_control_20261021.py'


class NominalControl(unittest.TestCase):
    def test_exact_unchanged_control_and_full_data(self):
        text=SOURCE.read_text();ast.parse(text)
        for token in ('TRAIN=1600,TEST=411,VALID=0',"loop.validate_state(m,model,original,set())",
            "m.sha(session/'capture/0/phase.bmp')",'optimizer_updates=0',
            "evidence['old_nominal']['correct']==20",'not_unique_ambient_light_causality=True'):
            self.assertIn(token,text)
        for token in ('.cuda(','.backward(','.step(','.settings(', 'loop.gpu_admission(', 'loop.refine('):
            self.assertNotIn(token,text)


if __name__=='__main__':unittest.main()
