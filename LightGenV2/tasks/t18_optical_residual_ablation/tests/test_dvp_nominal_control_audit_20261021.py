from pathlib import Path
import importlib.util
import unittest
from copy import deepcopy

SOURCE=Path(__file__).resolve().parents[1]/'lab_caltech20_dvp_nominal_control_audit_20261021.py'
spec=importlib.util.spec_from_file_location('control_audit',SOURCE)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class StimulusIdentity(unittest.TestCase):
    def test_byte_identity_not_checkpoint_filename(self):
        old=dict(sample_id='x',role='train',index=0,data_SHA256='d',phase_SHA256='p',canonical_SHA256='c',input_SHA256='b',frame_metadata=dict(exposure_us=60006.,gain=1.,format=0,bits=0,shape=[3648,5480],flip_h=False,flip_v=False,rotate=False,opposite=False,wait_ms=240))
        new=deepcopy(old);new['BMP_SHA256']=new.pop('input_SHA256')
        module.same_stimulus(old,new)
        for key in ('canonical_SHA256','BMP_SHA256'):
            changed=deepcopy(new);changed[key]='changed'
            with self.assertRaises(AssertionError):module.same_stimulus(old,changed)
        for key,value in (('exposure_us',5000.),('gain',4.),('flip_v',True)):
            changed=deepcopy(new);changed['frame_metadata'][key]=value
            with self.assertRaises(AssertionError):module.same_stimulus(old,changed)
    def test_no_training_or_device_call(self):
        text=SOURCE.read_text()
        for token in ('.Bench(','.settings(','.backward(','.step('):self.assertNotIn(token,text)
        self.assertIn('all2011_canonical_input_BMP_phase_ID_and_device_readbacks_identical=True',text)


if __name__=='__main__':unittest.main()
