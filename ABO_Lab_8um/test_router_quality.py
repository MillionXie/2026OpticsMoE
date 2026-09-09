import copy
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from common import config,ROOT,write,read,sha,hardware_identity
from router_quality import assess,require_accepted
from hardware import canonical
import recover_router
import run


def route():
    return {'probabilities':[[.514,.161,.164,.161]],'detector_energy':[[19685,8036,8218,8021]],
            'selected_indices':[[0,2]],'weights':[[.95265,0,.30407,0]]}


def fixture_validator(frame,r,c):
    q=c['router_quality']
    assert q['minimum_topk_probability_margin']==0
    if np.percentile(frame,99)<q['minimum_p99_uint8']: raise ValueError('insufficient_brightness')


class QualityTests(unittest.TestCase):
    def setUp(self):
        self.contract={'router_quality':{'minimum_topk_probability_margin':.01,
                         'minimum_p99_uint8':8.,'maximum_saturated_pixel_fraction':.02,
                         'minimum_dynamic_range_uint8':4.}}
        self.frame=np.full((478,478),20,np.uint8)

    def test_margin_warning_no_route_or_contract_mutation(self):
        r=route(); before=copy.deepcopy((r,self.contract))
        q=assess(self.frame,r,self.contract,fixture_validator)
        self.assertTrue(q['accepted']); self.assertEqual(q['warnings'],['ambiguous_top2_margin'])
        self.assertAlmostEqual(q['top2_probability_margin'],.003)
        self.assertEqual((r,self.contract),before)

    def test_other_hard_failure_propagates(self):
        for reason in ['insufficient_brightness','excess_saturation','insufficient_dynamic_range','uniform_detector_region_energy']:
            def reject(*args): raise ValueError(reason)
            q=assess(self.frame,route(),self.contract,reject)
            self.assertFalse(q['accepted'])
            with self.assertRaisesRegex(ValueError,reason): require_accepted(q)

    def test_exact_tie_does_not_modify_top2(self):
        r=route(); r['probabilities']=[[.514,.162,.162,.162]]
        q=assess(self.frame,r,self.contract,fixture_validator)
        self.assertTrue(q['accepted']); self.assertEqual(q['top2_probability_margin'],0)
        self.assertEqual(r['selected_indices'],[[0,2]])

    def test_bad_frame_fails(self):
        with self.assertRaises(ValueError): assess(self.frame.astype(float),route(),self.contract,fixture_validator)

    def test_original_imported_validator_when_available(self):
        if not (ROOT/'runtime/abo_dual/common.py').is_file(): self.skipTest('Imported runtime is not Git source')
        from common import setup_imports
        setup_imports()
        from abo_dual.common import validate_router_capture
        c=copy.deepcopy(self.contract); c['router']={'eps':1e-8}
        frame=self.frame.copy(); frame[:239]=1
        with self.assertRaisesRegex(ValueError,'ambiguous_top2_margin'):
            validate_router_capture(frame,route(),c)
        self.assertTrue(assess(frame,route(),c)['accepted'])
        for raw in [np.full_like(frame,2),np.full_like(frame,255),np.full_like(frame,20)]:
            self.assertFalse(assess(raw,route(),c)['accepted'])
        r=route(); r['detector_energy']=[[5,5,5,5]]
        self.assertFalse(assess(frame,r,c)['accepted'])

    def test_recovery_checks_pixels_and_is_idempotent(self):
        self.recovery_case()

    def test_recovery_keeps_rejecting_dark_frame(self):
        self.recovery_case(dark=True)

    def test_recovery_rejects_modified_canonical_png(self):
        self.recovery_case(tamper=True)

    def recovery_case(self,dark=False,tamper=False):
        c,_=config(ROOT/'lab.json'); c['geometry_confirmed']=True
        c['logical_corners_full_sensor_xy']=dict(zip(('top_left','top_right','bottom_right','bottom_left'),
            [[0,0],[477,0],[477,477],[0,477]]))
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp); session=base/'sessions/test'; played=session/'play/language_router'
            out=session/'ccd/title_000/language_router'; out.parent.mkdir(parents=True); played.mkdir(parents=True)
            bmp=Image.fromarray(self.frame); bmp.save(played/'00000.bmp'); bmp.save(base/'phase.bmp')
            write(session/'session.json',{'hardware_identity':hardware_identity(c),
                'samples':[{'id':'title_000','kind':'title'},{'id':'title_001','kind':'title'}]})
            write(session/'optical_contract.json',self.contract)
            write(played/'manifest.json',{'stage':'language_router','hardware_identity':hardware_identity(c),
                'phase_file':'phase.bmp','phase_sha256':sha(base/'phase.bmp'),
                'entries':[{'id':i,'bmp':'00000.bmp','sha256':sha(played/'00000.bmp')} for i in ['title_000','title_001']]})
            raw=np.full((478,478),2 if dark else 20,np.uint8); frame=canonical(raw,c)
            Image.fromarray(raw).save(out.with_suffix('.tif')); Image.fromarray(frame).save(out.with_suffix('.png'))
            write(out.with_suffix('.json'),{'hardware_identity':hardware_identity(c),'raw_shape':[478,478],'raw_dtype':'uint8'})
            source_files=[out.with_suffix(x) for x in ['.png','.tif','.json']]
            if tamper:
                frame[200,200]+=1; Image.fromarray(frame).save(out.with_suffix('.png'))
            before={str(p):sha(p) for p in source_files}
            fake=types.ModuleType('abo_dual.common'); fake.routing_from_ccd=lambda f,c:route(); fake.validate_router_capture=fixture_validator
            with patch.dict('sys.modules',{'abo_dual.common':fake}),patch('common.setup_imports'),\
                 patch.object(recover_router,'setup_imports'),patch.object(recover_router,'ROOT',base),\
                 patch.object(run,'session_path',lambda name:session):
                with self.assertRaisesRegex(ValueError,'Legacy CCD'): recover_router.recover('test','language_router',c)
                if dark or tamper:
                    with self.assertRaises(ValueError): recover_router.recover('test','language_router',c,True)
                    self.assertFalse(out.with_suffix('.record.json').exists())
                else:
                    result=recover_router.recover('test','language_router',c,True)
                    self.assertEqual(result['recovered'],['title_000']); self.assertEqual(result['not_yet_captured'],1)
                    self.assertFalse(result['hardware_opened'])
                    self.assertEqual(run.measured(session,{'id':'title_000'})['language_router'],route())
                    record=sha(out.with_suffix('.record.json'))
                    self.assertEqual(recover_router.recover('test','language_router',c,True)['already_valid'],['title_000'])
                    self.assertEqual(sha(out.with_suffix('.record.json')),record)
            self.assertEqual(before,{str(p):sha(p) for p in source_files})


if __name__=='__main__': unittest.main()
