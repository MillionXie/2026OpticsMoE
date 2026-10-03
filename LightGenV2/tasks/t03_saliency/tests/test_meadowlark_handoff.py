import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import yaml
from PIL import Image
from LightGenV2.tasks.t03_saliency import meadowlark_cli as m
from LightGenV2.tasks.t06_video_quality_assessment.lab_runtime import write
from LightGenV2.tasks.t06_video_quality_assessment.lab_bench import raster


class MeadowlarkTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        (self.root/'lut').write_bytes(b'test LUT')
        (self.root/'geometry').write_bytes(b'test geometry identity')
        self.raw=dict(amplitude_slm=dict(driver='meadowlark_pcie',pixel_pitch_um=17,expected_resolution_wh=[1024,1024],lut_file='lut'),
            phase_slm=dict(driver='manual',pixel_pitch_um=8,expected_resolution_wh=[1920,1200]),
            amplitude_roi=dict(width=478,height=478,center_x=511.5,center_y=511.5),
            camera=dict(driver='tucam',exposure_us=3500,saved_frame_size_wh=[478,478],saved_frame_bit_depth=8,saved_frame_input_range=[0,65535],
                detector_geometry=dict(enabled=True,contract_file='geometry',expected_file_sha256=m.sha(self.root/'geometry'))),
            output_extension='.png',require_phase_mask=True,confirm_before_start=True,max_files=None,settle_delay_ms=200)
        self.yaml=self.root/'hardware.yaml';self.save_yaml()
        self.a=SimpleNamespace(config=self.root/'LAB.json',hardware_config=self.yaml,phase_center=[960,600],
            phase_gray_encoding='inverted_255_minus_g',phase_orientation='h',amplitude_orientation='none',stage='vision_router')
        m.bind(self.a)

    def save_yaml(self):
        self.yaml.write_text(yaml.safe_dump(self.raw),encoding='utf-8')

    def test_binding_and_real_raster_encoding(self):
        c=m.verify_binding(self.a)
        self.assertEqual(c['amplitude_slm']['center_xy'],[512,512])
        b=raster(np.zeros((478,478)),c['phase_slm'],'phase')
        self.assertEqual(b.shape,(1200,1920));self.assertTrue((b==255).all())
        (self.root/'lut').write_bytes(b'different LUT')
        with self.assertRaises(ValueError):m.verify_binding(self.a)

    def test_forbid_auto_exposure_and_old_config(self):
        self.raw['camera']['auto_exposure']=True;self.save_yaml()
        with self.assertRaises(ValueError):m.hardware(self.yaml)
        with self.assertRaises(ValueError):m.verify_binding(self.a)

    def prepare_fake_capture(self):
        c=m.read(self.a.config);s=self.root/'session';out=s/'ccd'/self.a.stage;out.mkdir(parents=True)
        image=out/'field_00000.png';Image.fromarray(np.full((478,478),31,np.uint8)).save(image)
        e=dict(key='field_00000',sha256='amplitude',upstream_ccd_sha256={})
        mf=dict(entries=[e],phase_sha256='phase')
        state=dict(hardware_sha256='hardware',measured_stages=[])
        row=dict(ccd_capture=image.name,output_sha256=m.sha(image),amplitude_bmp_sha256='amplitude',phase_mask_sha256='phase',
            saved_frame_orientation='canonical_model_xy',detector_geometry_file_sha256=c['native_sdk']['assets']['geometry']['sha256'],
            per_frame_minmax_normalization='False',background_subtraction='False')
        log=s/'sdk_log'/self.a.stage;log.mkdir(parents=True)
        def save():
            with (log/'capture_manifest.csv').open('w',newline='',encoding='utf-8') as h:
                w=csv.DictWriter(h,fieldnames=list(row));w.writeheader();w.writerow(row)
        save()
        return s,state,c,mf,image,row,save

    def test_import_preserves_pixels_and_marks_stage(self):
        s,state,c,mf,image,row,save=self.prepare_fake_capture();before=m.sha(image)
        with patch.object(m,'prepared',return_value=(s,state,c,mf,None)):m.import_captures(self.a)
        self.assertEqual(m.sha(image),before)
        self.assertEqual(m.read(s/'session.json')['measured_stages'],['vision_router'])
        self.assertEqual(m.read(image.with_suffix('.record.json'))['quality']['p99'],31)

    def test_reject_wrong_phase_or_noncanonical_output(self):
        s,state,c,mf,image,row,save=self.prepare_fake_capture()
        for key,value in [('phase_mask_sha256','wrong'),('saved_frame_orientation','native')]:
            old=row[key];row[key]=value;save()
            with patch.object(m,'prepared',return_value=(s,state,c,mf,None)):
                with self.assertRaises(ValueError):m.import_captures(self.a)
            self.assertFalse((s/'session.json').exists());row[key]=old


if __name__=='__main__':unittest.main()
