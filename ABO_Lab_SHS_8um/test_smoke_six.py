"""Scheduling fixtures only: NEVER hardware evidence."""
import io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from smoke_six import run,STAGES
from phase_hdmi import sha

def png(a,fmt='PNG'):
    b=io.BytesIO();Image.fromarray(a).save(b,format=fmt);return b.getvalue()

class Owner:
    def __init__(self,*args):self.display=type('Display',(),{'audit':{}})();self.audit=[]
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def show(self,path,expected=None):return {'phase_sha256':sha(path)}
    def restart(self):self.audit.append('restart')

class Remote:
    def __init__(self):
        self.jobs=[];self.files={};self.mf={};self.expected={}
        self.bmp=png(np.zeros((1200,1920),np.uint8),'BMP')
        import hashlib
        self.phase_sha=hashlib.sha256(self.bmp).hexdigest()
        rng=np.random.default_rng(1);self.a=png(rng.integers(20,120,(1080,1920),dtype=np.uint8));self.b=png(rng.integers(20,120,(1080,1920),dtype=np.uint8))
    def exists(self,path):return False
    def putjson(self,path,obj):self.files[path]=json.dumps(obj).encode()
    def job(self,s):
        self.jobs.append(s)
        if s['action']=='prepare':
            n=4 if s['stage'].startswith('vision') else 104
            self.mf[s['stage']]={'entries':[{'id':f'image_{i}'} for i in range(n)],'phase_file':'p.bmp','phase_sha256':self.phase_sha}
    def read(self,path):
        if path.endswith('manifest.json'):return self.mf[path.split('/')[-2]]
        return {'phase_sha256':self.phase_sha,'capture_mode':'real'}
    def download(self,path,dest):
        dest.parent.mkdir(parents=True,exist_ok=True)
        if path.endswith('.bmp'):data=self.bmp
        elif path.endswith('raw.png'):data=self.a if '_flat/' in path else self.b
        elif path.endswith('capture.json'):
            data=json.dumps({'camera':{k:{'value':v} for k,v in {'ExposureTime':150,'Gain':'Gain_X4','AcquisitionFrameRate':100,'PixelFormat':'Mono8'}.items()}}).encode()
        else:data=b'{"software_fixture":true}'
        dest.write_bytes(data)

class SmokeTests(unittest.TestCase):
    def test_full_requires_explicit_dataset_and_no_selection(self):
        c={'diagnostic_only':True,'diagnostic_session':'smoke_fixture','geometry_confirmed':True,
           'geometry_evidence':{'method':'measured_markers_with_asymmetric_check','report_sha256':'fixture'},
           'camera':{'gain':'Gain_X4'},'logical_corners_full_sensor_xy':dict(zip(['top_left','top_right','bottom_right','bottom_left'],[[100,100],[1500,100],[1500,900],[100,900]]))}
        with tempfile.TemporaryDirectory() as d:
            for extra,limit in [({},0),({'diagnostic_full_dataset':True},4),({'diagnostic_full_dataset':True,'diagnostic_query_indices':[0]},0)]:
                with self.assertRaisesRegex(ValueError,'Full diagnostic requires'):
                    run(Remote(),{},dict(c,**extra),'fixture.json',Path(d)/'out',limit,full_dataset=True)

    def test_full_counts_preserve_2400_queries_and_100_titles(self):
        c={'diagnostic_only':True,'diagnostic_full_dataset':True,'diagnostic_session':'smoke_fixture','geometry_confirmed':True,
           'geometry_evidence':{'method':'measured_markers_with_asymmetric_check','report_sha256':'fixture'},
           'camera':{'gain':'Gain_X4'},'logical_corners_full_sensor_xy':dict(zip(['top_left','top_right','bottom_right','bottom_left'],[[100,100],[1500,100],[1500,900],[100,900]]))}
        remote=Remote()
        def stop_after_init(spec):
            remote.jobs.append(spec)
            raise RuntimeError('fixture stop before hardware')
        remote.job=stop_after_init
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'out'
            with self.assertRaisesRegex(RuntimeError,'fixture stop'):
                run(remote,{},c,'fixture.json',out,0,full_dataset=True)
            r=json.loads((out/'report.json').read_text())
            self.assertEqual(sum(r['expected_stage_counts']),14700)
            self.assertEqual(r['queries'],2400)
            self.assertFalse(r['production_qualified'])
            self.assertEqual(remote.jobs[0]['limit'],0)

    def test_stuck_phase_stops_after_three_attempts_without_capture(self):
        c={'diagnostic_only':True,'diagnostic_session':'smoke_fixture','geometry_confirmed':True,
           'geometry_evidence':{'method':'measured_markers_with_asymmetric_check','report_sha256':'fixture'},
           'camera':{'gain':'Gain_X4'},'logical_corners_full_sensor_xy':dict(zip(['top_left','top_right','bottom_right','bottom_left'],[[100,100],[1500,100],[1500,900],[100,900]]))}
        remote=Remote();remote.a=remote.b
        with tempfile.TemporaryDirectory() as d,patch('smoke_six.PhaseOwner',Owner):
            out=Path(d)/'out'
            with self.assertRaisesRegex(RuntimeError,'after 3 attempts'):run(remote,{},c,'fixture.json',out,4)
            r=json.loads((out/'report.json').read_text());self.assertEqual(len(r['stages'][0]['preflight_attempts']),3)
        self.assertFalse(any(j['action']=='capture' for j in remote.jobs))
    def test_brightness_warning_never_accepts_wrong_shape(self):
        from smoke_six import post_state
        self.assertEqual(post_state(.999,.22),'rejected_postcheck')
        self.assertEqual(post_state(.999,.22,True),'real_capture_complete_photometric_warning')
        self.assertEqual(post_state(.90,.05,True),'rejected_postcheck')
        self.assertEqual(post_state(.999,.05,True),'real_capture_complete')
    def test_declared_query_indices_keep_all_titles(self):
        from diagnostic_config import select_queries
        samples=[{'kind':'title','id':i} for i in range(100)]+[{'kind':'image','id':i} for i in range(2400)]
        selected=select_queries(samples,[0,600,1200,1800],4)
        self.assertEqual(len(selected),104);self.assertEqual([s['id'] for s in selected[100:]],[0,600,1200,1800])
        for indices,limit in [([0,0],2),([-1],1),([2400],1),([0],4)]:
            with self.assertRaises(ValueError):select_queries(samples,indices,limit)
    def test_all_six_and_100_titles_kept(self):
        c={'diagnostic_only':True,'diagnostic_session':'smoke_fixture','geometry_confirmed':True,
           'geometry_evidence':{'method':'measured_markers_with_asymmetric_check','report_sha256':'fixture'},
           'camera':{'gain':'Gain_X4'},'logical_corners_full_sensor_xy':dict(zip(['top_left','top_right','bottom_right','bottom_left'],[[100,100],[1500,100],[1500,900],[100,900]]))}
        remote=Remote()
        with tempfile.TemporaryDirectory() as d,patch('smoke_six.PhaseOwner',Owner):
            out=Path(d)/'out';run(remote,{},c,'fixture.json',out,4)
            report=json.loads((out/'report.json').read_text())
            self.assertEqual(report['status'],'completed');self.assertEqual(report['total_real_captures'],324)
            self.assertFalse(report['production_qualified'])
        self.assertEqual([s['stage'] for s in remote.jobs if s['action']=='capture'],STAGES)
        self.assertEqual(remote.jobs[-1]['action'],'evaluate')

if __name__=='__main__':unittest.main()
