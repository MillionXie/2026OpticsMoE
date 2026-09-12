"""Six-stage SOFTWARE simulation of scheduling only; no simulated CCD enters real workflow."""
import tempfile,unittest
from pathlib import Path
from unittest.mock import patch,Mock
from guarded_workflow import execute,write,read,STAGES
from phase_fingerprint import digest
from phase_hdmi import sha

class Owner:
    def __init__(self,*args):self.audit=[]
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def recover(self,*args):pass

class Remote:
    root='test'
    def __init__(self,files,phase_hashes):self.files=files;self.hashes=phase_hashes;self.jobs=[];self.records={}
    def exists(self,p):return p in self.records
    def read(self,p):return self.records[p]
    def ps(self,*args):pass
    def putjson(self,p,obj):self.records[p]=obj
    def download(self,p,dest):dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(self.files[p])
    def job(self,s):
        self.jobs.append(s);action=s['action'];session=s.get('session','s')
        if action=='init':self.records[f'sessions/{session}/session.json']={'test_queries':4}
        if action=='prepare':
            stage=s['stage'];self.records[f'sessions/{session}/play/{stage}/manifest.json']={'entries':[{'id':'image_1'}],
                'phase_file':stage+'.bmp','phase_sha256':self.hashes[stage]}
        if action in ('capture_batch','accept_batch','quarantine_batch'):
            path=f"sessions/{session}/phase_batches/{s['batch_id']}.json"
            self.records[path]={'state':{'capture_batch':'pending','accept_batch':'accepted','quarantine_batch':'quarantined'}[action]}
        if action=='evaluate':self.files[f'sessions/{session}/results/metrics.json']=b'{"software_fixture":true}'

class Guard:
    fail_pre=False
    def __init__(self,*args):self.first=True
    def ensure(self,stage,path,label):
        if self.fail_pre:raise RuntimeError('Deliberate rejected precheck')
        return {'phase_sha256':sha(path),'optical_verification':{'passed':True,'target':stage,'bank_id':'fixture'}},[]
    def postcheck(self,stage,*args):
        passed=not self.first;self.first=False
        return {'passed':passed,'target':stage}

class CoordinatorTests(unittest.TestCase):
    def fixture(self,out):
        lut=out/'lut';lut.write_bytes(b'lut');probe=b'probe'
        files={'probe.bmp':probe};phasepaths={};hashes={}
        for stage in [*STAGES,'flat','lens']:
            path=out/(stage+'.bmp');path.write_bytes(stage.encode());phasepaths[stage]=str(path);hashes[stage]=sha(path);files[stage+'.bmp']=stage.encode()
        c={'geometry_confirmed':True,'camera':{'gain':'Gain_X4'},'phase_slm':{'lut_sha256':sha(lut)},
           'logical_corners_full_sensor_xy':dict(zip(['top_left','top_right','bottom_right','bottom_left'],[[10,10],[500,10],[500,500],[10,500]]))}
        import hashlib
        bank={'approved':False,'hardware_config_sha256':digest(c),'lut_sha256':sha(lut),'phase_sha256':hashes,
              'phase_paths':phasepaths,'probe_remote':'probe.bmp','probe_sha256':hashlib.sha256(probe).hexdigest()}
        bank['bank_id']=digest(bank);bank['approved']=True;bank['approval']={'bank_id':bank['bank_id']};write(out/'bank.json',bank)
        return Remote(files,hashes),c,{'phase_lut':str(lut),'orientation_and_phase_response_verified':True},out/'bank.json'
    def test_six_stages_quarantine_retry_then_evaluate(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);remote,c,link,bank=self.fixture(out)
            with patch('guarded_workflow.PhaseOwner',Owner),patch('guarded_workflow.Guard',Guard):
                execute(remote,c,link,out,'s',4,bank)
            self.assertEqual([j['stage'] for j in remote.jobs if j['action']=='prepare'],STAGES)
            self.assertEqual(sum(j['action']=='capture_batch' for j in remote.jobs),7)
            self.assertEqual(sum(j['action']=='quarantine_batch' for j in remote.jobs),1)
            self.assertEqual(remote.jobs[-1]['action'],'evaluate')
            self.assertEqual([r['state'] for r in read(out/'journal.json')['batches']].count('accepted'),6)
    def test_failed_precheck_never_starts_capture(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);remote,c,link,bank=self.fixture(out)
            with patch('guarded_workflow.PhaseOwner',Owner),patch('guarded_workflow.Guard',Guard),patch.object(Guard,'fail_pre',True):
                with self.assertRaises(RuntimeError):execute(remote,c,link,out,'s',4,bank)
            self.assertFalse(any(j['action'] in ('capture_batch','evaluate') for j in remote.jobs))
    def test_old_unguarded_session_cannot_skip_verification(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);remote,c,link,bank=self.fixture(out)
            remote.records['sessions/s/session.json']={'test_queries':4};remote.records['sessions/s/ccd']={}
            remote.sftp=Mock();remote.sftp.listdir.return_value=['image_old']
            with self.assertRaisesRegex(ValueError,'unguarded CCDs'):execute(remote,c,link,out,'s',4,bank)
            self.assertFalse(remote.jobs)
    def test_remote_bank_binding_cannot_change(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);remote,c,link,bank=self.fixture(out)
            remote.records['sessions/s/session.json']={'test_queries':4};remote.records['sessions/s/guard_identity.json']={'bank_id':'other'}
            with self.assertRaisesRegex(ValueError,'another reference'):execute(remote,c,link,out,'s',4,bank)
            self.assertFalse(remote.jobs)

if __name__=='__main__':unittest.main()
