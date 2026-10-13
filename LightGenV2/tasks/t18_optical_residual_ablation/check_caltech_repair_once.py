"""Synthetic Linux protocol checks only. No scientific model/dataset/test inference."""
import argparse,copy,json,os,stat,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import caltech_repair_once as f

class ProtocolChecks(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.data=self.root/'synthetic.bin';self.data.write_bytes(b'protocol fixture, not image arrays');self.previous=f.DATA_SHA;f.DATA_SHA=f.sha(self.data);self.final=self.root/'final';fits={};self.checkpoints={}
        for key in f.KEYS:
            depth=int(key[1]);rho=0. if key.endswith('rho0') else .3;cp=self.root/(key+'.synthetic');cp.write_bytes(key.encode());self.checkpoints[key]=cp
            fits[key]=dict(depth=depth,rho=rho,selected_epoch=2,epochs_completed=3,selection_split='val',selected_state_kind='ema',saved_CPU_verified=True,test_evaluations=0,checkpoint_sha256=f.sha(cp),initial_parameter_sha256='a'*64,data_sha256=f.DATA_SHA,sources={'synthetic_only.py':'b'*64},config=dict(depth=depth,residual_rho=rho,contract=dict(rho=rho,main_depth=depth,graph='synthetic protocol fixture')),orders=[f.identity([depth,k,'order']) for k in range(3)],transforms=[f.identity([depth,k,'transform']) for k in range(3)])
        self.receipt=dict(receipt_kind='formal_repaired_saved_CPU_fit_audit',passed=True,all_six_fit_audited=True,all_fit_processes_exited=True,test_evaluations=0,test_arrays_materialized=0,data_sha256=f.DATA_SHA,formal_profile_sha256='c'*64,fits=fits,scope='Synthetic filesystem protocol fixture; no real fit audit or locks')
    def tearDown(self):f.DATA_SHA=self.previous;self.tmp.cleanup()
    def ready(self):f.seal(self.final,self.receipt);f.claim_suite(self.final)
    def test_incomplete_all6_rejected_without_directory(self):
        self.receipt['fits'].pop('L3_rho03')
        with self.assertRaises(AssertionError):f.seal(self.final,self.receipt)
        self.assertFalse(self.final.exists())
    def test_alive_or_test_used_audit_rejected(self):
        for field,value in [('all_fit_processes_exited',False),('test_evaluations',1),('test_arrays_materialized',1)]:
            r=copy.deepcopy(self.receipt);r[field]=value
            with self.assertRaises(AssertionError):f.seal(self.final,r)
            self.assertFalse(self.final.exists())
    def test_pair_scientific_mismatch_rejected(self):
        self.receipt['fits']['L4_rho03']['config']['contract']['graph']='different'
        with self.assertRaises(AssertionError):f.seal(self.final,self.receipt)
        self.assertFalse(self.final.exists())
    def test_raw_or_test_selection_rejected(self):
        for field,value in [('selected_state_kind','raw'),('selection_split','test')]:
            r=copy.deepcopy(self.receipt);r['fits']['L2_rho03'][field]=value
            with self.assertRaises(AssertionError):f.seal(self.final,r)
            self.assertFalse(self.final.exists())
    def test_existing_partial_directory_preserved(self):
        self.final.mkdir();m=self.final/'partial';m.write_bytes(b'keep')
        with self.assertRaises(AssertionError):f.seal(self.final,self.receipt)
        self.assertEqual(m.read_bytes(),b'keep')
    def test_suite_and_success_refuse_repeat(self):
        self.ready();key='L4_rho03';calls=[]
        def callback(selection):
            self.assertTrue((self.final/key/'test_started.json').exists());calls.append('once');return dict(synthetic=True)
        f.test_once(self.final,key,self.checkpoints[key],self.data,callback)
        with self.assertRaises(FileExistsError):f.claim_suite(self.final)
        with self.assertRaises(AssertionError):f.test_once(self.final,key,self.checkpoints[key],self.data,callback)
        self.assertEqual(calls,['once']);self.assertEqual(f.read(self.final/key/'test_completed.json')['test_evaluations'],1)
    def test_failure_claim_and_reason_preserved_no_loader_retry(self):
        self.ready();key='L3_rho03';calls=[]
        def callback(selection):
            self.assertTrue((self.final/key/'test_started.json').exists());calls.append('after_claim');raise RuntimeError('synthetic loader refusal')
        with self.assertRaises(RuntimeError):f.test_once(self.final,key,self.checkpoints[key],self.data,callback)
        before=f.sha(self.final/key/'test_failure.json')
        with self.assertRaises(AssertionError):f.test_once(self.final,key,self.checkpoints[key],self.data,callback)
        self.assertEqual(calls,['after_claim']);self.assertEqual(f.sha(self.final/key/'test_failure.json'),before)
    def test_file_and_directory_fsync_precede_loader(self):
        self.ready();key='L2_rho0';events=[];original=os.fsync
        def fsync(fd):events.append('directory' if stat.S_ISDIR(os.fstat(fd).st_mode) else 'file');return original(fd)
        def callback(selection):events.append('loader');return dict(synthetic=True)
        with patch.object(f.os,'fsync',fsync):f.test_once(self.final,key,self.checkpoints[key],self.data,callback)
        self.assertEqual(events[:3],['file','directory','loader'])
    def test_fsync_failure_keeps_exclusive_marker(self):
        marker=self.root/'synthetic_claim.json'
        with patch.object(f.os,'fsync',side_effect=OSError('synthetic fsync failure')):
            with self.assertRaises(OSError):f.durable(marker,dict(original=True))
        self.assertTrue(marker.exists())
        with self.assertRaises(FileExistsError):f.durable(marker,dict(replacement=True))
        self.assertEqual(f.read(marker),dict(original=True))
    def test_tampered_identity_refused_before_loader(self):
        self.ready();key='L2_rho03';self.checkpoints[key].write_bytes(b'changed');calls=[]
        with self.assertRaises(AssertionError):f.test_once(self.final,key,self.checkpoints[key],self.data,lambda _:calls.append(1))
        self.assertFalse((self.final/key/'test_started.json').exists());self.assertEqual(calls,[])
    def test_tampered_lock_refused(self):
        self.ready();key='L4_rho0';p=self.final/key/'selection_lock.json';s=f.read(p);s['epoch']=1;p.write_text(json.dumps(s))
        with self.assertRaises(AssertionError):f.validate_locks(self.final)
    def test_suite_required_before_test_claim(self):
        f.seal(self.final,self.receipt);key='L4_rho03'
        with self.assertRaises(FileNotFoundError):f.test_once(self.final,key,self.checkpoints[key],self.data,lambda _:dict(synthetic=True))
        self.assertFalse((self.final/key/'test_started.json').exists())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();assert os.name=='posix' and os.environ.get('CUDA_VISIBLE_DEVICES')=='';a.out.mkdir(parents=True,exist_ok=False)
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ProtocolChecks);result=unittest.TextTestRunner(verbosity=2).run(suite);receipt=dict(passed=result.wasSuccessful(),checks=result.testsRun,failures=len(result.failures),errors=len(result.errors),sources={n:f.sha(Path(__file__).with_name(n)) for n in ('caltech_repair_once.py','check_caltech_repair_once.py')},scope='Synthetic temporary filesystem protocol fixtures only; no real final locks or scientific result',model_forward_calls=0,test_arrays_materialized=0,test_evaluations=0,devices_opened=0,gpu_used=False)
    f.durable(a.out/('checks.json' if result.wasSuccessful() else 'failure.json'),receipt)
    raise SystemExit(0 if result.wasSuccessful() else 1)
