"""Pure admission-policy tests; no Torch import, SDK, data or training."""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest

SOURCE=Path(__file__).resolve().parents[1]/'lab_caltech20_dvp_conditioned_head_20261020.py'


class AdmissionTests(unittest.TestCase):
    def fixture(self,depth=4,prefix=4):
        root=Path('strict_admission_fixture').resolve();session=root/f'L{depth}_strict8pp_20261020';other=6 if depth==4 else 4
        a=SimpleNamespace(sequence=root,depth=depth,prefix=prefix,mode='row-fit',attempt='strict8pp_20261020')
        source=root/'task/lab_caltech20_dvp_layerwise_20261019.py';store={str(root/'assets/data_manifest.json'):dict(sha256='DATA')}
        digest=lambda p:'sha:'+str(p)
        for stage in ['router']+[str(i) for i in range(prefix-1)]:
            p=session/'capture'/stage/'report.json';arrays=dict(train=dict(samples=1600,SHA256='TR'),test=dict(samples=411,SHA256='TE'))
            r=dict(state='complete',TRAIN=1600,TEST=411,physical_ccd=2011,sdk_released=True,data_SHA256='DATA',PT_SHA256='PT',phase_SHA256='PHASE',arrays=arrays)
            v=dict(passed=True,SDK_released=True,all_receipts_and_array_SHA_verified=True,report_SHA256=digest(p),contract_SHA256=digest(p.with_name('contract.json')),PT_SHA256='PT',phase_SHA256='PHASE',arrays=deepcopy(arrays))
            store[str(p)]=r;store[str(p.with_name('independent_capture_audit.json'))]=v
        args=['python',str(source),'--mode','capture','--depth',str(other),'--attempt',a.attempt,'--sequence',str(root),'--stage','0']
        owner=SimpleNamespace(pid=42,cmdline=lambda:args,create_time=lambda:123.)
        lock=dict(pid=42,camera='DVP',pilot=False,session=str(root/f'L{other}_strict8pp_20261020'),stage='0')
        core=SimpleNamespace(__file__=str(source),session_for=lambda a:session,read=lambda p:store[str(p)],sha=digest)
        node=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='strict_parallel_capture_proof')
        scope=dict(Path=Path,core=core);exec(compile(ast.Module(body=[node],type_ignores=[]),'<admission>','exec'),scope)
        return a,lock,owner,store,args,scope['strict_parallel_capture_proof']

    def test_completed_other_depth_works_both_directions(self):
        for depth,prefix in ((4,4),(6,1)):
            a,lock,owner,store,args,run=self.fixture(depth,prefix);proof=run(a,lock,owner)
            self.assertTrue(proof['other_depth_only']);self.assertEqual(len(proof['complete_measured_prefix_proofs']),prefix)
            self.assertFalse(proof['SDK_used']);self.assertFalse(proof['GPU_used'])

    def test_same_depth_and_calibration_locks_rejected(self):
        for key,value in (('session',str(Path('strict_admission_fixture/L4_strict8pp_20261020').resolve())),('camera','calibration'),('pilot',True)):
            a,lock,owner,store,args,run=self.fixture();lock[key]=value
            with self.assertRaises(AssertionError):run(a,lock,owner)

    def test_foreign_script_or_attempt_rejected(self):
        for key,value in ((1,'foreign/lab_caltech20_dvp_layerwise_20261019.py'),(7,'old_attempt')):
            a,lock,owner,store,args,run=self.fixture();args[key]=value
            with self.assertRaises(AssertionError):run(a,lock,owner)

    def test_incomplete_or_wrong_arrays_rejected(self):
        for key,value in (('state','capturing'),('sdk_released',False),('data_SHA256','FOREIGN'),('physical_ccd',24)):
            a,lock,owner,store,args,run=self.fixture();report=next(v for k,v in store.items() if Path(k).name=='report.json');report[key]=value
            with self.assertRaises(AssertionError):run(a,lock,owner)
        a,lock,owner,store,args,run=self.fixture();audit=next(v for k,v in store.items() if k.endswith('independent_capture_audit.json'));audit['arrays']['test']['SHA256']='wrong'
        with self.assertRaises(AssertionError):run(a,lock,owner)

    def test_no_hardware_calls_or_lock_mutation(self):
        node=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='strict_parallel_capture_proof')
        text=ast.unparse(node)
        for token in ('Bench(','.unlink(','.open(','.write_text(','.cuda('):self.assertNotIn(token,text)


if __name__=='__main__':unittest.main()
