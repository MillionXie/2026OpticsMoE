"""Conditional single-profile depth2/3 validation after eligible L4 study; no test."""
import argparse,copy,hashlib,json,math,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import numpy as np
import torch
import caltech_repair234 as r
import audit_caltech_repair234 as audit_core
import export_caltech_repair234 as export
from check_caltech_repair234_export import reverse
HERE=Path(__file__).resolve().parent
POLICY=dict(depths=[2,3],rho=.3,maximum_epochs=150,maximum_seconds=14400,validation_accuracy_min=.50,validation_macro_min=.50,each_val_recall_min=.20,maximum_gap_pp=15,scope='One independent fresh check for each depth using eligible L4 alpha; not formal six-arm training',automatic_retry=False)
ADDITIONAL=('caltech_repair23_validate.py','export_caltech_repair234.py','check_caltech_repair234_export.py','CALTECH_REPAIR23_PROTOCOL_20261014.md')

def sources():
    src=r.sources()
    for name in ADDITIONAL:src[(HERE/name).relative_to(r.ROOT).as_posix()]=r.sha(HERE/name)
    return src

def reference(study,require_native=True):
    state=r.read(study/'status.json');meta=r.read(study/'metadata.json')
    assert len(state['completed'])==3 and state['active']==state['pending']==0 and not state['failures']
    assert audit_core.exited(meta['pid']) and all(audit_core.exited(j['pid']) and j['exit_code']==0 for j in state['completed'])
    proof=r.read(study/'cpu_report/audit.json');assert proof['passed'];key=proof['chosen_profile_case'];assert key in proof['eligible'] and proof['gates'][key]['passed']
    fit=proof['fits'][key];assert fit==r.read(study/key/'fit_result.json') and fit['depth']==4 and fit['rho']==.3
    r.engine.verify_sources(meta['sources'],meta['git_commit']);assert meta['sources']==r.sources()==fit['sources'];assert r.sha(study/key/'best_checkpoint.pt')==fit['checkpoint_sha256']
    result=dict(study=str(study.resolve()),case=key,mixup_alpha=fit['config']['mixup_alpha'],selected_epoch=fit['selected_epoch'],checkpoint_sha256=fit['checkpoint_sha256'],cpu_audit_sha256=r.sha(study/'cpu_report/audit.json'),data_sha256=fit['config']['cache_sha256'],test_evaluations=0)
    if require_native:
        receipt=r.read(study/'selected_native_bmp/checks.json');assert receipt['passed'] and receipt['checkpoint_sha256']==fit['checkpoint_sha256'] and receipt['test_evaluations']==receipt['test_arrays_read']==receipt['devices_opened']==0 and receipt['selected_trained_ema']
        assert len(receipt['checks'])==3 and all(c['native_exact'] and c['probability_max_error']<=1e-6 and c['captures']==5 for c in receipt['checks'])
        assert receipt['source_sha256']=={n:r.sha(HERE/n) for n in ADDITIONAL}
        r.engine.verify_sources(sources(),receipt['tool_git_commit'])
        result['native_receipt_sha256']=r.sha(study/'selected_native_bmp/checks.json')
    return result

def bind(alpha,ref):
    r.bind(alpha);old_read=r.engine.read
    def read(path):
        value=old_read(path)
        if Path(path)==r.PLAN:value=dict(value,scope=POLICY['scope'],depth_validation_policy=POLICY,parent_L4_validation=ref)
        return value
    r.engine.read=read;r.engine.sources=sources

def eligibility(fit,history):
    tm,vm=fit['metrics']['train'],fit['metrics']['val'];tail=history[-min(20,len(history)):]
    checks=dict(val_accuracy=vm['accuracy']>=POLICY['validation_accuracy_min'],val_macro=vm['balanced_accuracy']>=POLICY['validation_macro_min'],selected_no_zero=min(tm['recall'])>0 and min(vm['recall'])>0,each_val_recall=min(vm['recall'])>=POLICY['each_val_recall_min'],no_persistent_zero=all(sum(h['val']['recall'][k]==0 for h in tail)<=len(tail)/2 for k in range(20)),gap=100*(tm['accuracy']-vm['accuracy'])<=POLICY['maximum_gap_pp'],routing_all_slots=all(v==1 for split in (tm,vm) for v in split['routing']['nonzero_fraction']),routing_input_dependence=all(max(split['routing']['weight_std'])>1e-4 for split in (tm,vm)))
    return dict(passed=all(checks.values()),checks=checks,selected_epoch=fit['selected_epoch'],gap_pp=100*(tm['accuracy']-vm['accuracy']),test_evaluations=0,policy=POLICY)

def fit(a):
    assert a.depth in (2,3);ref=reference(a.study);bind(ref['mixup_alpha'],ref);a.rho=.3;a.resume_parent=None;a.data=Path(r.read(r.PLAN)['data']);r.engine.fit(a)
    f=r.read(a.out/'fit_result.json');r.save(a.out/'validation_gate.json',eligibility(f,r.read(a.out/'history.json')),exclusive=True)

def queue(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';ref=reference(a.study);src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=r.ROOT,text=True).strip();r.engine.verify_sources(src,commit);a.out.mkdir(parents=True,exist_ok=False)
    r.save(a.out/'metadata.json',dict(pid=os.getpid(),sources=src,git_commit=commit,policy=POLICY,parent_L4_validation=ref,command=sys.argv,test_evaluations=0),exclusive=True);pending=[2,3];active=[];completed=[];failures=[]
    while pending or active:
        for j in list(active):
            rc=j['process'].poll()
            if rc is not None:
                j['log'].close();record={k:v for k,v in j.items() if k not in ('process','log')};record['exit_code']=rc;(completed if rc==0 else failures).append(record);active.remove(j)
        if failures:pending=[]
        busy={j['gpu_uuid'] for j in active}
        for index in (6,2,5):
            gpu=r.ALLOWED[index]
            if failures or not pending or gpu in busy:continue
            r.gpu_budget.ALLOWED.update(r.ALLOWED.values())
            try:r.gpu_budget.preflight(gpu,shared=True,memory_gib=2)
            except AssertionError:continue
            depth=pending.pop(0);key=f'L{depth}_rho03';out=a.out/key;assert not out.exists();handle=(a.out/(key+'.log')).open('x');cmd=[sys.executable,str(Path(__file__).resolve()),'--phase','fit','--study',str(a.study),'--depth',str(depth),'--gpu',gpu,'--out',str(out)];p=subprocess.Popen(cmd,cwd=r.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=handle,stderr=subprocess.STDOUT);active.append(dict(case=key,depth=depth,pid=p.pid,gpu_uuid=gpu,process=p,log=handle));busy.add(gpu)
        r.save(a.out/'current_processes.json',dict(active=[{k:v for k,v in j.items() if k not in ('process','log')} for j in active],pending=pending))
        r.save(a.out/'status.json',dict(state='validation_failed' if failures else 'validation_running' if active or pending else 'validation_complete_awaiting_CPU_review',active=len(active),pending=len(pending),completed=completed,failures=failures,test_evaluations=0))
        if active or pending:time.sleep(5)
    r.engine.verify_sources(src,commit)

def audit(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not a.out.exists();ref=reference(a.study);meta=r.read(a.base/'metadata.json');state=r.read(a.base/'status.json');assert audit_core.exited(meta['pid']) and state['active']==state['pending']==0 and not state['failures'] and len(state['completed'])==2 and meta['parent_L4_validation']==ref
    r.engine.verify_sources(meta['sources'],meta['git_commit']);assert meta['sources']==sources();m=r.manifest(Path(r.read(r.PLAN)['data']));fits={};gates={}
    for j in state['completed']:
        key=j['case'];f=audit_core.verify_arm(a.base,key,Path(r.read(r.PLAN)['data']),m);assert f['depth']==j['depth'] and f['rho']==.3 and f['config']['mixup_alpha']==ref['mixup_alpha'] and f['sources']==meta['sources'];assert f['config']['parent_L4_validation']==ref and f['config']['depth_validation_policy']==POLICY
        gate=eligibility(f,r.read(a.base/key/'history.json'));assert gate==r.read(a.base/key/'validation_gate.json');fits[key]=f;gates[key]=gate
    assert set(fits)=={'L2_rho03','L3_rho03'};a.out.mkdir(parents=True,exist_ok=False);r.save(a.out/'audit.json',dict(passed=True,all_depths_eligible=all(g['passed'] for g in gates.values()),fits=fits,gates=gates,parent_L4_validation=ref,model_forward_calls=0,test_arrays_read=0,test_evaluations=0,formal_training_started=False),exclusive=True)

def native(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';ref=reference(a.study,require_native=False);tool_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=r.ROOT,text=True).strip();r.engine.verify_sources(sources(),tool_commit);out=a.out;assert out==a.study/'selected_native_bmp';out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    f=r.read(a.study/ref['case']/'fit_result.json');cp=a.study/ref['case']/'best_checkpoint.pt';state=torch.load(cp,map_location='cpu',weights_only=False);assert state['config']==f['config'] and state['epoch']==ref['selected_epoch'];model=r.CorrectedMoE(4,.3).eval();model.load_state_dict(state['model'],strict=True);export.export_selected(model,f['config'],out/'static',cp,ref['checkpoint_sha256'],f['config']['classes'],f['sources'])
    torch.manual_seed(20261014);probes={'random':torch.randint(0,256,(1,3,150,150),dtype=torch.uint8),'dim':torch.full((1,3,150,150),4,dtype=torch.uint8),'black':torch.zeros((1,3,150,150),dtype=torch.uint8)};checks=[]
    for probe,image in probes.items():
        amp=r.input_amplitude(image)
        with torch.no_grad():expected=model(amp)[0]
        current=[amp];weights=[None];calls=[];stats=[]
        def replay(stage,a,p,ids):
            qa,qp=export.encode_stage(model,current[0],stage,weights[0]);assert np.array_equal(qa,np.rint(a*255).astype(np.uint8)) and np.array_equal(qp,p);dest=out/(probe+'_'+stage);receipt=export.save_sample_bmps(model,current[0],stage,dest,weights[0]);gray=reverse(dest/'amplitude_000000_native.bmp','amplitude');phase=reverse(dest/'phase_native.bmp','phase');assert np.array_equal(gray,qa[0]) and np.array_equal(phase,qp);calls.append(stage);stats.append(receipt['brightness'][0])
            field=torch.from_numpy(gray.astype(np.float32)/255)[None]*torch.exp(2j*math.pi*torch.from_numpy(phase.astype(np.float32)/256))[None];field=torch.nn.functional.pad(field,(20,20,20,20));intensity=model.propagator(field).abs().square()[:,20:498,20:498]
            if stage=='router':weights[0]=export.routing_from_ccd(model,intensity)
            elif int(stage)<3:current[0]=export.amplitude_from_ccd(model,intensity,model.stages[int(stage)+1])
            return intensity.numpy()
        with torch.no_grad():actual=model(amp,callback=replay,ids=[probe])[0]
        error=float((actual-expected).abs().max());assert error<=1e-6 and calls==['router','0','1','2','3'] and torch.equal(weights[0],model.routing_weights)
        if probe=='black':assert all(v['active']['maximum']==0 for v in stats) and torch.allclose(actual,torch.full_like(actual,.05),atol=1e-8,rtol=0)
        checks.append(dict(probe=probe,passed=True,native_exact=True,probability_max_error=error,captures=5,brightness=stats))
    assert r.sha(cp)==ref['checkpoint_sha256'];r.engine.verify_sources(f['sources'],r.read(a.study/'metadata.json')['git_commit']);r.save(out/'checks.json',dict(passed=True,checks=checks,selected_trained_ema=True,tool_git_commit=tool_commit,checkpoint_sha256=ref['checkpoint_sha256'],parent_L4_validation=ref,source_sha256={n:r.sha(HERE/n) for n in ADDITIONAL},model_forward_calls=6,test_arrays_read=0,test_evaluations=0,devices_opened=0,gpu_used=False,physical_acceptance=False),exclusive=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','queue','audit','native'],required=True);p.add_argument('--study',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--base',type=Path);p.add_argument('--depth',type=int);p.add_argument('--gpu');a=p.parse_args();assert not a.out.exists()
    try:globals()[a.phase](a)
    except BaseException as e:
        if a.out.exists():r.save(a.out/'failure.json',dict(error=repr(e),automatic_retry=False,test_evaluations=0),exclusive=True)
        raise
