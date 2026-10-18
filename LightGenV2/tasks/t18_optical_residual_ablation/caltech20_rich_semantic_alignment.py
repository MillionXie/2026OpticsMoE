"""Fixed TRAIN teacher feature targets, same optical/electronic inference graph."""
import ast,inspect,types
import caltech20_rich_dualview_hosthash_repair as original
for n,v in vars(original).items():
    if not n.startswith('__') and n!='original':globals()[n]=v
import caltech20_rich_teacher_alignment as alignment
from functools import lru_cache
FILES=('caltech20_rich_semantic_alignment.py','caltech20_rich_semantic_alignment_profile.json','check_caltech20_rich_semantic_alignment.py','audit_caltech20_rich_semantic_alignment.py','CALTECH20_RICH_SEMANTIC_ALIGNMENT_PROTOCOL_20261018.md')
ACTIVE_DEPTH=4;FEATURE_TARGET=None
def config():
    c=original.config();c.update(s.read(HERE/FILES[1]));return c
OUT=HERE/'runs/simulation'/config()['run'];CPU=HERE/'runs/smoke/caltech20_rich_semantic_alignment_cpu_20261018'
def sources():
    z=alignment.sources()
    for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return z
def select(key):
    global ACTIVE_FULL_RMS,ACTIVE_DEPTH
    ACTIVE_DEPTH=config()['cases'][key]['depth'];ACTIVE_FULL_RMS=True
class Alpha40MoE(original.Alpha40MoE):
    def __init__(self,depth,rho=.3):
        super().__init__(depth,rho);self.full_rms_gradient=True
        self.semantic_hook=self.linear.register_forward_pre_hook(self.keep_feature)
    def keep_feature(self,module,args):self.semantic_feature=args[0]
    def contract(self):
        c=super().contract();c.update(TRAIN_only_semantic_supervision=True,new_inference_parameters=0);return c
@lru_cache(None)
def target_data(depth):
    rec=s.read(alignment.OUT/'audit_completed.json');assert rec['actual_audit_returncode']==0 and rec['audit_sha256']==config()['alignment_science_sha256']==s.sha(alignment.OUT/'science/audit.json');result=s.read(alignment.OUT/'result.json');assert result['sources']==alignment.sources() and result['TRAIN3141_only']
    row=result['checks'][str(depth)];path=alignment.OUT/f'L{depth}_train_targets.npz';assert s.sha(path)==row['targets_sha256'] and row['teacher_target_parent_head_TRAIN_accuracy']>=.85
    with np.load(path,allow_pickle=False) as z:features,ids,labels=[z[k].copy() for k in ('features','ids','labels')]
    assert ids.tolist()==base.manifest()['split_ids']['train'] and np.bincount(labels,minlength=20).tolist()==base.manifest()['supports']['train']
    return features,row['alignment_scale'],row['targets_sha256']
for name in ('identity','claim','rates','seed_for','load_predictions','groups','incumbent','initialize','js_loss','numeric_gate_kernel','fit'):
    f=getattr(original,name);globals()[name]=types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__)
def gate(depth,fresh,data,case):
    a=torch.arange(8,device='cuda',dtype=torch.float32);assert s.digest(a.detach().cpu())==s.digest(torch.arange(8,dtype=torch.float32))
    f=original.original.gate;check=types.FunctionType(f.__code__,globals(),'retained_numeric_gate',f.__defaults__,f.__closure__);q=check(depth,fresh,data,case);q.update(actual_GPU_host_hash_roundtrip_passed=True,TRAIN_feature_targets_only=True);return q
def mixed(data,idx,theta,epoch,batch,cfg,device):
    global FEATURE_TARGET
    f=types.FunctionType(original.mixed.__code__,globals(),'clean_augmented_views',original.mixed.__defaults__,original.mixed.__closure__);result=f(data,idx,theta,epoch,batch,cfg,device);features,scale,sha=target_data(ACTIVE_DEPTH)
    rg=np.random.default_rng(np.random.SeedSequence([17,epoch,batch,20261013,4]));lam=float(rg.beta(cfg['mixup_alpha'],cfg['mixup_alpha']));lam=max(lam,1-lam);perm=rg.permutation(len(idx));target=lam*features[idx]+(1-lam)*features[idx][perm]
    FEATURE_TARGET=torch.nn.functional.layer_norm(torch.tensor(target,device=device),(256,),eps=1e-5);return result
def feature_loss(student,target,mass):return .5*(mass*(student-target).square().sum(1)).mean()
text=inspect.getsource(original.original.gradients);hash_site='s.digest(CLEAN_X)';assert text.count(hash_site)==1;text=text.replace(hash_site,'s.digest(CLEAN_X.detach().cpu())')
site="pc,cc=model(CLEAN_X[off:off+micro]);model.linear.mask";assert text.count(site)==1;text=text.replace(site,"pc,cc=model(CLEAN_X[off:off+micro]);clean_features=model.semantic_feature;model.linear.mask")
site="loss=.5*clean+.5*aug+cfg['KD_weight']*kd+cfg['consistency_weight']*js+cfg['phase_anchor_weight']*anchor";assert text.count(site)==1;text=text.replace(site,"alignment_value=feature_loss(clean_features,FEATURE_TARGET[off:off+micro],mass[off:off+micro]);scale=target_data(model.depth)[1];"+site+"+scale*alignment_value")
site="parts=dict(clean=0.,augmented=0.,KD_clean=0.,JS=0.,phase_anchor=0.)";assert text.count(site)==1;text=text.replace(site,"parts=dict(clean=0.,augmented=0.,KD_clean=0.,JS=0.,phase_anchor=0.,feature_alignment=0.)")
site="zip(parts,(clean,aug,kd,js,anchor))";assert text.count(site)==1;text=text.replace(site,"zip(parts,(clean,aug,kd,js,anchor,alignment_value))")
site="new_inference_parameters=0)";assert text.count(site)==1;text=text.replace(site,"new_inference_parameters=0,alignment_scale=scale,alignment_target_sha256=target_data(model.depth)[2])")
exec(compile(text,__file__+':same_dualview_plus_fixed_TRAIN_semantic_loss','exec'),globals())
def ready():
    p=s.read(CPU/'checks.json');assert p['passed'] and p['sources']==sources() and p['all_actual_kernel_namespaces_resolved'];identity();old.controls()
    for d in (4,6):target_data(d)
def campaign():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not OUT.exists();ready();claim(OUT.with_name(OUT.name+'.claim.json'),'two_semantic_candidates_after_TRAIN_geometry_proof');OUT.mkdir();s.save(OUT/'metadata.json',dict(identity(),pid=os.getpid(),config=config(),frozen_controls=old.controls()),exclusive=True);pending=list(config()['cases']);jobs=[];complete=[];failures=[];start=time.monotonic()
    while pending or jobs:
        assert time.monotonic()-start<14400,'Queue/fit fault protection, no extra candidates';slots=resources.slots();used={q['gpu'] for q in jobs}
        for index in config()['gpu_order']:
            gpu=resources.GPU[index]
            if not pending or gpu in used or slots['counts'][gpu]>0:continue
            key=pending.pop(0);s.gpu_budget.ALLOWED.update(resources.GPU.values());s.gpu_budget.preflight(gpu,shared=True,memory_gib=2);log=(OUT/(key+'.log')).open('x');p=subprocess.Popen([os.sys.executable,str(HERE/FILES[0]),'--phase','fit','--key',key,'--gpu',gpu],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT);jobs.append(dict(key=key,gpu=gpu,pid=p.pid,process=p,log=log));used.add(gpu)
        for job in list(jobs):
            rc=job['process'].poll()
            if rc is None:continue
            job['log'].close();ex={k:v for k,v in job.items() if k not in ('process','log')};ex.update(returncode=rc,actual_exit=True);s.save(OUT/(job['key']+'.producer_exit.json'),ex,exclusive=True);jobs.remove(job)
            if rc:failures.append(ex);pending=[];continue
            out=OUT/job['key'];claim(out/'audit_claim.json','semantic_alignment_independent_saved_science')
            with (out/'audit.log').open('x') as log:arc=subprocess.run([os.sys.executable,str(HERE/FILES[3]),'--key',job['key']],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT).returncode
            if arc:failures.append(dict(key=job['key'],audit_returncode=arc));pending=[];continue
            assert s.read(out/'science/audit.json')['passed'];s.save(out/'audit_completed.json',dict(passed=True,actual_audit_returncode=arc,audit_sha256=s.sha(out/'science/audit.json')),exclusive=True);complete.append(ex)
        s.save(OUT/'status.json',dict(state='semantic_candidates_failed' if failures else 'semantic_candidates_training_or_waiting' if jobs or pending else 'semantic_candidates_review_required',active=[{k:v for k,v in q.items() if k not in ('process','log')} for q in jobs],pending=pending,complete=complete,failures=failures))
        if pending or jobs:time.sleep(2)
    assert not failures
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','campaign'],required=True);p.add_argument('--key');p.add_argument('--gpu');a=p.parse_args()
    if a.key:select(a.key)
    try:campaign() if a.phase=='campaign' else fit(a.key,a.gpu)
    except BaseException as e:
        out=OUT/a.key if a.key else OUT;mark=out/'failure.json' if out.exists() else out.with_name(out.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
