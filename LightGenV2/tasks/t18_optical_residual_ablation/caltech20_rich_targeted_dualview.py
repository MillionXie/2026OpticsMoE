"""Same hybrid inference: evidence-driven RMS gradients and clean-view KD."""
import ast,inspect,types,importlib
import caltech20_rich_alpha40_rebalance as core
for n,v in vars(core).items():
    if not n.startswith('__'):globals()[n]=v
import caltech20_rich_branch_investigation as diagnosis
import caltech20_rich_fivegpu_resources as resources
import caltech20_rich_projected_alpha_contained as physical
from caltech20_rich_projected_alpha_model import convert as remap
from caltech20_rich_alpha40_model import rms,features,energies
FILES=('caltech20_rich_targeted_dualview.py','caltech20_rich_targeted_dualview_profile.json','check_caltech20_rich_targeted_dualview.py','audit_caltech20_rich_targeted_dualview.py','CALTECH20_RICH_TARGETED_DUALVIEW_PROTOCOL_20261018.md')
ACTIVE_FULL_RMS=True;CLEAN_X=None
def config():
    c=core.config();c.update(s.read(HERE/FILES[1]));return c
OUT=HERE/'runs/simulation'/config()['run'];CPU=HERE/'runs/smoke/caltech20_rich_targeted_dualview_cpu_20261018'
ALPHA_MIN=physical.ALPHA_MIN
def sources():
    z=diagnosis.sources()
    for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return z
def select(key):
    global ACTIVE_FULL_RMS
    ACTIVE_FULL_RMS=bool(config()['cases'][key]['full_rms_gradient'])
class Alpha40MoE(physical.ContainedMoE):
    def __init__(self,depth,rho=.3):
        super().__init__(depth,rho);self.full_rms_gradient=ACTIVE_FULL_RMS
    def context(self,seed,count,cfg):
        assert cfg['phase_dropout']==cfg['ccd_gain_noise']==cfg['ccd_read_noise']==0
        self.robust=None;self.offset=0
    def forward(self,amplitude,callback=None,ids=None):
        self.last={};self.trace=[];intensity=self.capture(amplitude,'router',callback=callback,ids=ids);self.trace.append('router')
        energy=energies(intensity,self.router_masks);centered=energy-energy.mean(1,keepdim=True);dense=torch.softmax(centered/(centered.square().mean(1,keepdim=True)+1e-8).sqrt()/2,1)
        weights=dense/dense.square().sum(1,keepdim=True).sqrt().clamp_min(1e-8);self.routing_weights=weights.detach()
        for k in range(self.depth):
            intensity=self.capture(amplitude,str(k),weights,callback,ids);self.trace.append(self.stages[k]);target=self.stages[k+1] if k<self.depth-1 else 'global';size=478 if target=='global' else 224
            optical=self.nonlinear(intensity,target);electronic=self.electronic[k](amplitude,size);re,ro=rms(electronic),rms(optical)
            if not self.training or not self.full_rms_gradient:re,ro=re.detach(),ro.detach()
            a=self.fusion_logits[k].float();mixed=(1-a)*(electronic/re)+a*(optical/ro);rm=rms(mixed)
            if not self.training or not self.full_rms_gradient:rm=rm.detach()
            amplitude=re*mixed/rm
        f=features(amplitude);p=torch.softmax(self.linear(f),1);energy=energies(intensity,self.detectors);self.readout=dict(features=f.detach(),amplitude=amplitude.detach(),ccd=intensity.detach())
        return p,energy.sum(1)/intensity.sum((-2,-1)).clamp_min(1e-12)
    def contract(self):
        c=super().contract();c.update(id='singleRouter_EG_directAlpha04_same5140_dualview_RMS_gradient_20261018',training_full_RMS_derivative=self.full_rms_gradient,clean_inference_function_unchanged=True,training_only_two_view=True,legacy_fusion_keys_store_physical_alpha=True);return c
def identity():
    z=sources();git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();engine.verify_sources(z,git);return dict(sources=z,git_commit=git)
def claim(path,phase):s.save(path,dict(identity(),pid=os.getpid(),phase=phase,automatic_retry=False),exclusive=True)
for name in ('rates','seed_for','load_predictions','groups'):
    f=getattr(core,name);globals()[name]=types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__)
def incumbent(depth):return core.incumbent(depth)
def initialize(depth,fresh):
    assert not fresh,'Preserve retained accuracy; no fresh initialization or alpha reset';engine.seed(17);q,pt=diagnosis.parents.parent(depth);model=Alpha40MoE(depth);model.load_state_dict(remap(pt['model'],q['alpha_low']));model.anchor();return model
def mixed(data,idx,theta,epoch,batch,cfg,device):
    global CLEAN_X
    x,y,mask,crop=core.mixed(data,idx,theta,epoch,batch,cfg,device)
    clean=base.encode(data[0][idx].to(device),None,100);rg=np.random.default_rng(np.random.SeedSequence([17,epoch,batch,20261013,4]));lam=float(rg.beta(cfg['mixup_alpha'],cfg['mixup_alpha']));lam=max(lam,1-lam);perm=torch.tensor(rg.permutation(len(idx)),device=device)
    CLEAN_X=lam*clean+(1-lam)*clean[perm]
    return x,y,mask,crop
def js_loss(p,q):
    middle=(p+q)/2
    return .5*((p*(p.clamp_min(1e-12).log()-middle.clamp_min(1e-12).log())).sum(1)+(q*(q.clamp_min(1e-12).log()-middle.clamp_min(1e-12).log())).sum(1)).mean()
def gradients(model,x,y,q,mass,mask,weights,cfg,micro):
    assert CLEAN_X is not None and CLEAN_X.shape==x.shape;model.train();model.zero_grad(set_to_none=True);ps=[];cs=[];parts=dict(clean=0.,augmented=0.,KD_clean=0.,JS=0.,phase_anchor=0.);total=0.
    for off in range(0,len(x),micro):
        size=min(micro,len(x)-off);factor=size/len(x);model.offset=off;model.linear.mask=None;pc,cc=model(CLEAN_X[off:off+micro]);model.linear.mask=mask[off:off+micro];pa,ca=model(x[off:off+micro])
        clean=base.loss(model,pc,cc,y[off:off+micro],weights,cfg);aug=base.loss(model,pa,ca,y[off:off+micro],weights,cfg);kd=old.kd.KD_loss(pc,q[off:off+micro],mass[off:off+micro],cfg);js=js_loss(pc,pa)
        anchor=sum((p-model.phase_anchors[n]).square().sum() for n,p in model.named_parameters() if n in model.phase_anchors)/2
        loss=.5*clean+.5*aug+cfg['KD_weight']*kd+cfg['consistency_weight']*js+cfg['phase_anchor_weight']*anchor
        (loss*factor).backward();total+=float(loss.detach())*factor;ps.append(pa.detach());cs.append(ca.detach())
        for name,value in zip(parts,(clean,aug,kd,js,anchor)):parts[name]+=float(value.detach())*factor
    model.linear.mask=None;model.view_info=dict(parts,weighted_sum=total,teacher_view='unaugmented_MixUp_same_ids_permutation_mass',full_rms_gradient=model.full_rms_gradient,clean_macro_sha256=s.digest(CLEAN_X),new_inference_parameters=0)
    return torch.cat(ps),torch.cat(cs),total
def ready():
    p=s.read(CPU/'checks.json');assert p['passed'] and p['sources']==sources() and p['five_gpu_guard']['passed'] and p['all_actual_kernel_namespaces_resolved'];identity();old.controls()
    for d in (2,4,6):
        q=s.read(diagnosis.OUT/f'L{d}/diagnosis.json');ex=s.read(diagnosis.OUT/f'L{d}.producer_exit.json');assert q['passed'] and q['sources']==diagnosis.sources() and ex['returncode']==0 and ex['actual_exit'] and q['TEST_images_read']==q['optimizer_steps']==0
gate_text=inspect.getsource(core.gate).replace('def gate(', 'def numeric_gate_kernel(',1)
gate_site='gradients(model,x,y,q,mass,mask,weights,cfg,4)';assert gate_text.count(gate_site)==1
gate_text=gate_text.replace(gate_site,"gradients(model,x,y,q,mass,mask,weights,cfg,cfg['microbatch_size'])")
exec(compile(gate_text,__file__+':same_numeric_gate_micro1vs2','exec'),globals())
def gate(depth,fresh,data,case):
    model=initialize(depth,fresh).cuda().eval();q,pt=diagnosis.parents.parent(depth);mod=importlib.import_module(q['module']);reference=getattr(mod,q['class_name'])(depth,.3).cuda().eval();reference.load_state_dict(pt['model'])
    with torch.no_grad():
        x=base.encode(data[0][:4].cuda(),None,100);p,c=model(x);a,b=reference(x);pe=float((p-a).abs().max());ce=float((c-b).abs().max());assert pe<=3e-6 and ce<=3e-6
    del model,reference;torch.cuda.empty_cache();proof=numeric_gate_kernel(depth,fresh,data,case);proof.update(parent_clean_probability_maxabs=pe,parent_clean_capture_maxabs=ce,full_rms_gradient=ACTIVE_FULL_RMS,teacher_applied_to_clean_view_only=True,micro_comparison=[1,config()['microbatch_size']])
    return proof
text=inspect.getsource(core.fit);assert text.count('opt.step();steps+=1')==1;text=text.replace('opt.step();steps+=1','opt.step();model.project();steps+=1')
assert text.count(gate_site)==1;text=text.replace(gate_site,"gradients(model,x,y,q,mass,mask,weights,cfg,cfg['microbatch_size'])")
site='crop_sha256=crop,loss=loss,clip_norm=norm';assert text.count(site)==1;text=text.replace(site,'crop_sha256=crop,loss=loss,clip_norm=norm,view_info=model.view_info')
exec(compile(text,__file__+':reviewed_fit_with_view_and_projected_alpha','exec'),globals())
def campaign():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not OUT.exists();ready();assert resources.slots()['total']==0;claim(OUT.with_name(OUT.name+'.claim.json'),'five_targeted_residual_method_cases');OUT.mkdir();s.save(OUT/'metadata.json',dict(identity(),pid=os.getpid(),config=config(),frozen_controls=old.controls()),exclusive=True);jobs=[];complete=[];failures=[]
    for key,case in config()['cases'].items():
        gpu=resources.GPU[case['gpu']];s.gpu_budget.ALLOWED.update(resources.GPU.values());s.gpu_budget.preflight(gpu,shared=True,memory_gib=2);log=(OUT/(key+'.log')).open('x');p=subprocess.Popen([os.sys.executable,str(HERE/FILES[0]),'--phase','fit','--key',key,'--gpu',gpu],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT);jobs.append(dict(key=key,gpu=gpu,pid=p.pid,process=p,log=log))
    while jobs:
        for job in list(jobs):
            rc=job['process'].poll()
            if rc is None:continue
            job['log'].close();ex={k:v for k,v in job.items() if k not in ('process','log')};ex.update(returncode=rc,actual_exit=True);s.save(OUT/(job['key']+'.producer_exit.json'),ex,exclusive=True);jobs.remove(job)
            if rc:failures.append(ex);continue
            out=OUT/job['key'];claim(out/'audit_claim.json','independent_dualview_saved_science');
            with (out/'audit.log').open('x') as log:arc=subprocess.run([os.sys.executable,str(HERE/FILES[3]),'--key',job['key']],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT).returncode
            if arc:failures.append(dict(key=job['key'],audit_returncode=arc));continue
            assert s.read(out/'science/audit.json')['passed'];s.save(out/'audit_completed.json',dict(passed=True,actual_audit_returncode=arc,audit_sha256=s.sha(out/'science/audit.json')),exclusive=True);complete.append(ex)
        s.save(OUT/'status.json',dict(state='targeted_dualview_running' if jobs else 'targeted_dualview_review_required' if not failures else 'targeted_dualview_failed',active=[{k:v for k,v in q.items() if k not in ('process','log')} for q in jobs],pending=[],complete=complete,failures=failures))
        if jobs:time.sleep(2)
    assert not failures,'Preserve failure, no automatic retry'
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','campaign'],required=True);p.add_argument('--key');p.add_argument('--gpu');a=p.parse_args()
    if a.key:select(a.key)
    try:campaign() if a.phase=='campaign' else fit(a.key,a.gpu)
    except BaseException as e:
        out=OUT/a.key if a.key else OUT;mark=out/'failure.json' if out.exists() else out.with_name(out.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
