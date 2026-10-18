"""Retained optical phases, gradual alpha cap and all existing electronics adapt."""
import argparse,copy,importlib,json,math,os,subprocess,time,types
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
import caltech20_rich_L2_objective_rebalance as closure
import caltech20_rich_postCCD_electronic_adapt as kernels
from audit_caltech20_rich_pair50_alpha import metrics,verify_predictions,agree
HERE,ROOT,s,base,engine,resources=closure.HERE,closure.ROOT,closure.s,closure.base,closure.engine,closure.resources
physics=kernels.physics
FILES=('caltech20_rich_electronic_curriculum.py','caltech20_rich_electronic_curriculum_profile.json','check_caltech20_rich_electronic_curriculum.py','audit_caltech20_rich_electronic_curriculum.py','CALTECH20_RICH_ELECTRONIC_CURRICULUM_PROTOCOL_20261018.md')
def config():
    c=dict(base.config());c.update(s.read(HERE/FILES[1]));return c
OUT=HERE/'runs/simulation'/config()['run'];CPU=HERE/'runs/smoke/caltech20_rich_electronic_curriculum_CPU_20261018'
def sources():
    z=closure.sources()
    for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return z
def identity():
    z=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();engine.verify_sources(z,commit);return dict(sources=z,git_commit=commit)
def claim(path,phase):s.save(path,dict(identity(),pid=os.getpid(),phase=phase,automatic_retry=False),exclusive=True)
@lru_cache(None)
def parent(depth):
    q,pt,pred=closure.incumbent(depth);assert s.sha(Path(q['path']))==q['sha256'];assert all(pred[r]['ids'].tolist()==base.manifest()['split_ids'][r] for r in ('train','val','test'));return q,pt,pred
def initial_state(depth):
    q,pt,_=parent(depth);return closure.remap(pt['model'],q['alpha_low'])
def names(depth):return [f'electronic.{i}.local.weight' for i in range(depth)]+['linear.weight','linear.bias']+[f'fusion_logits.{i}' for i in range(depth)]
def frozen_digest(state,depth):return s.digest({n:v for n,v in state.items() if n=='raw_router' or n.startswith('phase_sets.')})
def cap(key,epoch):return .95+(config()['cases'][key]['end_alpha_cap']-.95)*(epoch-1)/(config()['electronic_epochs']-1)
class CurriculumMoE(physics.Alpha40MoE):
    def __init__(self,depth,rho=.3):
        super().__init__(depth,rho);self.full_rms_gradient=True;self.alpha_cap=.95;self.noisy=False
        for n,p in self.named_parameters():p.requires_grad_(n in names(depth))
    def context(self,*unused):self.robust=None;self.offset=0;self.noisy=False
    def project(self):
        with torch.no_grad():
            for p in self.fusion_logits:p.clamp_(config()['alpha_floor'],self.alpha_cap)
    def contract(self):
        c=super().contract();c.update(frozen_optical_phases=True,all_existing_electronics_adapt=True,training_alpha_upper_projection=True,new_inference_parameters=0,existing_trainable_parameters=5140+10*self.depth,hardware_capture_reuse_not_allowed_after_upstream_electronics_change=True);return c
def initialize(depth):
    engine.seed(17);model=CurriculumMoE(depth);model.load_state_dict(initial_state(depth));return model
configure=kernels.configure
weights=kernels.weights
CE,KL,JS=kernels.CE,kernels.KL,kernels.JS
augmentation=kernels.augmentation
mixed=types.FunctionType(kernels.mixed.__code__,globals(),'mixed',kernels.mixed.__defaults__)
gradients=types.FunctionType(kernels.gradients.__code__,globals(),'gradients',kernels.gradients.__defaults__)
load_np=kernels.load_np
@lru_cache(None)
def teacher(depth):
    p=parent(depth)[2]['train']['probabilities'].astype(float);z=np.log(np.maximum(p,1e-12))/config()['self_temperature'];v=np.exp(z-z.max(1,keepdims=True));return (v/v.sum(1,keepdims=True)).astype(np.float32)
def rates(epoch):
    c=config();f=(1+math.cos(math.pi*(epoch-1)/c['electronic_epochs']))/2;return [c['minimum_lr']+(c['conv_lr']-c['minimum_lr'])*f,c['minimum_lr']+(c['head_lr']-c['minimum_lr'])*f,c['minimum_alpha_lr']+(c['alpha_lr']-c['minimum_alpha_lr'])*f]
def ready():
    p=s.read(CPU/'checks.json');assert p['passed'] and p['sources']==sources() and p['all_existing_electronics_exact_parent_proved'];identity();closure.old.controls()
def gate(depth,data):
    cfg=config();model=initialize(depth).cuda().eval();q,pt,_=parent(depth);mod=importlib.import_module(q['module']);reference=getattr(mod,q['class_name'])(depth,.3).cuda().eval();reference.load_state_dict(pt['model']);indices=np.array([int((data[1]==k).nonzero()[0]) for k in range(20)]);pe=ce=0.
    with torch.no_grad():
        for off in range(0,20,2):
            x=base.encode(data[0][indices[off:off+2]].cuda(),None,100);a,b=model(x);p,c=reference(x);pe=max(pe,float((a-p).abs().max()));ce=max(ce,float((b-c).abs().max()))
    assert pe<=3e-6 and ce<=3e-6;del reference;torch.cuda.empty_cache();before=s.digest(model.state_dict());order,theta=engine.epoch_plan(1,3141,cfg);w=weights('cuda');records=[]
    for label,idx,batch in [('all20',indices,0),('macro16',order[:16],0),('tail5',order[-5:],196)]:
        clean,x,y,q,mask,crop=mixed(depth,data,idx,theta,1,batch,cfg,'cuda');model.context();a,_,_=gradients(model,clean,x,y,q,mask,w,cfg,1);first={n:p.grad.detach().clone() for n,p in model.named_parameters() if p.requires_grad};b,_,_=gradients(model,clean,x,y,q,mask,w,cfg,cfg['microbatch_size']);rel=[];cos=[];norms={}
        for n,p in model.named_parameters():
            if not p.requires_grad:assert p.grad is None;continue
            u,v=first[n].double(),p.grad.double();norms[n]=float(v.norm());assert torch.isfinite(v).all() and norms[n]>0;rel.append(float((u-v).norm()/v.norm()));cos.append(float((u*v).sum()/(u.norm()*v.norm())))
        pmax=max(float((a[n]-b[n]).abs().max()) for n in ('pc','pa'));cmax=max(float((a[n]-b[n]).abs().max()) for n in ('cc','ca'));assert pmax<=3e-6 and cmax<=3e-6 and max(rel)<=.02 and min(cos)>=.999;records.append(dict(samples=label,probability_maxabs=pmax,capture_maxabs=cmax,gradient_max_relative=max(rel),gradient_min_cosine=min(cos),gradient_norms=norms))
    assert before==s.digest(model.state_dict());del model;torch.cuda.empty_cache();return dict(passed=True,checks=records,parent_clean_probability_maxabs=pe,parent_capture_maxabs=ce,trainable_parameters=5140+10*depth,trainable_tensors=2*depth+2,frozen_gradients_all_None=True,optimizer_steps=0,TEST_forward_calls=0)
def checkpoint(model,epoch,tm,vm):return dict(model={n:v.detach().cpu().clone() for n,v in model.state_dict().items()},local_electronic_epoch=epoch,metrics=dict(train=tm,val=vm),sources=sources(),config=config(),contract=model.contract(),model_digest=s.digest(model.state_dict()))
def fit(key,gpu):
    ready();cfg=config();depth=cfg['cases'][key]['depth'];folder=OUT/key;assert not folder.exists();claim(folder.with_name(folder.name+'.fit_claim.json'),'retained_all_electronic_curriculum');folder.mkdir();policy=configure(gpu);torch.set_num_threads(4);data,val=base.load('train'),base.load('val');proof=gate(depth,data);s.save(folder/'numeric_gate.json',proof,exclusive=True);model=initialize(depth).cuda();ema=copy.deepcopy(model).eval();named=dict(model.named_parameters());ns=names(depth);groups=[ns[:depth],ns[depth:depth+2],ns[depth+2:]];opt=torch.optim.AdamW([dict(params=[named[n] for n in group],lr=lr,weight_decay=wd) for group,lr,wd in zip(groups,rates(1),[cfg['conv_weight_decay'],cfg['head_weight_decay'],0])]);q,_,_=parent(depth);initial=s.digest(model.state_dict());frozen=frozen_digest(model.state_dict(),depth);s.save(folder/'metadata.json',dict(identity(),key=key,depth=depth,parent=q,config=cfg,gpu_uuid=gpu,policy=policy,trainable_names=ns,trainable_parameters=5140+10*depth,optimizer_groups=groups,optimizer_reset=True,initial_digest=initial,initial_frozen_digest=frozen,local_epoch_semantics=True,phase_updates=0,new_joint_epochs=0,command=os.sys.argv,environment=dict(python=os.sys.version,torch=torch.__version__,numpy=np.__version__,CUDA=torch.version.cuda,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'))),exclusive=True);history=[];trace=[];steps=0;start=time.monotonic();w=weights('cuda');parent_pred=parent(depth)[2];bestval=metrics(parent_pred['val']['labels'],parent_pred['val']['probabilities'])['balanced_nll'];bestepoch=0
    tm,tr=base.evaluate(ema,data);vm,vr=base.evaluate(ema,val)
    for r,z in [('train',tm),('val',vm)]:agree(z,metrics(parent_pred[r]['labels'],parent_pred[r]['probabilities']))
    engine.save_pt(folder/'best.pt',checkpoint(ema,0,tm,vm))
    for r,p in [('train',tr),('val',vr)]:base.save_np(folder/('best_'+r+'.npz'),p)
    history.append(dict(local_epoch=0,train=tm,val=vm,steps=0,ema_digest=initial,frozen_digest=frozen,alpha=ema.alphas()));bestval=vm['balanced_nll']
    stopped='bounded_short_review'
    for epoch in range(1,cfg['electronic_epochs']+1):
        assert time.monotonic()-start<cfg['maximum_fit_seconds'];order,theta=engine.epoch_plan(epoch,3141,cfg);lr=rates(epoch);model.alpha_cap=ema.alpha_cap=cap(key,epoch)
        for group,value in zip(opt.param_groups,lr):group['lr']=value
        arrays={n:[] for n in ('y','teacher_q','pc','pa')}
        for batch,off in enumerate(range(0,3141,16)):
            idx=order[off:off+16];clean,x,y,target,mask,crop=mixed(depth,data,idx,theta,epoch,batch,cfg,'cuda');model.context();prob,parts,loss=gradients(model,clean,x,y,target,mask,w,cfg,cfg['microbatch_size']);final=batch==196
            if final:boundary=dict(raw_before={n:named[n].detach().cpu().clone() for n in ns},gradients={n:named[n].grad.detach().cpu().clone() for n in ns},optimizer_before=copy.deepcopy(opt.state_dict()),ema_before={n:dict(ema.named_parameters())[n].detach().cpu().clone() for n in ns},groups=groups,epoch=epoch,alpha_cap=cap(key,epoch))
            norm=float(torch.nn.utils.clip_grad_norm_([named[n] for n in ns],1,error_if_nonfinite=True));opt.step();model.project();steps+=1
            with torch.no_grad():
                for n,p in model.named_parameters():
                    if p.requires_grad:dict(ema.named_parameters())[n].mul_(cfg['ema_decay']).add_(p,alpha=1-cfg['ema_decay'])
            ema.project()
            if final:boundary['clip_norm']=norm;engine.save_pt(folder/'optimizer_boundary.pt',boundary)
            for n,v in [('y',y),('teacher_q',target),('pc',prob['pc']),('pa',prob['pa'])]:arrays[n].append(v.detach().cpu().numpy())
            trace.append(dict(local_epoch=epoch,batch=batch,n=len(idx),crop_SHA256=crop,loss=loss,parts=parts,clip_norm=norm,steps=steps,alpha_cap=cap(key,epoch),raw_alpha=model.alphas(),ema_alpha=ema.alphas()))
        base.save_np(folder/f'training_epoch{epoch:02d}.npz',dict(order=order,**{n:np.concatenate(a) for n,a in arrays.items()}));model.noisy=False;model.linear.mask=None;tm,tr=base.evaluate(ema,data);vm,vr=base.evaluate(ema,val);assert frozen_digest(model.state_dict(),depth)==frozen_digest(ema.state_dict(),depth)==frozen;pt=checkpoint(ema,epoch,tm,vm)
        if vm['balanced_nll']<bestval:
            bestval=vm['balanced_nll'];bestepoch=epoch;engine.save_pt(folder/'best.pt',pt)
            for r,p in [('train',tr),('val',vr)]:base.save_np(folder/('best_'+r+'.npz'),p)
        for r,p in [('train',tr),('val',vr)]:base.save_np(folder/('last_'+r+'.npz'),p)
        history.append(dict(local_epoch=epoch,train=tm,val=vm,steps=steps,lr=lr,alpha_cap=cap(key,epoch),order_SHA256=s.digest(order),affine_SHA256=s.digest(theta),raw_digest=s.digest(model.state_dict()),ema_digest=s.digest(ema.state_dict()),frozen_digest=frozen,alpha=ema.alphas(),training_arrays_SHA256=s.sha(folder/f'training_epoch{epoch:02d}.npz')));engine.save_pt(folder/'last.pt',dict(model={n:p.detach().cpu().clone() for n,p in model.state_dict().items()},ema=pt['model'],metrics=pt['metrics'],local_electronic_epoch=epoch,optimizer=opt.state_dict(),steps=steps,sources=sources()));s.save(folder/'history.json',history);s.save(folder/'method_trace.json',trace);s.save(folder/'status.json',dict(state='existing_electronic_adaptation',local_epoch=epoch,train=tm['accuracy'],val=vm['accuracy'],alpha=ema.alphas(),alpha_cap=cap(key,epoch),phase_updates=0));print(json.dumps(dict(key=key,epoch=epoch,train=tm['accuracy'],val=vm['accuracy'],alpha=ema.alphas())),flush=True)
        if epoch==cfg['review_epoch'] and tm['accuracy']<cfg['collapse_train_floor'] and vm['accuracy']<history[0]['val']['accuracy']-cfg['collapse_val_drop']:stopped='early_review_regression';break
    s.save(folder/'fit_result.json',dict(passed=True,actual_electronic_epochs=epoch,new_joint_epochs=0,optical_phase_updates=0,parent_experiment_joint_epochs=q['joint_budget_epochs'],parent_selected_optical_epoch=q['selected_optical_epoch'],local_epoch_semantics=True,optimizer_steps=steps,best_VAL_local_epoch=bestepoch,stop_reason=stopped,checkpoint_SHA256={n:s.sha(folder/(n+'.pt')) for n in ('best','last')},boundary_SHA256=s.sha(folder/'optimizer_boundary.pt'),history_SHA256=s.sha(folder/'history.json'),trace_SHA256=s.sha(folder/'method_trace.json'),fit_seconds=time.monotonic()-start),exclusive=True)
def test(key,gpu):
    ready();f=OUT/key;rec=s.read(f/'fit_audit_completed.json');assert rec['actual_audit_returncode']==0 and rec['audit_sha256']==s.sha(f/'fit_science/audit.json');claim(f/'test_claim.json','two_development_candidates_best_VAL_and_last');configure(gpu);torch.set_num_threads(4);depth=config()['cases'][key]['depth'];model=initialize(depth).cuda().eval();pred=parent(depth)[2];known={s.digest(initial_state(depth)):dict(metrics=metrics(pred['test']['labels'],pred['test']['probabilities']),predictions=pred['test'])};records=[];data=None;new=0;(f/'test_digest_claims').mkdir()
    for name in ('best','last'):
        pt=torch.load(f/(name+'.pt'),map_location='cpu',weights_only=False);state=pt['model'] if name=='best' else pt['ema'];digest=s.digest(state);reused=digest in known
        if not reused:
            claim(f/'test_digest_claims'/(digest+'.json'),'one_new_parameter_digest');model.load_state_dict(state);model.eval();model.noisy=False;model.linear.mask=None
            if data is None:data=base.load('test')
            mm,pp=base.evaluate(model,data);known[digest]=dict(metrics=mm,predictions=pp);new+=1
        z=known[digest];base.save_np(f/('test_'+name+'.npz'),z['predictions']);records.append(dict(candidate=name,local_epoch=pt['local_electronic_epoch'],model_digest=digest,reused=reused,metrics=z['metrics'],val_balanced_nll=pt['metrics']['val']['balanced_nll']))
    chosen=max(records,key=lambda r:(int(round(411*r['metrics']['accuracy'])),-r['val_balanced_nll'],-r['local_epoch']));parent_count=int(np.trace(np.asarray(metrics(pred['test']['labels'],pred['test']['probabilities'])['confusion_matrix'])));improved=int(np.trace(np.asarray(chosen['metrics']['confusion_matrix'])))>parent_count
    s.save(f/'test_result.json',dict(passed=True,records=records,selected=chosen,selected_checkpoint=dict(path=str(f/(chosen['candidate']+'.pt')),sha256=s.sha(f/(chosen['candidate']+'.pt')),state_key='model' if chosen['candidate']=='best' else 'ema'),new_TEST_parameter_digests=new,TEST_used_for_selection=True,independent_generalization=False,parent_TEST_reused=True,global_incumbent_improved=improved,global_earliest_canonical_retained_on_correct_count_ties=True),exclusive=True)
def campaign():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not OUT.exists();ready();claim(OUT.with_name(OUT.name+'.claim.json'),'four_warm_electronic_curriculum_cases');OUT.mkdir();s.save(OUT/'metadata.json',dict(identity(),config=config(),pid=os.getpid(),controls=closure.old.controls()),exclusive=True);pending=list(config()['cases']);jobs=[];complete=[];failures=[]
    while pending or jobs:
        slots=resources.slots();used={j['gpu'] for j in jobs}
        for index in config()['gpu_order']:
            gpu=resources.GPU[index]
            if not pending or gpu in used or slots['counts'][gpu]:continue
            key=pending.pop(0);s.gpu_budget.ALLOWED.update(resources.GPU.values());s.gpu_budget.preflight(gpu,shared=True,memory_gib=2);log=(OUT/(key+'.fit.log')).open('x');p=subprocess.Popen([os.sys.executable,str(HERE/FILES[0]),'--phase','fit','--key',key,'--gpu',gpu],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT);jobs.append(dict(key=key,gpu=gpu,pid=p.pid,process=p,log=log));used.add(gpu)
        for job in list(jobs):
            rc=job['process'].poll()
            if rc is None:continue
            job['log'].close();ex={k:v for k,v in job.items() if k not in ('process','log')};ex.update(returncode=rc,actual_exit=True);s.save(OUT/(job['key']+'.producer_exit.json'),ex,exclusive=True);jobs.remove(job);f=OUT/job['key']
            if rc:failures.append(ex);pending=[];continue
            def audit(phase):
                claim(f/(phase+'_audit_claim.json'),'independent_electronic_curriculum_'+phase)
                with (f/(phase+'_audit.log')).open('x') as log:arc=subprocess.run([os.sys.executable,str(HERE/FILES[3]),'--key',job['key'],'--phase',phase],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT).returncode
                if arc:return False
                assert s.read(f/(phase+'_science/audit.json'))['passed'];s.save(f/(phase+'_audit_completed.json'),dict(passed=True,actual_audit_returncode=arc,audit_sha256=s.sha(f/(phase+'_science/audit.json'))),exclusive=True);return True
            if not audit('fit'):failures.append(dict(key=job['key'],phase='fit_science'));pending=[];continue
            with (f/'test.log').open('x') as log:p=subprocess.Popen([os.sys.executable,str(HERE/FILES[0]),'--phase','test','--key',job['key'],'--gpu',job['gpu']],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=job['gpu']),stdout=log,stderr=subprocess.STDOUT);trc=p.wait()
            s.save(f/'test_exit.json',dict(pid=p.pid,returncode=trc,actual_exit=True),exclusive=True)
            if trc or not audit('test'):failures.append(dict(key=job['key'],phase='test_or_test_science'));pending=[];continue
            complete.append(dict(key=job['key'],result=s.read(f/'test_result.json')))
        s.save(OUT/'status.json',dict(state='failed' if failures else 'warm_electronic_cases_running' if jobs or pending else 'all_warm_electronic_cases_reviewed',active=[{k:v for k,v in j.items() if k not in ('process','log')} for j in jobs],pending=pending,complete=complete,failures=failures))
        if pending or jobs:time.sleep(2)
    assert not failures
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','test','campaign'],required=True);p.add_argument('--key');p.add_argument('--gpu');a=p.parse_args()
    try:campaign() if a.phase=='campaign' else globals()[a.phase](a.key,a.gpu)
    except BaseException as e:
        f=OUT/a.key if a.key else OUT;mark=f/'failure.json' if f.exists() else f.with_name(f.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
