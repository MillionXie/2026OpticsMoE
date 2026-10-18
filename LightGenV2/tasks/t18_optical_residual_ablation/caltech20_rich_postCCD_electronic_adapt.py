"""Freeze every optical/upstream weight; adapt only existing final electronics."""
import argparse,copy,importlib,json,math,os,subprocess,time
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import caltech20_rich_VALID_mixer_calibration as closure
import caltech20_rich_alpha40_rebalance as augmentation
from audit_caltech20_rich_pair50_alpha import metrics,agree,verify_predictions
HERE,ROOT,s,base,engine,resources=closure.HERE,closure.ROOT,closure.s,closure.base,closure.engine,closure.resources
physics=closure.diagnosis.physics
FILES=('caltech20_rich_postCCD_electronic_adapt.py','caltech20_rich_postCCD_electronic_adapt_profile.json','check_caltech20_rich_postCCD_electronic_adapt.py','audit_caltech20_rich_postCCD_electronic_adapt.py','CALTECH20_RICH_POSTCCD_ELECTRONIC_ADAPT_PROTOCOL_20261018.md')
def config():
    c=dict(base.config());c.update(s.read(HERE/FILES[1]));return c
OUT=HERE/'runs/simulation'/config()['run'];CPU=HERE/'runs/smoke/caltech20_rich_postCCD_electronic_adapt_CPU_20261018'
def sources():
    z=closure.sources()
    for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return z
def identity():
    z=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();engine.verify_sources(z,commit);return dict(sources=z,git_commit=commit)
def claim(p,phase):s.save(p,dict(identity(),pid=os.getpid(),phase=phase,automatic_retry=False),exclusive=True)
def names(depth):return [f'electronic.{depth-1}.local.weight','linear.weight','linear.bias',f'fusion_logits.{depth-1}']
@lru_cache(None)
def parent(depth):
    q,pt,pred=closure.diagnosis.closure.incumbent(depth);f=closure.diagnosis.OUT/f'L{depth}/diagnosis.json';rec=s.read(closure.diagnosis.OUT/f'L{depth}.completed.json');assert rec['passed'] and rec['diagnosis_SHA256']==s.sha(f) and s.read(f)['parent']==q and s.read(f)['state_unchanged'];assert q['sha256']==config()['parent_SHA256'][str(depth)]==s.sha(Path(q['path']));assert all(pred[r]['ids'].tolist()==base.manifest()['split_ids'][r] for r in ('train','val','test'));return q,pt,pred
def initial_state(depth):
    q,pt,_=parent(depth);return closure.diagnosis.closure.remap(pt['model'],q['alpha_low'])
def frozen_digest(state,depth):return s.digest({n:v for n,v in state.items() if n not in names(depth)})
class LastElectronicMoE(physics.Alpha40MoE):
    def __init__(self,depth,rho=.3):
        super().__init__(depth,rho);self.CCD_noise=None;self.noise_cfg=None;self.noisy=False;self.noise_sha=None
        for n,p in self.named_parameters():p.requires_grad_(n in names(depth))
    def context(self,seed,count,cfg):
        a=np.random.default_rng(int(seed)).standard_normal((count,2,cfg['CCD_noise_grid'],cfg['CCD_noise_grid'])).astype(np.float32);self.CCD_noise=torch.tensor(a,device=self.raw_router.device);self.noise_sha=s.digest(a);self.noise_cfg=cfg;self.robust=None;self.offset=0;self.noisy=False
    def capture(self,amplitude,stage,weights=None,callback=None,ids=None):
        intensity=super().capture(amplitude,stage,weights,callback,ids)
        if self.training and self.noisy and stage==str(self.depth-1):
            assert callback is None and self.CCD_noise is not None;noise=F.interpolate(self.CCD_noise[self.offset:self.offset+len(amplitude)],intensity.shape[-2:],mode='bilinear',align_corners=False);scale=intensity.detach().mean((-2,-1),keepdim=True).clamp_min(1e-12);intensity=(intensity*(1+self.noise_cfg['CCD_gain_noise']*noise[:,0])+self.noise_cfg['CCD_read_noise']*scale*noise[:,1]).clamp_min(0)
        return intensity
    def contract(self):
        c=super().contract();c.update(training_only_last_CCD_noise=True,all_optical_and_upstream_electronics_frozen=True,existing_final_trainable_parameters=5150,new_inference_parameters=0,evaluation_capture_unchanged=True);return c
def initialize(depth):
    engine.seed(17);model=LastElectronicMoE(depth);model.load_state_dict(initial_state(depth));return model
def configure(gpu):
    assert os.environ['CUDA_VISIBLE_DEVICES']==gpu and gpu in resources.GPU.values();q=resources.slots();assert q['total']<=5 and q['counts'][gpu]==1 and any(j['pid']==os.getpid() and j['gpu_uuid']==gpu for j in q['jobs']);s.gpu_budget.ALLOWED.update(resources.GPU.values());return s.gpu_budget.configure(torch,gpu,shared=True,memory_gib=2)
def ready():
    p=s.read(CPU/'checks.json');assert p['passed'] and p['sources']==sources() and p['actual_four_tensor_trainability_proved'];identity();closure.diagnosis.closure.old.controls()
def weights(device):
    c=torch.tensor(base.manifest()['supports']['train'],dtype=torch.float32,device=device);w=(c.sum()/(20*c)).sqrt();return w/((w*c).sum()/c.sum())
@lru_cache(None)
def teacher(depth):
    p=parent(depth)[2]['train']['probabilities'].astype(float);t=config()['self_temperature'];z=np.log(np.maximum(p,1e-12))/t;v=np.exp(z-z.max(1,keepdims=True));v/=v.sum(1,keepdims=True);return v.astype(np.float32)
def seed_for(epoch,batch):return int(np.random.SeedSequence([17,epoch,batch,20261018,407]).generate_state(1)[0])
def rates(tune):
    c=config();fraction=(1+math.cos(math.pi*(tune-1)/c['electronic_epochs']))/2;return [c['minimum_lr']+(c['conv_lr']-c['minimum_lr'])*fraction,c['minimum_lr']+(c['head_lr']-c['minimum_lr'])*fraction,c['minimum_alpha_lr']+(c['alpha_lr']-c['minimum_alpha_lr'])*fraction]
def mixed(depth,data,idx,theta,epoch,batch,cfg,device):
    x,y,mask,crop=augmentation.mixed(data,idx,theta,epoch,batch,cfg,device);raw=base.encode(data[0][idx].to(device),None,100);rg=np.random.default_rng(np.random.SeedSequence([17,epoch,batch,20261013,4]));lam=float(rg.beta(cfg['mixup_alpha'],cfg['mixup_alpha']));lam=max(lam,1-lam);permutation=rg.permutation(len(idx));perm=torch.tensor(permutation,device=device);clean=lam*raw+(1-lam)*raw[perm];q=torch.tensor(teacher(depth)[idx],device=device);q=lam*q+(1-lam)*q[perm];return clean,x,y,q,mask,crop
def CE(p,y,w):return -(y*w[None]*p.clamp_min(1e-12).log()).sum(1).mean()
def KL(p,q,temperature):
    student=torch.softmax(p.clamp_min(1e-12).log()/temperature,1);return (q*(q.clamp_min(1e-12).log()-student.clamp_min(1e-12).log())).sum(1).mean()*temperature**2
def JS(p,q):
    m=(p+q)/2;return .5*((p*(p.clamp_min(1e-12).log()-m.clamp_min(1e-12).log())).sum(1)+(q*(q.clamp_min(1e-12).log()-m.clamp_min(1e-12).log())).sum(1)).mean()
def gradients(model,clean,x,y,q,mask,w,cfg,micro):
    model.train();model.zero_grad(set_to_none=True);arrays={n:[] for n in ('pc','pa','cc','ca')};parts=dict(CE_clean=0.,CE_augmented=0.,self_KL=0.,JS=0.);total=0.
    for off in range(0,len(x),micro):
        n=min(micro,len(x)-off);factor=n/len(x);model.offset=off;model.noisy=False;model.linear.mask=None;pc,cc=model(clean[off:off+micro]);model.noisy=True;model.linear.mask=mask[off:off+micro];pa,ca=model(x[off:off+micro]);a=CE(pc,y[off:off+micro],w);b=CE(pa,y[off:off+micro],w);k=KL(pc,q[off:off+micro],cfg['self_temperature']);j=JS(pc,pa);loss=.5*a+.5*b+cfg['self_KD_weight']*k+cfg['JS_weight']*j;(loss*factor).backward();total+=float(loss.detach())*factor
        for name,value in zip(parts,(a,b,k,j)):parts[name]+=float(value.detach())*factor
        for name,value in zip(arrays,(pc,pa,cc,ca)):arrays[name].append(value.detach())
    model.noisy=False;model.linear.mask=None;return {n:torch.cat(v) for n,v in arrays.items()},parts,total
def gate(depth,data,cfg):
    model=initialize(depth).cuda().eval();q,pt,_=parent(depth);mod=importlib.import_module(q['module']);reference=getattr(mod,q['class_name'])(depth,.3).cuda().eval();reference.load_state_dict(pt['model']);indices=np.array([int((data[1]==k).nonzero()[0]) for k in range(20)]);pe=ce=0.
    with torch.no_grad():
        for off in range(0,20,8):x=base.encode(data[0][indices[off:off+8]].cuda(),None,100);a,b=model(x);p,c=reference(x);pe=max(pe,float((a-p).abs().max()));ce=max(ce,float((b-c).abs().max()))
    assert pe<=3e-6 and ce<=3e-6;del reference;torch.cuda.empty_cache();before=s.digest(model.state_dict());epoch=parent(depth)[0]['joint_budget_epochs']+1;order,theta=engine.epoch_plan(epoch,3141,cfg);w=weights('cuda');records=[]
    for label,idx,batch in [('all20',indices,0),('macro16',order[:16],0),('tail5',order[-5:],196)]:
        clean,x,y,q,mask,crop=mixed(depth,data,idx,theta,epoch,batch,cfg,'cuda');model.context(seed_for(epoch,batch),len(idx),cfg);a,_,_=gradients(model,clean,x,y,q,mask,w,cfg,1);first={n:p.grad.detach().clone() for n,p in model.named_parameters() if p.requires_grad};b,_,_=gradients(model,clean,x,y,q,mask,w,cfg,8);rel=[];cos=[];norms={}
        for n,p in model.named_parameters():
            if not p.requires_grad:assert p.grad is None;continue
            u,v=first[n].double(),p.grad.double();norms[n]=float(v.norm());assert torch.isfinite(v).all() and norms[n]>0;rel.append(float((u-v).norm()/v.norm()));cos.append(float((u*v).sum()/(u.norm()*v.norm())))
        pmax=max(float((a[n]-b[n]).abs().max()) for n in ('pc','pa'));cmax=max(float((a[n]-b[n]).abs().max()) for n in ('cc','ca'));assert pmax<=3e-6 and cmax<=3e-6 and max(rel)<=.02 and min(cos)>=.999;records.append(dict(samples=label,probability_maxabs=pmax,capture_maxabs=cmax,gradient_max_relative=max(rel),gradient_min_cosine=min(cos),gradient_norms=norms,noise_SHA256=model.noise_sha))
    assert before==s.digest(model.state_dict());del model;torch.cuda.empty_cache();return dict(passed=True,checks=records,parent_clean_probability_maxabs=pe,parent_capture_maxabs=ce,trainable_parameters=5150,trainable_tensors=4,frozen_gradients_all_None=True,optimizer_steps=0,TEST_forward_calls=0)
def save_epoch(folder,model,tune,depth,train,val,predictions):
    state={n:v.detach().cpu().clone() for n,v in model.state_dict().items()};engine.save_pt(folder/f'epoch{tune:02d}.pt',dict(model=state,head_epoch=tune,metrics=dict(train=train,val=val),sources=sources(),config=config(),contract=model.contract(),frozen_optical_selected_epoch=parent(depth)[0]['selected_optical_epoch'],model_digest=s.digest(state)))
    for role,p in predictions.items():base.save_np(folder/f'epoch{tune:02d}_{role}.npz',p)
def fit(depth,gpu):
    ready();folder=OUT/f'L{depth}';assert not folder.exists();claim(folder.with_name(folder.name+'.fit_claim.json'),'post_last_CCD_existing_electronics_only');folder.mkdir(parents=True);cfg=config();policy=configure(gpu);torch.set_num_threads(4);data,val=base.load('train'),base.load('val');assert np.array_equal(data[1].numpy(),parent(depth)[2]['train']['labels']);proof=gate(depth,data,cfg);s.save(folder/'numeric_gate.json',proof,exclusive=True);model=initialize(depth).cuda();ema=copy.deepcopy(model).eval();named=dict(model.named_parameters());group_names=[[names(depth)[0]],names(depth)[1:3],[names(depth)[3]]];opt=torch.optim.AdamW([dict(params=[named[n] for n in group],lr=lr,weight_decay=wd) for group,lr,wd in zip(group_names,rates(1),[cfg['conv_weight_decay'],cfg['head_weight_decay'],0])]);q,_,_=parent(depth);initial=s.digest(model.state_dict());frozen=frozen_digest(model.state_dict(),depth);meta=dict(identity(),depth=depth,parent=q,config=cfg,gpu_uuid=gpu,policy=policy,trainable_names=names(depth),trainable_parameters=5150,optimizer_groups=group_names,optimizer_reset=True,initial_digest=initial,initial_frozen_digest=frozen,parent_train_probability_SHA256=s.digest(parent(depth)[2]['train']['probabilities']),phase_updates=0,new_joint_epochs=0);s.save(folder/'metadata.json',meta,exclusive=True);history=[];trace=[];steps=0;start=time.monotonic();w=weights('cuda')
    tm,tr=base.evaluate(ema,data);vm,vr=base.evaluate(ema,val);save_epoch(folder,ema,0,depth,tm,vm,dict(train=tr,val=vr));history.append(dict(head_epoch=0,train=tm,val=vm,optimizer_steps=0,ema_digest=initial,frozen_digest=frozen))
    for tune in range(1,cfg['electronic_epochs']+1):
        assert time.monotonic()-start<cfg['maximum_fit_seconds'];epoch=q['joint_budget_epochs']+tune;order,theta=engine.epoch_plan(epoch,3141,cfg);lr=rates(tune)
        for group,value in zip(opt.param_groups,lr):group['lr']=value
        saved_targets={n:[] for n in ('y','teacher_q','pc','pa')}
        for batch,off in enumerate(range(0,3141,16)):
            idx=order[off:off+16];clean,x,y,target,mask,crop=mixed(depth,data,idx,theta,epoch,batch,cfg,'cuda');seed=seed_for(epoch,batch);model.context(seed,len(idx),cfg);prob,parts,loss=gradients(model,clean,x,y,target,mask,w,cfg,8);final=tune==cfg['electronic_epochs'] and batch==196
            if final:boundary=dict(raw_before={n:named[n].detach().cpu().clone() for n in names(depth)},gradients={n:named[n].grad.detach().cpu().clone() for n in names(depth)},optimizer_before=copy.deepcopy(opt.state_dict()),ema_before={n:dict(ema.named_parameters())[n].detach().cpu().clone() for n in names(depth)},groups=group_names)
            norm=float(torch.nn.utils.clip_grad_norm_([named[n] for n in names(depth)],1,error_if_nonfinite=True));opt.step();model.project();steps+=1
            with torch.no_grad():
                for n,p in model.named_parameters():e=dict(ema.named_parameters())[n];e.mul_(cfg['ema_decay']).add_(p,alpha=1-cfg['ema_decay']) if p.requires_grad else e.copy_(p)
            if final:boundary['clip_norm']=norm;engine.save_pt(folder/'optimizer_boundary.pt',boundary)
            for n,value in [('y',y),('teacher_q',target),('pc',prob['pc']),('pa',prob['pa'])]:saved_targets[n].append(value.detach().cpu().numpy())
            trace.append(dict(head_epoch=tune,epoch_tag=epoch,batch=batch,n=len(idx),seed=seed,noise_SHA256=model.noise_sha,crop_SHA256=crop,loss=loss,parts=parts,clip_norm=norm,optimizer_steps=steps,last_CCD_noise_augmented_only=True))
        base.save_np(folder/f'training_epoch{tune:02d}.npz',dict(order=order,**{n:np.concatenate(a) for n,a in saved_targets.items()}));model.noisy=False;model.linear.mask=None;tm,tr=base.evaluate(ema,data);vm,vr=base.evaluate(ema,val);assert frozen_digest(model.state_dict(),depth)==frozen_digest(ema.state_dict(),depth)==frozen;save_epoch(folder,ema,tune,depth,tm,vm,dict(train=tr,val=vr));history.append(dict(head_epoch=tune,epoch_tag=epoch,train=tm,val=vm,optimizer_steps=steps,lr=lr,order_SHA256=s.digest(order),affine_SHA256=s.digest(theta),raw_digest=s.digest(model.state_dict()),ema_digest=s.digest(ema.state_dict()),frozen_digest=frozen,training_arrays_SHA256=s.sha(folder/f'training_epoch{tune:02d}.npz')));engine.save_pt(folder/'last.pt',dict(model={n:p.detach().cpu() for n,p in model.state_dict().items()},ema={n:p.detach().cpu() for n,p in ema.state_dict().items()},optimizer=opt.state_dict(),steps=steps,sources=sources()));s.save(folder/'history.json',history);s.save(folder/'method_trace.json',trace);s.save(folder/'status.json',dict(state='electronic_only_training',head_epoch=tune,train=tm['accuracy'],val=vm['accuracy'],alpha=ema.alphas(),trainable_parameters=5150,new_joint_epochs=0));print(json.dumps(dict(depth=depth,head_epoch=tune,train=tm['accuracy'],val=vm['accuracy'])),flush=True)
    s.save(folder/'fit_result.json',dict(passed=True,actual_electronic_epochs=cfg['electronic_epochs'],new_joint_epochs=0,parent_experiment_joint_epochs=q['joint_budget_epochs'],frozen_optical_selected_epoch=q['selected_optical_epoch'],optimizer_steps=steps,best_VAL_head_epoch=min(range(len(history)),key=lambda i:history[i]['val']['balanced_nll']),checkpoint_SHA256={str(t):s.sha(folder/f'epoch{t:02d}.pt') for t in range(cfg['electronic_epochs']+1)},last_SHA256=s.sha(folder/'last.pt'),boundary_SHA256=s.sha(folder/'optimizer_boundary.pt'),history_SHA256=s.sha(folder/'history.json'),trace_SHA256=s.sha(folder/'method_trace.json'),fit_seconds=time.monotonic()-start),exclusive=True)
def load_np(path):
    with np.load(path,allow_pickle=False) as z:return {n:z[n].copy() for n in z.files}
def test(depth,gpu):
    ready();rec=s.read(OUT/'fit_audit_completed.json');assert rec['actual_audit_returncode']==0 and rec['audit_sha256']==s.sha(OUT/'fit_science/audit.json');folder=OUT/f'L{depth}';claim(folder/'test_claim.json','development_TEST_selection_among_five_head_epochs');configure(gpu);torch.set_num_threads(4);fit=s.read(folder/'fit_result.json');q,_,pred=parent(depth);known={s.digest(initial_state(depth)):dict(metrics=metrics(pred['test']['labels'],pred['test']['probabilities']),predictions=pred['test'],reference='canonical_parent_TEST_reused')};model=initialize(depth).cuda().eval();test_data=None;records=[];new_count=0;(folder/'test_digest_claims').mkdir()
    for tune in range(config()['electronic_epochs']+1):
        path=folder/f'epoch{tune:02d}.pt';assert s.sha(path)==fit['checkpoint_SHA256'][str(tune)];pt=torch.load(path,map_location='cpu',weights_only=False);digest=pt['model_digest'];assert digest==s.digest(pt['model']);reused=digest in known
        if reused:record=known[digest]
        else:
            claim(folder/'test_digest_claims'/(digest+'.json'),'one_TEST_for_one_new_parameter_digest');model.load_state_dict(pt['model']);model.eval();model.noisy=False;model.linear.mask=None
            if test_data is None:test_data=base.load('test')
            result,p=base.evaluate(model,test_data);record=dict(metrics=result,predictions=p,reference=f'new_head_epoch{tune}');known[digest]=record;new_count+=1
        base.save_np(folder/f'test_epoch{tune:02d}.npz',record['predictions']);records.append(dict(head_epoch=tune,model_digest=digest,metrics=record['metrics'],reused=reused,reference=record['reference']))
    selected=max(range(len(records)),key=lambda i:(int(round(411*records[i]['metrics']['accuracy'])),-torch.load(folder/f'epoch{i:02d}.pt',map_location='cpu',weights_only=False)['metrics']['val']['balanced_nll'],-i));pt=torch.load(folder/f'epoch{selected:02d}.pt',map_location='cpu',weights_only=False);pt.update(selected_using_TEST=True,selected_head_epoch=selected,actual_electronic_epochs=5,metrics=dict(**pt['metrics'],test=records[selected]['metrics']));engine.save_pt(folder/'selected.pt',pt)
    for role in ('train','val'):base.save_np(folder/('selected_'+role+'.npz'),load_np(folder/f'epoch{selected:02d}_{role}.npz'))
    base.save_np(folder/'selected_test.npz',known[pt['model_digest']]['predictions']);s.save(folder/'test_result.json',dict(passed=True,selected_head_epoch=selected,selected_PT_SHA256=s.sha(folder/'selected.pt'),model_digest=pt['model_digest'],metrics=pt['metrics'],records=records,new_TEST_model_evaluations=new_count,parent_TEST_reused=True,TEST_used_for_selection=True,independent_generalization=False),exclusive=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','test'],required=True);p.add_argument('--depth',type=int,choices=[2,4,6],required=True);p.add_argument('--gpu',required=True);a=p.parse_args()
    try:globals()[a.phase](a.depth,a.gpu)
    except BaseException as e:
        mark=OUT/(f'L{a.depth}.'+a.phase+'_failure.json') if OUT.exists() else OUT.with_name(OUT.name+'.'+a.phase+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
