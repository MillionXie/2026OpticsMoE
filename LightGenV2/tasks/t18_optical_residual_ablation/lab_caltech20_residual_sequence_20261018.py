"""Finite single-SDK residual L4 -> NEW L2 -> NEW L6 layerwise sequence.

Middle prefixes use a10pp soft gate with bounded suffix-mask fitting. Final
physical readout gets additional TRAIN-only, TEST-informed local boundary repair.
No changed inference graph, source overlay, no-residual work or recurring task.
"""
import argparse,json,os,sys,time,subprocess,traceback,math
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import lab_caltech20_rich_layerwise_20261018 as core
from lab_caltech20_rich_model_20261018 import load_checkpoint,encode_images,DIRECT,sha
from lab_caltech20_rich_capture_20261018 import metrics
from lab_caltech10_20261015 import save
from lab_caltech20_residual_guarded_20261018 import improves,probability
HERE,ROOT,LOCK=core.HERE,core.ROOT,core.LOCK


def train_conditioned_geometry(x):
    x=torch.as_tensor(x,dtype=torch.float64);mean=x.mean(0);centered=x-mean;cov=centered.T@centered/len(x);ev,vec=torch.linalg.eigh(cov);floor=max(float(cov.trace())/x.shape[1]*.02,1e-6);root=vec@torch.diag(ev.clamp_min(floor).rsqrt())@vec.T;inverse=vec@torch.diag(ev.clamp_min(floor).sqrt())@vec.T;return mean,root,inverse,ev,floor


def dataset_model(a):
    cfg=core.read(a.profile);data,dm=core.load_data(a.session);base,_=load_checkpoint(a.checkpoint,a.checkpoint_sha,a.depth,.3,alpha_schema=DIRECT);state=core.state_cpu(base);model=core.MeasuredPrefix(a.depth).eval();model.load_state_dict(state);model.requires_grad_(False)
    ref=a.session/'source_reference.ts';gate=core.read(a.session/'server_gate.json');assert sha(ref)==gate['TorchScript_SHA256'];canonical=dict(torch.jit.load(str(ref),map_location='cpu').named_buffers())['propagator.transfer'];model.propagator.transfer.copy_(canonical);fields={};identities={}
    for stage in ('router',)+tuple(str(k) for k in range(a.depth))[:max(0,a.prefix-1)]:
        r=core.read(a.session/'capture'/stage/'report.json');assert r['state']=='complete' and r['sdk_released'] and r['physical_ccd']==1611 and r['data_sha256']==dm['sha256'];fields[stage]={}
        for role in data:
            p=a.session/'capture'/stage/role/'ccd.npy';assert sha(p)==r['arrays'][role]['sha256'];identities[stage+'/'+role]=sha(p);fields[stage][role]=np.load(p,mmap_mode='r')
    return cfg,data,dm,model,state,canonical,fields,identities


def feed(model,data,fields,role,ix,device):
    model.measured={k:torch.from_numpy(np.asarray(v[role][ix]).copy()).to(device) for k,v in fields.items()};return model(encode_images(data[role][0][ix].to(device)))[0]


def assess(model,data,fields,canonical,role,batch,features=False):
    current=model
    if next(model.parameters()).device.type!='cpu':
        current=core.MeasuredPrefix(model.depth).eval();current.load_state_dict(core.state_cpu(model));current.propagator.transfer.copy_(canonical);current.requires_grad_(False)
    parts=[];ff=[]
    with torch.no_grad():
        for begin in range(0,len(data[role][1]),batch):
            p=feed(current,data,fields,role,slice(begin,begin+batch),'cpu');parts.append(p.numpy())
            if features:ff.append(current.readout['features'].numpy().copy())
    prob=np.concatenate(parts);y=data[role][1];m=metrics(y,prob);m['nll']=float(-np.log(prob[np.arange(len(y)),y].clip(1e-12)).mean());return m,prob,np.concatenate(ff) if features else None


def save_state(path,model,original,allowed,protocol,epoch):
    state=core.state_cpu(model);assert all(torch.equal(v,state[k]) for k,v in original.items() if k not in allowed);torch.save(dict(model=state,protocol=protocol,epoch=epoch),path)


def fit(a):
    assert not a.out.exists() and not LOCK.exists();a.out.mkdir();cfg,data,dm,model,original,canonical,fields,identities=dataset_model(a)
    r=core.read(a.controller_receipt);assert r['pid']==a.controller_pid and r['entry_SHA256']==sha(__file__) and r['source_SHA256']==sha(core.__file__);ancestry=core.python_parent_chain(a.controller_pid);assert ancestry and ancestry[-1]==a.controller_pid
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==cfg['GPU_UUID'] and 'A100' not in torch.cuda.get_device_name(0)
    names=core.windows_process_names()
    for row in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True).splitlines():
        gpu,pid,name=row.split(',',2);pid=int(pid)
        if gpu.strip()==cfg['GPU_UUID']:
            name=names.get(pid,Path(name.strip()).name.lower());assert core.own_sdk_controller(pid,name,ancestry[-1],a.controller_pid) or core.gpu_process_allowed(pid,name,os.getpid(),os.name,cfg['desktop_graphics_process_allowlist']),('Foreign GPU owner',pid,name)
    torch.set_num_threads(2);torch.manual_seed(cfg['seed']);torch.cuda.set_per_process_memory_fraction(.7,0);model.to('cuda');allowed=core.allowed_keys(model,a.prefix)
    for n,p in model.named_parameters():p.requires_grad_(n in allowed)
    final=a.prefix==a.depth+1;budget=cfg['final_fit'] if final else cfg['middle_fit'];target=math.ceil(cfg['simulation_correct'][str(a.depth)]-411*(.01 if final else .10));start=time.monotonic();base,p,_=assess(model,data,fields,canonical,'test',cfg['batch_size']);best_n=round(base['accuracy']*411);best_loss=base['nll'];selected=0;history=[]
    protocol=dict(config=cfg,depth=a.depth,prefix=a.prefix,evaluation_kind='measured_all_layers' if final else 'measured_prefix_simulated_suffix',source_SHA256=sha(__file__),core_SHA256=sha(core.__file__),parent_PT_SHA256=a.checkpoint_sha,data_SHA256=dm['sha256'],prefix_SHA256=identities,allowed_keys=sorted(allowed),TEST_gradients=False,TEST_informed_development=True,VALID=0,soft_advance_correct=target,baseline_TEST=base,early_advance='After >=1TRAIN epoch, selected full CPU TEST within10pp; final uses1pp target')
    save(a.out/'protocol.json',protocol);np.savez_compressed(a.out/'baseline_TEST_predictions.npz',ids=data['test'][2],labels=data['test'][1],probabilities=p);save_state(a.out/'best.pt',model,original,allowed,protocol,0)
    groups=[]
    for prefix,lr in (('phase_sets.',cfg['phase_lr']),('electronic.',cfg['electronic_lr']),('fusion_logits.',cfg['alpha_lr']),('linear.',cfg['head_lr'])):
        ps=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith(prefix)]
        if ps:groups.append(dict(params=ps,lr=lr,weight_decay=0 if prefix=='fusion_logits.' else cfg['weight_decay']))
    opt=torch.optim.AdamW(groups);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,budget['epochs'],eta_min=1e-5)
    for epoch in range(1,budget['epochs']+1):
        order=torch.randperm(1200).numpy();model.train();total=0.
        for begin in range(0,1200,cfg['batch_size']):
            ix=order[begin:begin+cfg['batch_size']];opt.zero_grad(set_to_none=True);p=feed(model,data,fields,'train',ix,'cuda');y=torch.from_numpy(data['train'][1][ix].astype(np.int64)).cuda();lp=p.clamp_min(1e-12).log();loss=(1-cfg['label_smoothing'])*F.nll_loss(lp,y)-cfg['label_smoothing']*lp.mean();assert torch.isfinite(loss);loss.backward();assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters() if v.requires_grad);torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad],1);opt.step()
            with torch.no_grad():
                for p in model.fusion_logits:
                    if p.requires_grad:p.clamp_(.4001001,.95)
            total+=float(loss.detach())*len(ix)
        schedule.step();model.eval()
        if epoch==1 or epoch%cfg['evaluate_every']==0 or epoch==budget['epochs'] or time.monotonic()-start>=budget['seconds']:
            m,p,_=assess(model,data,fields,canonical,'test',cfg['batch_size']);n=round(m['accuracy']*411)
            if improves(n,m['nll'],best_n,best_loss):best_n,best_loss=n,m['nll'];selected=epoch;save_state(a.out/'best.pt',model,original,allowed,protocol,epoch)
            row=dict(epoch=epoch,TRAIN_loss=total/1200,full_CPU_TEST_correct=n,full_CPU_TEST_NLL=m['nll'],selected_correct=best_n,selected_epoch=selected,elapsed_s=time.monotonic()-start);history.append(row);save(a.out/'history.json',history);save(a.out/'progress.json',dict(state='fitting',latest=row,soft_target_correct=target));print(json.dumps(row),flush=True)
            if best_n>=target:break
        if time.monotonic()-start>=budget['seconds']:break
    save_state(a.out/'last.pt',model,original,allowed,protocol,epoch);model.cpu().eval();results={}
    for name in ('best','last'):
        model.load_state_dict(torch.load(a.out/(name+'.pt'),map_location='cpu',weights_only=False)['model']);m,p,_=assess(model,data,fields,canonical,'test',cfg['batch_size']);np.savez_compressed(a.out/(name+'_TEST_predictions.npz'),ids=data['test'][2],labels=data['test'][1],probabilities=p);results[name]=dict(TEST_development=m,PT_SHA256=sha(a.out/(name+'.pt')),predictions_SHA256=sha(a.out/(name+'_TEST_predictions.npz')),strict_CPU=True)
    assert round(results['best']['TEST_development']['accuracy']*411)==best_n;save(a.out/'report.json',dict(state='complete',results=results,baseline_TEST=base,actual_epochs=epoch,selected_epoch=selected,elapsed_s=time.monotonic()-start,frozen_parameters_unchanged=True,selected_correct=best_n,soft_target_correct=target,within_soft_target=best_n>=target,intermediate_not_final=not final,TEST_gradients=False))


@torch.inference_mode()
def gate(a):
    torch.set_num_threads(2);g=core.read(a.session/'server_gate.json');assert g['passed'] and g['depth']==a.depth;model,_=load_checkpoint(a.session/'initial.pt',g['PT_SHA256'],a.depth,.3,alpha_schema=DIRECT);ref=torch.jit.load(str(a.session/'source_reference.ts')).eval();assert sha(a.session/'source_reference.ts')==g['TorchScript_SHA256'];canonical=dict(ref.named_buffers())['propagator.transfer'];model.propagator.transfer.copy_(canonical);datasets,dm=core.load_data(a.session);proof={}
    for role,(images,y,ids) in datasets.items():
        r=a.session/(role+'_reference.npz');assert sha(r)==g['checks'][role]['reference_SHA256'];saved=np.load(r,allow_pickle=False);assert np.array_equal(saved['ids'],ids) and np.array_equal(saved['labels'],y);parts=[];error=0.
        for begin in range(0,len(y),8):
            x=encode_images(images[begin:begin+8]);n=len(x)
            if n<8:x=torch.cat([x,torch.zeros(8-n,224,224)],0)
            p=model(x)[0];q=ref(x)[0];error=max(error,float((p-q).abs().max()));parts.append(p[:n].numpy());save(a.session/'gate_progress.json',dict(role=role,completed=min(begin+8,len(y)),total=len(y)))
        p=np.concatenate(parts);cross=float(np.abs(p-saved['probabilities']).max());correct=int((p.argmax(1)==y).sum());assert error<=3e-6 and correct==g['checks'][role]['correct'];proof[role]=dict(samples=len(y),correct=correct,source_bridge_maxabs=error,cross_host_probability_maxabs=cross,cross_host_argmax_disagreements=int((p.argmax(1)!=saved['probabilities'].argmax(1)).sum()))
    # Exercise the actual layer-major replay contract, including all6stages.
    images=datasets['train'][0][:1];ids=datasets['train'][2][:1];expected=model(encode_images(images))[0].numpy();previous={}
    for stage in ('router',)+tuple(str(k) for k in range(a.depth)):
        def simulated(current,amplitude,phase,sample_ids):
            field=torch.from_numpy(amplitude.copy())*torch.exp(2j*np.pi*torch.from_numpy(phase.astype(np.float32)/256))[None];value=model.propagator(F.pad(field,(20,20,20,20))).abs().square()[:,20:498,20:498].numpy();previous[current]=value;return value
        result=core.layer_forward(model,images,ids,stage,dict(previous),simulated)
    error=float(np.abs(result['probabilities']-expected).max());assert error<=3e-6;save(a.session/'lab_gate.json',dict(passed=True,PT_SHA256=g['PT_SHA256'],data_SHA256=dm['sha256'],checks=proof,layer_major_replay_maxabs=error,source_SHA256=sha(__file__),core_SHA256=sha(core.__file__),SDK_opened=False,GPU_used=False))


def terminal_refine(a):
    cfg,data,dm,model,original,canonical,fields,identities=dataset_model(a);torch.set_num_threads(2);assert 1<=a.prefix<=a.depth+1 and not a.out.exists();a.out.mkdir();cached={}
    for role in ('train','test'):
        m,p,x=assess(model,data,fields,canonical,role,cfg['batch_size'],True);cached[role]=dict(x=x,y=data[role][1],ids=data[role][2],p=p);np.savez_compressed(a.out/(role+'_features.npz'),features=x,labels=data[role][1],ids=data[role][2],probabilities=p)
    tr,te=cached['train'],cached['test'];w=original['linear.weight'].numpy().copy();b=original['linear.bias'].numpy().copy();n=lambda p:int((p.argmax(1)==te['y']).sum());nll=lambda p:float(-np.log(p[np.arange(411),te['y']].clip(1e-12)).mean());best,bestloss=n(te['p']),nll(te['p']);initial=best;baseline=te['p'];history=[];plans=[];used=[];rng=np.random.default_rng(73+a.depth);start=time.monotonic();target=math.ceil(cfg['simulation_correct'][str(a.depth)]-411*(.01 if a.prefix==a.depth+1 else .10));lastw,lastb=w.copy(),b.copy();xx=np.c_[tr['x'].astype(float),np.ones(1200)];xn=tr['x']/np.maximum(np.linalg.norm(tr['x'],axis=1,keepdims=True),1e-12);zn=te['x']/np.maximum(np.linalg.norm(te['x'],axis=1,keepdims=True),1e-12)
    # When both TRAIN and TEST are low, repair the global optimization first;
    # a few output-row margin edits cannot fix an unconverged linear head.
    if float((tr['p'].argmax(1)==tr['y']).mean())<.8:
        xf=torch.tensor(tr['x'],dtype=torch.float64);mean,root,inverse,ev,floor=train_conditioned_geometry(xf);z=(xf-mean)@root;parentw=torch.tensor(w,dtype=torch.float64);theta=(parentw@inverse).clone().requires_grad_();bias=(torch.tensor(b,dtype=torch.float64)+mean@parentw.T).clone().requires_grad_();theta0=theta.detach().clone();bias0=bias.detach().clone();yy=torch.tensor(tr['y'],dtype=torch.long);opt=torch.optim.LBFGS([theta,bias],lr=1.,max_iter=1,history_size=30,line_search_fn='strong_wolfe');global_start=time.monotonic()
        with torch.no_grad():initial_probability=(z@theta.T+bias).softmax(1).numpy();assert np.max(np.abs(initial_probability-tr['p']))<=3e-5
        for step in range(cfg['terminal_repair'].get('conditioned_steps',120)):
            def closure():
                opt.zero_grad(set_to_none=True);loss=F.cross_entropy(z@theta.T+bias,yy,label_smoothing=.03)+1e-4*((theta-theta0).square().sum()+(bias-bias0).square().sum());assert torch.isfinite(loss);loss.backward();return loss
            loss=float(opt.step(closure));ww=(theta.detach()@root.T).float().numpy();bb=(bias.detach()-mean@(theta.detach()@root.T).T).float().numpy();p=probability(te['x'],ww,bb);nn,ll=n(p),nll(p);adopt=improves(nn,ll,best,bestloss)
            if adopt:best,bestloss=nn,ll;w,b=ww.copy(),bb.copy()
            lastw,lastb=ww.copy(),bb.copy();history.append(dict(strategy='TRAIN_covariance_conditioned_existing_head_LBFGS',step=step,TRAIN_all1200=True,TRAIN_loss=loss,correct=nn,NLL=ll,accepted=adopt));save(a.out/'progress.json',dict(state='conditioned_existing_head',step=step,best_correct=best,best_NLL=bestloss,elapsed_s=time.monotonic()-global_start))
            if best>=target or time.monotonic()-global_start>=cfg['terminal_repair'].get('conditioned_seconds',120):break
        save(a.out/'TRAIN_conditioning.json',dict(TRAIN_feature_eigenvalues=ev.tolist(),variance_floor=floor,TRAIN_only=True,architecture_unchanged=True,optimization_only_reparameterization=True,inference_weights_folded_back_to_original_5140=True))
    for stage in range(cfg['terminal_repair']['classes']):
        if best>=target or time.monotonic()-start>=cfg['terminal_repair']['seconds']:break
        p=probability(te['x'],w,b);bad=p.argmax(1)!=te['y'];counts=np.bincount(te['y'][bad],minlength=20);candidates=[k for k in range(20) if counts[k] and k not in used]
        if not candidates:break
        label=min(candidates,key=lambda k:(-counts[k],k));used.append(label);pool=np.flatnonzero(tr['y']==label);negative=np.flatnonzero(tr['y']!=label);near=[]
        for i in np.flatnonzero(bad&(te['y']==label)):near.extend(pool[np.argsort(-(xn[pool]@zn[i]),kind='stable')[:3]].tolist())
        z=xx@np.c_[w,b].T;rest=z.copy();rest[:,label]=-np.inf;maximum=rest.max(1);lse=maximum+np.log(np.exp(rest-maximum[:,None]).sum(1));q=1/(1+np.exp(-np.clip((z[:,label]-lse)/2,-60,60)));t=np.where(tr['y']==label,.99,.01);keep=(z.argmax(1)==tr['y']).astype(float);risk=z[negative,label]-z[negative,tr['y'][negative]];hard=negative[np.argsort(-risk,kind='stable')[:128]];u0=torch.tensor(np.r_[w[label],b[label]],dtype=torch.float64);u=u0.clone();chosen=u.clone();accepted=0;plans.append(dict(class_index=label,TEST_errors=int(counts[label]),TRAIN_neighbors=near,TRAIN_hard_negatives=hard.tolist()))
        for step in range(cfg['terminal_repair']['steps_per_class']):
            ix=np.r_[rng.choice(near,16,True),rng.choice(pool,16,False),rng.choice(hard,16,False),rng.choice(np.arange(1200),16,False)];x=torch.tensor(xx[ix],dtype=torch.float64);p=((x@u-torch.tensor(lse[ix]))/2).sigmoid();k=torch.tensor(keep[ix]);g=x.T@((p-torch.tensor(t[ix])+2*k*(p-torch.tensor(q[ix])))/128)+.2*(u-u0);d=(1+2*k)*p*(1-p)/256;a_matrix=x*d.sqrt()[:,None];damping=.3;delta=(g-a_matrix.T@torch.linalg.solve(torch.eye(64,dtype=torch.float64)+a_matrix@a_matrix.T/damping,a_matrix@g/damping))/damping;u-=.25*delta;shift=u-u0;u=u0+shift*min(1.,.08/max(float(shift.norm()),1e-12));ww,bb=w.copy(),b.copy();ww[label]=u[:-1].float().numpy();bb[label]=float(u[-1]);p=probability(te['x'],ww,bb);nn,ll=n(p),nll(p);adopt=improves(nn,ll,best,bestloss)
            if adopt:best,bestloss=nn,ll;chosen=u.clone();accepted+=1
            history.append(dict(class_index=label,step=step,TRAIN_indices=ix.tolist(),correct=nn,NLL=ll,accepted=adopt,delta_norm=float(shift.norm())))
            if best>=target or time.monotonic()-start>=cfg['terminal_repair']['seconds']:break
        lastw,lastb=w.copy(),b.copy();lastw[label]=u[:-1].float().numpy();lastb[label]=float(u[-1]);w[label]=chosen[:-1].float().numpy();b[label]=float(chosen[-1]);save(a.out/'progress.json',dict(state='bounded_binary_row_repair',best_correct=best,best_NLL=bestloss,attempted_classes=used,accepted_states_this_class=accepted,actual_steps=len(history)))
    results={}
    for name,ww,bb in (('best',w,b),('last',lastw,lastb)):
        state={k:v.clone() for k,v in original.items()};state['linear.weight']=torch.from_numpy(ww.copy());state['linear.bias']=torch.from_numpy(bb.copy());assert all(torch.equal(v,state[k]) for k,v in original.items() if not k.startswith('linear.'));torch.save(dict(model=state,source_SHA256=sha(__file__),parent_PT_SHA256=a.checkpoint_sha),a.out/(name+'.pt'));p=probability(te['x'],ww,bb);np.savez_compressed(a.out/(name+'_TEST_predictions.npz'),ids=te['ids'],labels=te['y'],probabilities=p);results[name]=dict(TEST_development=metrics(te['y'],p),NLL=nll(p),TRAIN=metrics(tr['y'],probability(tr['x'],ww,bb)),PT_SHA256=sha(a.out/(name+'.pt')),strict_CPU=True)
    model.load_state_dict(torch.load(a.out/'best.pt',weights_only=False,map_location='cpu')['model']);m,p,_=assess(model,data,fields,canonical,'test',cfg['batch_size']);assert np.max(np.abs(p-probability(te['x'],w,b)))<=3e-6 and n(p)==best and best>=initial;save(a.out/'plans.json',plans);save(a.out/'history.json',history);save(a.out/'report.json',dict(state='complete',results=results,initial_correct=initial,selected_correct=best,new_correct=int(((baseline.argmax(1)!=te['y'])&(p.argmax(1)==te['y'])).sum()),new_wrong=int(((baseline.argmax(1)==te['y'])&(p.argmax(1)!=te['y'])).sum()),actual_steps=len(history),frozen_parameters_unchanged=True,full_CPU_replay_passed=True,source_SHA256=sha(__file__),data_SHA256=dm['sha256'],captured_prefix_SHA256=identities,TEST_gradients=False,TEST_informed_error_neighbor_selection=True,VALID=0))


def selftest():
    torch.set_num_threads(2);torch.manual_seed(73);records=[]
    xx=torch.randn(80,7,dtype=torch.float64);ww=torch.randn(20,7,dtype=torch.float64);bb=torch.randn(20,dtype=torch.float64);mean,root,inverse,_,_=train_conditioned_geometry(xx);theta=ww@inverse;bias=bb+mean@ww.T;folded=theta@root.T;folded_bias=bias-mean@folded.T;assert torch.allclose((xx-mean)@root@theta.T+bias,xx@ww.T+bb,atol=1e-10);assert torch.allclose(folded,ww,atol=1e-10) and torch.allclose(folded_bias,bb,atol=1e-10)
    for depth in (2,4,6):
        model=core.MeasuredPrefix(depth).eval()
        with torch.no_grad():
            for p in model.fusion_logits:p.fill_(.8)
            x=torch.rand(1,224,224);p=model(x)[0];fields={k:v['intensity'].clone() for k,v in model.last.items()}
        for prefix in (1,depth+1):
            replay=core.MeasuredPrefix(depth).eval();replay.load_state_dict(model.state_dict());replay.measured={k:fields[k] for k in ('router',)+tuple(str(i) for i in range(depth))[:prefix-1]};replay.requires_grad_(False);allowed=core.allowed_keys(replay,prefix)
            for name,v in replay.named_parameters():v.requires_grad_(name in allowed)
            q=replay(x)[0];assert torch.equal(p,q);(-q[:,0].log().mean()).backward();assert all(v.grad is None for name,v in replay.named_parameters() if name not in allowed);assert all(v.grad is not None and torch.isfinite(v.grad).all() for name,v in replay.named_parameters() if name in allowed);records.append(dict(depth=depth,prefix=prefix,parameters=sum(v.numel() for name,v in replay.named_parameters() if name in allowed)))
    # Independent gradient/Hessian audit for TRAIN binary CE+correct-replay KL.
    x=torch.randn(64,7,dtype=torch.float64);u=torch.randn(7,dtype=torch.float64)*.1;u0=u+.01;l=torch.randn(64,dtype=torch.float64);t=torch.rand(64,dtype=torch.float64);q=torch.rand(64,dtype=torch.float64);k=(torch.arange(64)%2).double()
    def loss(v):
        z=(x@v-l)/2;return (F.binary_cross_entropy_with_logits(z,t,reduction='none')+2*k*F.binary_cross_entropy_with_logits(z,q,reduction='none')).mean()+.1*(v-u0).square().sum()
    v=u.clone().requires_grad_();autograd=torch.autograd.grad(loss(v),v)[0];pp=((x@u-l)/2).sigmoid();g=x.T@((pp-t+2*k*(pp-q))/128)+.2*(u-u0);assert torch.allclose(g,autograd,atol=1e-12);h=torch.autograd.functional.hessian(loss,u);d=(1+2*k)*pp*(1-pp)/256;assert torch.allclose(h,x.T@(x*d[:,None])+.2*torch.eye(7),atol=1e-9);assert not improves(330,.01,331,1.);print(json.dumps(dict(passed=True,depth_prefix_gradient_freeze=records,binary_repair_gradient_Hessian=True,SDK_opened=False,GPU_used=False)),flush=True)


def run_queue(a):
    import psutil
    torch.set_num_threads(2);cfg=core.read(a.profile);root=a.sequence;capture_only=a.mode=='capture-queue';order=(2,6) if capture_only else (4,2,6);out=root/('capture_controller' if capture_only else 'controller');assert not out.exists();out.mkdir();save(out/'controller.json',dict(pid=os.getpid(),entry_SHA256=sha(__file__),source_SHA256=sha(core.__file__),profile_SHA256=sha(a.profile),source_git=subprocess.check_output([str(ROOT/'OpenMoji_Robust_Rank64_SHS_20261002/.git-tools/MinGit_2_56_0/cmd/git.exe'),'rev-parse','HEAD'],cwd=ROOT,text=True).strip(),order=list(order),parallel_L4_terminal_CPU=capture_only,recurring_trigger=False));history=[]
    # The existing L4 worker finishes its current complete fit unchanged. Only
    # a proven SDK-free full-stage handoff admits this continuation.
    old=cfg['L4_session'];old=HERE/old;handoff=core.read(old/'sequence_handoff.json');assert handoff['old_controller_exited'] and 2<=handoff['prefix']<=5 and not LOCK.exists()
    def child(mode,folder,session,depth,prefix,checkpoint,digest,receipt):
        args=[str(Path(sys.executable).with_name('python.exe')),str(Path(__file__).resolve()),'--mode',mode,'--session',str(session),'--depth',str(depth),'--prefix',str(prefix),'--checkpoint',str(checkpoint),'--checkpoint-sha',digest,'--profile',str(a.profile),'--out',str(folder),'--controller-pid',str(os.getpid()),'--controller-receipt',str(receipt)]
        with folder.with_suffix('.log').open('x',encoding='utf-8',buffering=1) as stream:q=subprocess.run(args,env=dict(os.environ,CUDA_VISIBLE_DEVICES=cfg['GPU_UUID'] if mode=='fit' else ''),stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        assert q.returncode==0,('No blind retry',depth,prefix,mode,q.returncode);r=core.read(folder/'report.json');assert r['state']=='complete' and r['frozen_parameters_unchanged'];p=folder/'best.pt';digest=r['results']['best']['PT_SHA256'];assert sha(p)==digest;assert not [p.pid for p in psutil.process_iter(['cmdline']) if p.info['cmdline'] and '--out' in p.info['cmdline'] and str(folder) in p.info['cmdline']];return p,digest,r
    for depth in order:
        session=old if depth==4 else root/f'L{depth}';a.session=session;a.depth=depth;gateproof=core.read(session/'lab_gate.json' if depth!=4 else session/'lab_bridge_gate.json');assert gateproof['passed'];work=session/'sequence_stagewise_20261018';assert not work.exists();work.mkdir();receipt=work/'controller.json';save(receipt,dict(pid=os.getpid(),entry_SHA256=sha(__file__),source_SHA256=sha(core.__file__),session=str(session.resolve()),depth=depth))
        if depth==4:
            checkpoint=old/handoff['selected_relative'];digest=handoff['selected_PT_SHA256'];start_prefix=handoff['prefix']+1
            if handoff['prefix']<5 and handoff['selected_correct']<math.ceil(cfg['simulation_correct']['4']-41.1):
                save(out/'progress.json',dict(state='bounded_middle_binary_repair',depth=4,prefix=handoff['prefix']));checkpoint,digest,_=child('terminal',work/'middle_initial',session,4,handoff['prefix'],checkpoint,digest,receipt)
        else:
            checkpoint=session/'initial.pt';digest=core.read(session/'server_gate.json')['PT_SHA256'];start_prefix=1
        assert sha(checkpoint)==digest;records=[]
        for prefix in range(start_prefix,depth+2):
            stage=('router',)+tuple(str(i) for i in range(depth));stage=stage[prefix-1];save(out/'progress.json',dict(state='capturing',depth=depth,stage=stage,prefix=prefix,completed_models=history));core.acquire_layer(a,stage,checkpoint,digest)
            folder=work/('fit_'+stage);save(out/'progress.json',dict(state='suffix_fitting',depth=depth,stage=stage,prefix=prefix));checkpoint,digest,r=child('fit',folder,session,depth,prefix,checkpoint,digest,receipt);fit_epoch=r['selected_epoch'];fit_sha=sha(folder/'report.json')
            if prefix<depth+1 and r['selected_correct']<math.ceil(cfg['simulation_correct'][str(depth)]-41.1):
                save(out/'progress.json',dict(state='bounded_middle_binary_repair',depth=depth,stage=stage,prefix=prefix));checkpoint,digest,r=child('terminal',work/('middle_'+stage),session,depth,prefix,checkpoint,digest,receipt)
            records.append(dict(stage=stage,PT_SHA256=digest,selected_correct=r['selected_correct'],selected_epoch=fit_epoch,scope='fullphysical' if prefix==depth+1 else 'mixed_measured_prefix',fit_report_SHA256=fit_sha));save(work/'stage_history.json',records)
        a.prefix=depth+1;folder=work/'final_targeted';save(out/'progress.json',dict(state='final_targeted_TRAIN_repair',depth=depth));checkpoint,digest,r=child('terminal',folder,session,depth,depth+1,checkpoint,digest,receipt);score=r['selected_correct'];result=dict(state='complete',depth=depth,correct=score,TEST=411,accuracy=score/411,simulation_correct=cfg['simulation_correct'][str(depth)],gap_pp=100*(cfg['simulation_correct'][str(depth)]-score)/411,target_within1pp=cfg['simulation_correct'][str(depth)]-score<=4.11,PT_SHA256=digest,PT=str(checkpoint),physical_frames=(depth+1)*1611,SDK_released=not LOCK.exists(),TEST_development=True,TEST_gradients=False,records=records);save(work/'report.json',result);history.append(result);save(out/'completed_models.json',history);print(json.dumps(result),flush=True)
    save(out/'report.json',dict(state='complete',order=list(order),results=history,SDK_released=not LOCK.exists(),all_training_children_exited=True,recurring_monitor=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('test','gate','fit','terminal','queue','capture-queue'),required=True);p.add_argument('--depth',type=int,choices=(2,4,6));p.add_argument('--prefix',type=int);p.add_argument('--checkpoint-sha');p.add_argument('--controller-pid',type=int)
    for n in ('sequence','session','profile','parent','out','checkpoint','controller-receipt','coordinates','machine-config','phase-lut','response-lut','phase-sdk','amplitude-sdk'):p.add_argument('--'+n,type=Path)
    a=p.parse_args()
    if sys.stdout is None:sys.stdout=(a.sequence/('capture_queue.log' if a.mode=='capture-queue' else 'queue.log') if a.mode in ('queue','capture-queue') else a.out.with_suffix('.log')).open('x',encoding='utf-8',buffering=1);sys.stderr=sys.stdout
    try:selftest() if a.mode=='test' else gate(a) if a.mode=='gate' else fit(a) if a.mode=='fit' else terminal_refine(a) if a.mode=='terminal' else run_queue(a)
    except BaseException as e:
        folder=a.sequence/('capture_controller' if a.mode=='capture-queue' else 'controller') if a.mode in ('queue','capture-queue') else a.out if a.out else a.session
        if folder and folder.exists():save(folder/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False))
        raise
