"""Freeze accepted optical weights, train the SAME 8554-parameter CCD head in sim."""
import argparse,copy,hashlib,json,math,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import caltech10_error_distill as previous
import caltech10_gap as optical
from resisc20_input_encoder import encode
from caltech10_sim_head_model import CCDHead,ccd_features,CONTRACT
s=previous.s;HERE=previous.HERE;ROOT=previous.ROOT;resources=previous.resources
FILES=('caltech10_sim_head.py','caltech10_sim_head_model.py','audit_caltech10_sim_head.py','caltech10_sim_head_profile.json','CALTECH10_SIM_HEAD_PROTOCOL_20261016.md')
P=s.read(HERE/'caltech10_sim_head_profile.json');RUN=P['run'];CPU=HERE/'runs/smoke'/P['cpu_run']
def sources():
    z=previous.sources()
    for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return z
def digest(state):return s.digest(state)
def claim(path,**data):s.save(path,dict(sources=sources(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),automatic_retry=False,command=sys.argv,**data),exclusive=True)
def metrics(y,p):
    y=np.asarray(y);p=np.asarray(p,np.float64);assert p.shape==(len(y),10) and np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-5)
    cm=np.zeros((10,10),np.int64);np.add.at(cm,(y,p.argmax(1)),1);support=cm.sum(1);assert (support>0).all();nll=-np.log(p[np.arange(len(y)),y].clip(1e-12));return dict(n=len(y),accuracy=float(cm.trace()/len(y)),balanced_accuracy=float((cm.diagonal()/support).mean()),recall=(cm.diagonal()/support).tolist(),support=support.tolist(),confusion_matrix=cm.tolist(),nll=float(nll.mean()),balanced_nll=float(np.mean([nll[y==k].mean() for k in range(10)])))
def load_parent(case):
    c=P['parents'][case];assert s.sha(c['path'])==c['sha'];ck=torch.load(c['path'],map_location='cpu',weights_only=False);assert digest(ck['model'])==c['digest'] and ck['epoch']==c['optical_epoch'];return c,ck
def cpu(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and a.out==CPU and not a.out.exists() and not a.out.with_suffix('.claim.json').exists();assert resources.slots()['total']==0;torch.set_num_threads(4)
    for case in P['parents']:load_parent(case)
    torch.manual_seed(17);h=CCDHead();initial=digest(h.net.state_dict());base=torch.softmax(torch.randn(3,10),1);x=torch.rand(3,478,478)
    with torch.no_grad():assert torch.allclose(h(x,base).softmax(1),base,atol=1e-7)
    # Independent NumPy reconstruction of adaptive pooling, including non-divisible bins.
    raw=x.numpy().astype(np.float64);z=np.log1p(raw/np.maximum(raw.mean((-2,-1),keepdims=True),1e-6));pooled=np.stack([z[:,int(np.floor(i*478/16)):int(np.ceil((i+1)*478/16)),int(np.floor(j*478/16)):int(np.ceil((j+1)*478/16))].mean((1,2)) for i in range(16) for j in range(16)],1);error=float(np.max(np.abs(pooled-ccd_features(x).numpy())));assert error<2e-5
    loss=F.cross_entropy(h(x,base),torch.tensor([0,1,2]));loss.backward();assert all(torch.isfinite(p.grad).all() for p in h.parameters()) and h.net[-1].weight.grad.norm()>0 and h.net[0].weight.grad.norm()==0
    src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();s.engine.verify_sources(src,commit);claim(a.out.with_suffix('.claim.json'),phase='head_CPU');a.out.mkdir();s.save(a.out/'checks.json',dict(passed=True,sources=src,git_commit=commit,head_parameters=8554,contract=CONTRACT,initial_head_net_digest=initial,pooling_numpy_max_error=error,zero_output_preserves_optical_probability=True,zero_first_layer_gradient_expected_at_initialization=True,real_optical_models=0,real_images=0,test_images=0,optimizer_steps=0),exclusive=True)
def data(role):
    cfg=next(iter(P['parents'].values()))['config'];path=Path(cfg['data']);m=optical.manifest(path)
    if role!='test':return optical.load_fit(path,role,m),m
    with np.load(path,allow_pickle=False) as f:x,y,ids=[f['test_'+n].copy() for n in ('images','labels','ids')]
    keep=np.isin(y,optical.INDICES);x=x[keep];y=optical.subset_labels(y[keep]);ids=ids[keep];assert ids.tolist()==m['split_ids']['test'] and np.bincount(y,minlength=10).tolist()==m['supports']['test'];return (torch.from_numpy(x.transpose(0,3,1,2).copy()),torch.from_numpy(y).long(),ids),m
def cache(root,case,role,model):
    path=root/case/('cache_'+role+'.npz')
    if path.exists():return dict(np.load(path,allow_pickle=False))
    assert model is not None;values,m=data(role);x,y,ids=values;features=[];base=[];probes=[]
    with torch.inference_mode():
        for start in range(0,len(y),8):
            p,_=model(encode(x[start:start+8].cuda(),None,100));ccd=model.last[str(model.depth-1)]['intensity'];f=ccd_features(ccd);features.append(f.cpu().numpy());base.append(p.cpu().numpy())
            if start==0:probes=ccd[:2].cpu().numpy()
            s.save(root/case/'progress.json',dict(state='frozen_optical_feature_cache',role=role,completed=min(start+8,len(y)),total=len(y),optical_weights_frozen=True))
    assert all(p.grad is None and not p.requires_grad for p in model.parameters());z=dict(features=np.concatenate(features),base=np.concatenate(base),labels=y.numpy(),ids=np.asarray(ids),probe_ccd=probes);np.savez_compressed(path,**z);s.save(path.with_suffix('.receipt.json'),dict(case=case,role=role,sources=sources(),cache_sha256=s.sha(path),optical_checkpoint_sha256=P['parents'][case]['sha'],optical_model_digest=digest(model.state_dict()),samples=len(y),contract=CONTRACT),exclusive=True);return z
def probability(head,features,base):
    with torch.inference_mode():return head.from_features(features,base).softmax(1).cpu().numpy()
def arm(a):
    root=a.out;case=a.case;c,ck=load_parent(case);assert a.gpu in resources.GPU.values() and os.environ.get('CUDA_VISIBLE_DEVICES')==a.gpu;resources.slots();s.gpu_budget.ALLOWED.update(resources.GPU.values());s.gpu_budget.configure(torch,a.gpu,shared=True,memory_gib=2);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    proof=s.read(CPU/'checks.json');assert proof['passed'] and proof['sources']==sources();s.engine.verify_sources(sources(),subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());folder=root/case;folder.mkdir(exist_ok=True);out=folder/f'segment{a.segment:04d}';assert not out.exists();out.mkdir();claim(out/'claim.json',case=case,segment=a.segment)
    model=None
    if not all((folder/('cache_'+role+'.npz')).exists() for role in ('train','val','test')):
        model=optical.SubsetMoE(c['depth'],c['rho']).cuda();model.load_state_dict(ck['model'],strict=True);model.eval();model.requires_grad_(False);assert sum(p.numel() for p in model.parameters())=={2:479364,3:680068,4:908552}[c['depth']]
    train=cache(root,case,'train',model);val=cache(root,case,'val',model)
    tx=torch.from_numpy(train['features']).cuda();tb=torch.from_numpy(train['base']).cuda();ty=torch.from_numpy(train['labels']).long().cuda();vx=torch.from_numpy(val['features']).cuda();vb=torch.from_numpy(val['base']).cuda()
    torch.manual_seed(17);head=CCDHead().cuda();initial=digest(head.net.state_dict());assert initial==proof['initial_head_net_digest'];head.mean.copy_(torch.from_numpy(train['features'].mean(0,dtype=np.float64).astype(np.float32)).cuda());head.std.copy_(torch.from_numpy(train['features'].std(0,dtype=np.float64).astype(np.float32).clip(.001)).cuda());opt=torch.optim.AdamW(head.parameters(),lr=P['learning_rate'],weight_decay=P['weight_decay']);assert sum(p.numel() for group in opt.param_groups for p in group['params'])==8554
    history=[];best=None;best_nll=float('inf');best_state=None;offset=0
    if a.segment>1:
        prev=folder/f'segment{a.segment-1:04d}';last=torch.load(prev/'last_checkpoint.pt',map_location='cpu',weights_only=False);head.load_state_dict(last['head_state'],strict=True);opt.load_state_dict(last['optimizer']);offset=last['epoch'];history=s.read(prev/'history.json');best=torch.load(prev/'best_checkpoint.pt',map_location='cpu',weights_only=False);best_nll=best['validation']['balanced_nll'];best_state=best['head_state']
    # Actual restored optical features, micro1 vs macro8 gradient proof; zero last layer must stay zero before real fitting.
    if a.segment==1:
        original=copy.deepcopy(head.state_dict());gen=torch.Generator(device='cuda').manual_seed(23)
        with torch.no_grad():head.net[-1].weight.copy_(torch.randn(head.net[-1].weight.shape,generator=gen,device='cuda')*.01)
        grads=[]
        for micro in (1,8):
            opt.zero_grad(set_to_none=True)
            for k in range(0,16,micro):(F.cross_entropy(head.from_features(tx[k:k+micro],tb[k:k+micro]),ty[k:k+micro])*len(ty[k:k+micro])/16).backward()
            grads.append([p.grad.detach().clone() for p in head.parameters()]);assert all(torch.isfinite(p.grad).all() and p.grad.norm()>0 for p in head.parameters())
        rel=max(float((g-h).norm()/(g.norm()+1e-12)) for g,h in zip(*grads));assert rel<.02;head.load_state_dict(original,strict=True);opt.zero_grad(set_to_none=True);s.save(out/'numeric_gate.json',dict(passed=True,gradient_relative_max=rel,optimizer_steps=0,initial_head_net_digest=initial,optical_parameter_gradients_none=True,head_parameters=8554,contract=CONTRACT),exclusive=True)
    started=time.monotonic()
    for epoch in range(offset+1,offset+P['segment_epochs']+1):
        assert time.monotonic()-started<1800;head.train();order=torch.randperm(len(ty),generator=torch.Generator().manual_seed(17+epoch));oh=hashlib.sha256(order.numpy().astype('<i8').tobytes()).hexdigest()
        for ix in order.split(64):opt.zero_grad(set_to_none=True);loss=F.cross_entropy(head.from_features(tx[ix],tb[ix]),ty[ix]);assert torch.isfinite(loss);loss.backward();assert all(torch.isfinite(p.grad).all() for p in head.parameters());torch.nn.utils.clip_grad_norm_(head.parameters(),5.);opt.step()
        head.eval();tm=metrics(train['labels'],probability(head,tx,tb));vm=metrics(val['labels'],probability(head,vx,vb));history.append(dict(epoch=epoch,train=tm,val=vm,order_sha256=oh,optimizer_steps=12*epoch));state={k:q.detach().cpu().clone() for k,q in head.state_dict().items()};checkpoint=dict(epoch=epoch,head_state=state,optimizer=copy.deepcopy(opt.state_dict()),train=tm,validation=vm,optical_checkpoint_path=c['path'],optical_checkpoint_sha256=c['sha'],optical_model_digest=c['digest'],head_parameters=8554,contract=CONTRACT,initial_head_net_digest=initial,normalization_fit='TRAIN only',sources=sources())
        if vm['balanced_nll']<best_nll:best_nll=vm['balanced_nll'];best_state=state;best=checkpoint
        s.save(folder/'progress.json',dict(state='head_training',head_epoch=epoch,train_accuracy=tm['accuracy'],val_accuracy=vm['accuracy'],val_balanced_nll=vm['balanced_nll'],optical_weights_frozen=True))
    assert best is not None;torch.save(best,out/'best_checkpoint.pt');torch.save(checkpoint,out/'last_checkpoint.pt');s.save(out/'history.json',history,exclusive=True);head.load_state_dict(best_state,strict=True);head.eval();hd=digest(head.state_dict());test=cache(root,case,'test',model);td=folder/'once_tests'/hd
    if td.exists():test_metric=s.read(td/'result.json')['test'];test_reused=True
    else:
        td.mkdir(parents=True);claim(td/'claim.json',head_digest=hd);prob=probability(head,torch.from_numpy(test['features']).cuda(),torch.from_numpy(test['base']).cuda());test_metric=metrics(test['labels'],prob);np.savez_compressed(td/'predictions.npz',probability=prob,labels=test['labels'],ids=test['ids']);s.save(td/'result.json',dict(test=test_metric,head_digest=hd,selected_head_epoch=best['epoch'],prediction_sha256=s.sha(td/'predictions.npz'),test_informed_development=True),exclusive=True);test_reused=False
    assert s.sha(c['path'])==c['sha'] and (model is None or (digest(model.state_dict())==c['digest'] and all(p.grad is None for p in model.parameters())));s.save(out/'fit_result.json',dict(case=case,actual_head_epochs=checkpoint['epoch'],selected_head_epoch=best['epoch'],train=best['train'],validation=best['validation'],test=test_metric,test_path=str(td),test_reused=test_reused,head_digest=hd,initial_head_net_digest=initial,optical_checkpoint_sha256=c['sha'],optical_model_digest=c['digest'],optical_weights_unchanged=True,optical_optimizer_steps=0,head_optimizer_steps=12*checkpoint['epoch'],head_parameters=8554,head_contract=CONTRACT,sources=sources(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),fit_seconds=time.monotonic()-started),exclusive=True)
def campaign(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and a.out==HERE/'runs/simulation'/RUN and not a.out.exists();proof=s.read(CPU/'checks.json');assert proof['passed'] and proof['sources']==sources();assert resources.slots()['total']==0;claim(a.out.with_suffix('.launch_claim.json'),phase='sim_head_campaign');a.out.mkdir();claim(a.out/'metadata.json',pid=os.getpid(),profile=P);remaining={2,3,4};segment=0;control={};comparisons={}
    while remaining:
        segment+=1;pending=[f'L{d}_rho{rho}' for d in sorted(remaining,reverse=True) for rho in (('03','0') if segment==1 else ('03',))];active=[]
        while pending or active:
            for job in list(active):
                if job['process'].poll() is None:continue
                rc=job['process'].returncode;job['log'].close();out=a.out/job['case']/f'segment{segment:04d}';s.save(out/'producer_exit.json',dict(pid=job['process'].pid,returncode=rc,actual_exit=optical.base.exited(job['process'].pid)),exclusive=True);assert rc==0,'Producer failed; no retry';claim(out/'audit_claim.json',phase='independent_head_audit');log=(out/'audit.log').open('x');ac=subprocess.run([sys.executable,str(HERE/'audit_caltech10_sim_head.py'),'--out',str(out)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT).returncode;log.close();assert ac==0,'Own independent audit failed; preserve and stop';audit=out/'audit.json';assert s.read(audit)['passed'];s.save(out/'audit_completed.json',dict(passed=True,actual_audit_returncode=ac,audit_sha256=s.sha(audit)),exclusive=True);active.remove(job)
            for index in (2,4,5,0):
                if not pending:break
                gpu=resources.GPU[index];live=resources.slots()
                if live['total']>=4 or live['counts'][gpu] or any(j['gpu']==gpu for j in active):continue
                s.gpu_budget.ALLOWED.update(resources.GPU.values())
                try:s.gpu_budget.preflight(gpu,shared=True,memory_gib=2)
                except AssertionError:continue
                case=pending.pop(0);log=(a.out/(case+f'_segment{segment:04d}.log')).open('x');process=subprocess.Popen([sys.executable,__file__,'--phase','arm','--case',case,'--segment',str(segment),'--gpu',gpu,'--out',str(a.out)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT);active.append(dict(case=case,gpu=gpu,process=process,log=log))
            s.save(a.out/'status.json',dict(state='sim_head_training',segment=segment,active=[dict(case=j['case'],gpu=j['gpu'],pid=j['process'].pid) for j in active],pending=pending,completed_depths=comparisons,optical_weights_frozen=True))
            if pending or active:time.sleep(2)
        for d in list(remaining):
            if segment==1:control[d]=s.read(a.out/f'L{d}_rho0'/f'segment{segment:04d}'/'fit_result.json')
            res=s.read(a.out/f'L{d}_rho03'/f'segment{segment:04d}'/'fit_result.json');gap=100*(res['test']['accuracy']-control[d]['test']['accuracy']);floor=P['targets'][str(d)];passed=res['test']['accuracy']>=floor['residual_accuracy']-1e-12 and gap>=floor['gap_pp']-1e-12;hist=s.read(a.out/f'L{d}_rho03'/f'segment{segment:04d}'/'history.json');best=min(hist,key=lambda z:z['val']['balanced_nll']);plateau=res['actual_head_epochs']>=15 and res['actual_head_epochs']-best['epoch']>=9
            comparison=dict(depth=d,residual=res,control=control[d],test_gap_pp=gap,target=floor,target_achieved=passed,state='target_achieved' if passed else 'review_required' if plateau else 'continuing_residual_head_only',test_informed_development=True);comparisons[str(d)]=comparison;s.save(a.out/f'L{d}_comparison.json',comparison)
            if passed or plateau:remaining.remove(d)
        s.save(a.out/'status.json',dict(state='sim_head_training' if remaining else 'complete_targets_or_review',segment=segment,remaining_depths=sorted(remaining),comparisons=comparisons))
    s.save(a.out/'completed.json',dict(state='complete_targets_or_review',comparisons=comparisons,sources=sources()),exclusive=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['cpu','campaign','arm'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--case');p.add_argument('--gpu');p.add_argument('--segment',type=int,default=1);a=p.parse_args()
    try:globals()[a.phase](a)
    except BaseException as error:
        mark=(a.out/a.case/f'segment{a.segment:04d}'/'failure.json') if a.phase=='arm' else a.out.with_suffix('.failure.json');mark.parent.mkdir(exist_ok=True,parents=True)
        if not mark.exists():s.save(mark,dict(error=repr(error),automatic_retry=False),exclusive=True)
        raise
