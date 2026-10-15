"""Keep the same CCD head; protect already-correct optical TRAIN decisions."""
import argparse,copy,hashlib,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import caltech10_sim_head as prior
from caltech10_sim_head_model import CCDHead,CONTRACT
s=prior.s;HERE=prior.HERE;ROOT=prior.ROOT;resources=prior.resources
P=s.read(HERE/'caltech10_sim_head_augmented_repair_profile.json');RUN=P['run'];CPU=HERE/'runs/smoke'/P['cpu_run']
FILES=('caltech10_sim_head_augmented_repair.py','audit_caltech10_sim_head_augmented_repair.py','caltech10_sim_head_augmented_repair_profile.json','CALTECH10_SIM_HEAD_AUGMENTED_REPAIR_PROTOCOL_20261016.md','caltech10_sim_head_preserve.py','audit_caltech10_sim_head_preserve.py','caltech10_sim_head_preserve_profile.json','CALTECH10_SIM_HEAD_PRESERVE_PROTOCOL_20261016.md','caltech10_sim_head_errors_only.py','audit_caltech10_sim_head_errors_only.py','caltech10_sim_head_errors_only_profile.json','CALTECH10_SIM_HEAD_ERRORS_ONLY_PROTOCOL_20261016.md','caltech10_sim_head_augmented.py','audit_caltech10_sim_head_augmented.py','caltech10_sim_head_augmented_profile.json','CALTECH10_SIM_HEAD_AUGMENTED_PROTOCOL_20261016.md')
def sources():
    z=prior.sources()
    for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return z
def loss_terms(logits,base,y):
    correct=(base.argmax(1)==y).detach();ce=F.cross_entropy(logits,y,reduction='none');kl=(base*(base.clamp_min(1e-8).log()-F.log_softmax(logits,1))).sum(1);return (torch.where(correct,P['correct_CE_weight'],1.)*ce+P['correct_KL_weight']*correct*kl).mean()
def select_key(h):return (h['val']['accuracy'],-h['val']['balanced_nll'])
def augmented_train(out,op,clean):
    values,manifest=prior.data('train');images,labels,ids=values;assert len(labels)==761 and np.array_equal(ids,clean['ids']) and np.array_equal(labels.numpy(),clean['labels']);ck=torch.load(op['path'],map_location='cpu',weights_only=False);assert s.sha(op['path'])==op['sha'] and prior.digest(ck['model'])==op['digest'];model=prior.optical.SubsetMoE(4,.3).cuda();model.load_state_dict(ck['model'],strict=True);model.eval();model.requires_grad_(False);features=[clean['features']];base=[clean['base']];records=[];probes=[]
    for view in (1,2):
        _,theta=s.engine.epoch_plan(view,761,P['augmentation_config']);records.append(dict(view=view,theta_sha256=hashlib.sha256(np.asarray(theta).tobytes()).hexdigest()));vf=[];vp=[]
        with torch.inference_mode():
            for ix in torch.arange(761).split(8):
                x=prior.encode(images[ix].cuda(),torch.as_tensor(theta[ix.numpy()],device='cuda'),100);p,_=model(x);ccd=model.last['3']['intensity'];f=prior.ccd_features(ccd);vf.append(f.cpu().numpy());vp.append(p.cpu().numpy())
                if int(ix[0])==0:probes.append(dict(view=view,ccd=ccd[:2].cpu().numpy(),features=f[:2].cpu().numpy()))
        features.append(np.concatenate(vf));base.append(np.concatenate(vp))
    assert prior.digest(model.state_dict())==op['digest'] and all(not p.requires_grad and p.grad is None for p in model.parameters());del model;torch.cuda.empty_cache();z=dict(features=np.concatenate(features),base=np.concatenate(base),labels=np.tile(clean['labels'],3),ids=np.tile(clean['ids'],3),views=np.repeat(np.arange(3),761));np.savez_compressed(out/'augmented_train.npz',**z)
    for q in probes:np.savez_compressed(out/f"view{q['view']}_probe.npz",ccd=q['ccd'],features=q['features'])
    s.save(out/'augmented_train_receipt.json',dict(sources=sources(),cache_sha256=s.sha(out/'augmented_train.npz'),optical_checkpoint_sha256=op['sha'],optical_model_digest=op['digest'],weights_frozen=True,optical_optimizer_steps=0,views=records,training_examples=2283,unique_TRAIN_ids=761,val_test_images_read=0,contract=CONTRACT),exclusive=True);return z

def preserve_failure():
    for path,expected in P['preserved_failure_sha256'].items():assert s.sha(path)==expected
    old=Path(next(iter(P['preserved_failure_sha256']))).parent;assert not list(old.rglob('fit_result.json')) and not list(old.rglob('augmented_train.npz'))

def cpu(a):
    torch.set_num_threads(4);assert a.out==CPU and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not a.out.exists() and resources.slots()['total']==0
    for case,c in P['cases'].items():
        assert s.sha(c['head_path'])==c['head_sha256'];ck=torch.load(c['head_path'],map_location='cpu',weights_only=False);assert prior.digest(ck['head_state'])==c['head_digest'] and ck['epoch']==3 and s.sha(P['optical_parents'][case]['path'])==P['optical_parents'][case]['sha']
    preserve_failure();cfg=P['augmentation_config'];assert cfg['seed']==17==torch.load(P['optical_parents']['L4_rho03']['path'],map_location='cpu',weights_only=False)['config']['seed']
    for view in (1,2):
        order,theta=s.engine.epoch_plan(view,761,cfg);rng=np.random.default_rng(np.random.SeedSequence([17,view,20261013]));reforder=rng.permutation(761).astype('<i8');angle=np.deg2rad(rng.uniform(-20,20,761));scale=rng.uniform(.9,1.1,761);ref=np.zeros((761,2,3),np.float32);ref[:,0,0]=np.cos(angle)/scale;ref[:,1,1]=ref[:,0,0];ref[:,0,1]=-np.sin(angle)/scale;ref[:,1,0]=np.sin(angle)/scale;ref[:,:,2]=rng.uniform(-5,5,(761,2))*2/150;assert np.array_equal(order,reforder) and np.array_equal(theta,ref)
    try:s.engine.epoch_plan(1,761,{})
    except KeyError:pass
    else:raise AssertionError('Missing seed must remain rejected')
    torch.manual_seed(23);z=torch.randn(5,10,dtype=torch.float64,requires_grad=True);b=torch.softmax(torch.randn(5,10,dtype=torch.float64),1);y=torch.tensor([int(b[0].argmax()),(int(b[1].argmax())+1)%10,2,3,4]);correct=b.argmax(1)==y;loss=loss_terms(z,b,y);g=torch.autograd.grad(loss,z)[0];p=z.softmax(1);one=F.one_hot(y,10);w=torch.where(correct,P['correct_CE_weight'],1.);reference=(w[:,None]*(p-one)+P['correct_KL_weight']*correct[:,None]*(p-b))/len(y);assert torch.allclose(g,reference,atol=1e-12,rtol=0)
    src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();s.engine.verify_sources(src,commit);s.save(a.out.with_suffix('.claim.json'),dict(sources=src,automatic_retry=False),exclusive=True);a.out.mkdir();s.save(a.out/'checks.json',dict(passed=True,sources=src,git_commit=commit,analytic_preservation_gradient=True,head_parameters=8554,independent_augmentation_seed_replay=True,preserved_failure_sha256=P['preserved_failure_sha256'],real_optical_models=0,real_image_forward_calls=0,test_images=0,optimizer_steps=0),exclusive=True)
def run_case(a):
    preserve_failure();c=P['cases'][a.case];oldroot=Path(P['cache_root'])/a.case;assert os.environ.get('CUDA_VISIBLE_DEVICES')==a.gpu and a.gpu in resources.GPU.values();s.gpu_budget.ALLOWED.update(resources.GPU.values());s.gpu_budget.configure(torch,a.gpu,shared=True,memory_gib=2);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest');proof=s.read(CPU/'checks.json');assert proof['passed'] and proof['sources']==sources();s.engine.verify_sources(sources(),subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());out=a.out;assert not out.exists();out.mkdir();s.save(out/'metadata.json',dict(pid=os.getpid(),sources=sources(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),profile=P,case=a.case),exclusive=True)
    cache={role:dict(np.load(oldroot/('cache_'+role+'.npz'),allow_pickle=False)) for role in ('train','val','test')}
    for role in cache:assert s.read((oldroot/('cache_'+role+'.npz')).with_suffix('.receipt.json'))['sources']==prior.sources()
    clean_train=cache['train'];augmented=augmented_train(out,P['optical_parents'][a.case],clean_train);cache['train']=augmented
    tensors={role:(torch.from_numpy(z['features']).cuda(),torch.from_numpy(z['base']).cuda(),torch.from_numpy(z['labels']).long().cuda()) for role,z in cache.items()};assert s.sha(c['head_path'])==c['head_sha256'];parent=torch.load(c['head_path'],map_location='cpu',weights_only=False);head=CCDHead().cuda();head.load_state_dict(parent['head_state'],strict=True);assert prior.digest(head.state_dict())==c['head_digest'];opt=torch.optim.AdamW(head.parameters(),lr=P['learning_rate'],weight_decay=.001)
    # Actual cached TRAIN gradient equivalence for the new loss, no optimizer update.
    x,b,y=tensors['train'];idx=torch.cat([torch.where(y==k)[0][:1] for k in range(10)]+[torch.arange(6,device='cuda')]);grads=[]
    for micro in (1,8):
        opt.zero_grad(set_to_none=True)
        for start in range(0,len(idx),micro):q=idx[start:start+micro];(loss_terms(head.from_features(x[q],b[q]),b[q],y[q])*len(q)/len(idx)).backward()
        grads.append([p.grad.detach().clone() for p in head.parameters()]);assert all(torch.isfinite(p.grad).all() and p.grad.norm()>0 for p in head.parameters())
    error=max(float((p-q).norm()/(p.norm()+1e-12)) for p,q in zip(*grads));assert error<.02 and prior.digest(head.state_dict())==c['head_digest'];opt.zero_grad(set_to_none=True);s.save(out/'numeric_gate.json',dict(passed=True,micro1vs8_max_relative_gradient=error,optimizer_steps=0,parent_head_state_preserved=True,head_parameters=8554),exclusive=True)
    clean_x=torch.from_numpy(clean_train['features']).cuda();clean_b=torch.from_numpy(clean_train['base']).cuda();history=[dict(head_epoch=3,phase_epoch=0,train=parent['train'],val=parent['validation'])];best=parent;bestkey=select_key(history[0]);best_state=parent['head_state'];best_epoch=3;phase=0;no_val_peak=0;train_mark=parent['train']['accuracy'];train_stale=0;known={q.name:q for q in (oldroot/'once_tests').iterdir() if (q/'result.json').exists()};[known.update({q.name:q for q in Path(folder).iterdir() if (q/'result.json').exists()}) for folder in P['known_tests_roots'] if Path(folder).exists()];control=P['controls'][a.case];baseline=P['targets'][str(P['optical_parents'][a.case]['depth'])]
    while True:
        segment=phase//3+1;dest=out/f'segment{segment:04d}';dest.mkdir();steps=[]
        for _ in range(3):
            phase+=1;epoch=phase+3;head.train();order=torch.randperm(len(y),generator=torch.Generator().manual_seed(17+epoch));step_logits=[];step_losses=[]
            for ids in order.split(64):
                opt.zero_grad(set_to_none=True);logits=head.from_features(x[ids],b[ids]);loss=loss_terms(logits,b[ids],y[ids]);assert torch.isfinite(loss);step_logits.append(logits.detach().cpu().numpy());step_losses.append(float(loss.detach()));loss.backward();assert all(torch.isfinite(p.grad).all() for p in head.parameters());torch.nn.utils.clip_grad_norm_(head.parameters(),5.);opt.step()
            np.savez_compressed(dest/f'loss_epoch{epoch:04d}.npz',logits=np.concatenate(step_logits),loss=np.array(step_losses),order=order.numpy());head.eval();tm=prior.metrics(clean_train['labels'],prior.probability(head,clean_x,clean_b));vx,vb,_=tensors['val'];vm=prior.metrics(cache['val']['labels'],prior.probability(head,vx,vb));row=dict(head_epoch=epoch,phase_epoch=phase,train=tm,val=vm);history.append(row);no_val_peak+=1;train_stale+=1
            if vm['accuracy']>max(h['val']['accuracy'] for h in history[:-1])+1e-12:no_val_peak=0
            if tm['accuracy']>train_mark+.005:train_mark=tm['accuracy'];train_stale=0
            ck=dict(epoch=epoch,phase_epoch=phase,head_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()},optimizer=copy.deepcopy(opt.state_dict()),train=tm,validation=vm,optical_checkpoint_path=P['optical_parents'][a.case]['path'],optical_checkpoint_sha256=P['optical_parents'][a.case]['sha'],head_parameters=8554,contract=CONTRACT,parent_head_sha256=c['head_sha256'],head_optimizer_reset_for_new_loss=True,sources=sources())
            if select_key(row)>bestkey:bestkey=select_key(row);best=ck;best_state=ck['head_state'];best_epoch=epoch
            s.save(out/'progress.json',dict(state='head_training',phase_epoch=phase,total_head_epoch=epoch,train_accuracy=tm['accuracy'],val_accuracy=vm['accuracy'],val_balanced_nll=vm['balanced_nll']))
        torch.save(best,dest/'best_checkpoint.pt');torch.save(ck,dest/'last_checkpoint.pt');s.save(dest/'history.json',history,exclusive=True);current=copy.deepcopy(head.state_dict());head.load_state_dict(best_state,strict=True);hd=prior.digest(head.state_dict());td=known.get(hd)
        if td is None:
            td=out/'once_tests'/hd;td.mkdir(parents=True);fx,fb,_=tensors['test'];prob=prior.probability(head,fx,fb);test=prior.metrics(cache['test']['labels'],prob);np.savez_compressed(td/'predictions.npz',probability=prob,labels=cache['test']['labels'],ids=cache['test']['ids']);s.save(td/'result.json',dict(test=test,head_digest=hd,selected_head_epoch=best_epoch,prediction_sha256=s.sha(td/'predictions.npz'),test_informed_development=True),exclusive=True);known[hd]=td
        test=s.read(td/'result.json')['test'];gap=100*(test['accuracy']-control['test']['accuracy']);passed=test['accuracy']>=baseline['residual_accuracy']-1e-12 and gap>=baseline['gap_pp']-1e-12;review=phase>=9 and no_val_peak>=6 and train_stale>=6;record=dict(case=a.case,actual_head_epochs=phase+3,new_phase_epochs=phase,selected_head_epoch=best_epoch,train=best['train'],validation=best['validation'],test=test,test_gap_pp=gap,control=control,target=baseline,target_achieved=passed,state='target_achieved' if passed else 'review_required' if review else 'head_training',head_parameters=8554,contract=CONTRACT,head_digest=hd,test_path=str(td),parent_head_sha256=c['head_sha256'],optical_checkpoint_sha256=P['optical_parents'][a.case]['sha'],optical_weights_frozen=True,optical_optimizer_steps=0,head_optimizer_state_steps=36*phase,head_optimizer_reset_for_loss=True,selection='first highest validation accuracy, ties minimum validation balancedNLL; no test epoch selection')
        assert s.sha(P['optical_parents'][a.case]['path'])==P['optical_parents'][a.case]['sha'];s.save(dest/'fit_result.json',record,exclusive=True);s.save(dest/'audit_claim.json',dict(sources=sources(),automatic_retry=False),exclusive=True);log=(dest/'audit.log').open('x');rc=subprocess.run([sys.executable,str(HERE/'audit_caltech10_sim_head_augmented_repair.py'),'--out',str(dest)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT).returncode;log.close();assert rc==0 and s.read(dest/'audit.json')['passed'];s.save(dest/'audit_completed.json',dict(passed=True,actual_audit_returncode=rc,audit_sha256=s.sha(dest/'audit.json')),exclusive=True);s.save(out/'comparison.json',record)
        if passed or review:s.save(out/'completed.json',record,exclusive=True);return
        head.load_state_dict(current,strict=True)
def campaign(a):
    assert a.out==HERE/'runs/simulation'/RUN and not a.out.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and resources.slots()['total']==0;proof=s.read(CPU/'checks.json');assert proof['passed'] and proof['sources']==sources();s.engine.verify_sources(sources(),subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());s.save(a.out.with_suffix('.launch_claim.json'),dict(sources=sources(),automatic_retry=False),exclusive=True);a.out.mkdir();s.save(a.out/'metadata.json',dict(pid=os.getpid(),sources=sources(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),profile=P),exclusive=True);jobs=[]
    for case,gpuindex in (('L4_rho03',4),):
        gpu=resources.GPU[gpuindex];s.gpu_budget.ALLOWED.update(resources.GPU.values());s.gpu_budget.preflight(gpu,shared=True,memory_gib=2);log=(a.out/(case+'.log')).open('x');child=subprocess.Popen([sys.executable,__file__,'--phase','run_case','--case',case,'--out',str(a.out/case),'--gpu',gpu],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT);jobs.append((case,child,log))
    s.save(a.out/'status.json',dict(state='preservation_head_training',workers=[dict(case=c,pid=p.pid) for c,p,_ in jobs]))
    for case,p,log in jobs:rc=p.wait();log.close();s.save(a.out/(case+'_exit.json'),dict(pid=p.pid,returncode=rc,actual_exit=prior.optical.base.exited(p.pid)),exclusive=True);assert rc==0
    s.save(a.out/'completed.json',dict(state='complete_targets_or_review',cases={c:s.read(a.out/c/'completed.json') for c,_,_ in jobs}),exclusive=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['cpu','run_case','campaign'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--case');p.add_argument('--gpu');a=p.parse_args()
    try:globals()[a.phase](a)
    except BaseException as error:
        mark=a.out/'failure.json' if a.out.exists() else a.out.with_suffix('.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(error),automatic_retry=False),exclusive=True)
        raise
