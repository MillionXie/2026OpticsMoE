"""Finite original optical suffix fit with original TRAIN teacher geometry.

Measured upstream remains frozen. Teacher supervision is TRAIN-only; full CPU
TEST selects a development checkpoint, never provides a derivative. New masks
are MIXED diagnostics until every affected downstream CCD is freshly acquired.
"""
import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import lab_caltech20_residual_sequence_20261018 as seq
from lab_caltech20_rich_model_20261018 import load_checkpoint,DIRECT,sha
from lab_caltech10_20261015 import save


def teacher_loss(student_logits,student_features,teacher_p,teacher_features,y):
    """No TEST input; retain geometry only where original TRAIN teacher is right."""
    assert not teacher_p.requires_grad and not teacher_features.requires_grad
    keep=(teacher_p.argmax(1)==y).to(student_logits.dtype)
    target=teacher_p.clamp_min(1e-12).sqrt();target=target/target.sum(1,keepdim=True)
    kl=F.kl_div((student_logits/2).log_softmax(1),target,reduction='none').sum(1)*4
    geometry=(student_features-teacher_features).square().mean(1)
    return ((.2*kl+.5*geometry)*keep).mean()


def selftest():
    torch.set_num_threads(2);torch.manual_seed(73)
    z=torch.randn(8,20,dtype=torch.float64,requires_grad=True);f=torch.randn(8,256,dtype=torch.float64,requires_grad=True);p=torch.randn(8,20,dtype=torch.float64).softmax(1);t=torch.randn(8,256,dtype=torch.float64);y=p.argmax(1);loss=teacher_loss(z,f,p,t,y);gz,gf=torch.autograd.grad(loss,(z,f));assert torch.isfinite(gz).all() and torch.isfinite(gf).all() and gf.abs().sum()>0
    eps=1e-5;zz=z.detach().clone();zz[0,0]+=eps;l1=teacher_loss(zz,f.detach(),p,t,y);zz[0,0]-=2*eps;l0=teacher_loss(zz,f.detach(),p,t,y);assert abs(float((l1-l0)/(2*eps)-gz[0,0]))<1e-8
    for depth,prefix in ((2,2),(6,4)):
        model=seq.core.MeasuredPrefix(depth).eval();x=torch.rand(2,224,224)
        with torch.no_grad():q=model(x)[0];fields={k:v['intensity'].clone() for k,v in model.last.items()}
        model.measured={k:fields[k] for k in ('router',)+tuple(str(k) for k in range(prefix-1))};model.requires_grad_(False);allowed=seq.core.allowed_keys(model,prefix)
        for n,v in model.named_parameters():v.requires_grad_(n in allowed)
        held={};hook=model.linear.register_forward_pre_hook(lambda _,args:held.update(features=args[0]));p2=model(x)[0];assert torch.equal(q,p2);loss=F.nll_loss(p2.clamp_min(1e-12).log(),p2.argmax(1))+teacher_loss(p2.clamp_min(1e-12).log(),held['features'],q.detach(),held['features'].detach()+.1,p2.argmax(1));loss.backward();hook.remove()
        assert all(v.grad is None for n,v in model.named_parameters() if n not in allowed)
        assert all(v.grad is not None and torch.isfinite(v.grad).all() for n,v in model.named_parameters() if n in allowed)
    print(json.dumps(dict(passed=True,exact_original_prefix_forward=True,TRAIN_teacher_derivative_checked=True,frozen_upstream=True,SDK_opened=False,GPU_used=False)))


def run(a):
    assert a.depth==2 and a.prefix==2 and not a.out.exists() and not seq.LOCK.exists()
    cfg=json.loads(a.profile.read_text());assert cfg['mask']['seconds']==600 and cfg['mask']['epochs']==80 and cfg['TRAIN']==1200 and cfg['TEST']==411 and cfg['VALID']==0
    assert not torch.cuda.is_initialized();os.environ['CUDA_VISIBLE_DEVICES']=cfg['GPU_UUID'];torch.set_num_threads(2);torch.manual_seed(cfg['seed'])
    admission=json.loads(a.admission.read_text());assert admission['state']=='admitted' and admission['parent_PT_SHA256']==a.checkpoint_sha and admission['TRAIN60_canonical_native_phase_passed']
    _,data,dm,model,original,canonical,fields,identities=seq.dataset_model(a);assert identities==admission['prefix_SHA256'] and list(fields)==['router','0'];assert sha(a.session/'source_reference.ts')==cfg['canonical_reference_SHA256'];assert sha(a.checkpoint)==a.checkpoint_sha
    baseline,bp,_=seq.assess(model,data,fields,canonical,'test',4);saved=np.load(Path(admission['parent_report']).parent/'best_TEST_predictions.npz');assert np.array_equal(saved['ids'],data['test'][2]) and np.array_equal(saved['labels'],data['test'][1]);assert np.max(np.abs(bp-saved['probabilities']))<=3e-6 and round(baseline['accuracy']*411)==283
    gate=json.loads((a.session/'server_gate.json').read_text());teacher,_=load_checkpoint(a.session/'initial.pt',gate['PT_SHA256'],2,.3,alpha_schema=DIRECT);teacher.propagator.transfer.copy_(canonical);teacher.requires_grad_(False);teacher_state=seq.core.state_cpu(teacher)
    assert 'A100' not in torch.cuda.get_device_name(0);names=seq.core.windows_process_names()
    for row in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True).splitlines():
        gpu,pid,name=row.split(',',2);pid=int(pid)
        if gpu.strip()==cfg['GPU_UUID']:assert seq.core.gpu_process_allowed(pid,names.get(pid,Path(name.strip()).name.lower()),os.getpid(),os.name,cfg['desktop_graphics_process_allowlist']),('GPU occupied; no sharing or killing',pid)
    torch.cuda.set_per_process_memory_fraction(.7,0);model.to('cuda');teacher.to('cuda').eval();a.out.mkdir();features=[];probabilities=[];held_teacher={};handle=teacher.linear.register_forward_pre_hook(lambda _,args:held_teacher.update(features=args[0].detach()))
    with torch.inference_mode():
        for begin in range(0,1200,4):
            p=teacher(seq.encode_images(data['train'][0][begin:begin+4].cuda()))[0];features.append(held_teacher['features'].cpu());probabilities.append(p.cpu());save(a.out/'progress.json',dict(state='original_TRAIN_teacher_cache',completed=min(begin+4,1200),total=1200,TEST_teacher_used=False))
    handle.remove();teacher.cpu();assert all(torch.equal(v,teacher.state_dict()[k]) for k,v in teacher_state.items());tf=torch.cat(features);tp=torch.cat(probabilities);np.savez_compressed(a.out/'TRAIN_teacher_features.npz',ids=data['train'][2],labels=data['train'][1],features=tf.numpy(),probabilities=tp.numpy());del teacher
    allowed=seq.core.allowed_keys(model,a.prefix)
    for n,v in model.named_parameters():v.requires_grad_(n in allowed)
    groups=[]
    for prefix,lr in (('phase_sets.',cfg['phase_lr']),('electronic.',cfg['electronic_lr']),('fusion_logits.',cfg['alpha_lr']),('linear.',cfg['head_lr'])):
        ps=[p for n,p in model.named_parameters() if n in allowed and n.startswith(prefix)]
        if ps:groups.append(dict(params=ps,lr=lr,weight_decay=0 if prefix=='fusion_logits.' else cfg['weight_decay']))
    protocol=dict(config=cfg,source_SHA256=sha(__file__),core_SHA256=sha(seq.core.__file__),sequence_source_SHA256=sha(seq.__file__),data_SHA256=dm['sha256'],captured_prefix_SHA256=identities,parent_PT_SHA256=a.checkpoint_sha,original_teacher_PT_SHA256=gate['PT_SHA256'],TRAIN_teacher_cache_SHA256=sha(a.out/'TRAIN_teacher_features.npz'),TRAIN=1200,TEST=411,VALID=0,TEST_gradients=False,TEST_informed_development=True,evaluation_kind='measured_prefix_simulated_suffix',allowed_keys=sorted(allowed),no_new_inference_layer=True,SDK_opened=False,GPU_used=True,teacher_correct_TRAIN_only=True,original_fullphysical318_preserved=True,old_global_CCD_not_loaded=True,requires_fresh_global_CCD=True)
    save(a.out/'protocol.json',protocol);np.savez_compressed(a.out/'baseline_TEST_predictions.npz',ids=data['test'][2],labels=data['test'][1],probabilities=bp);best=round(baseline['accuracy']*411);best_nll=baseline['nll'];selected=0;history=[];seq.save_state(a.out/'best.pt',model,original,allowed,protocol,0);optimizer=torch.optim.AdamW(groups);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,80,eta_min=1e-5);start=time.monotonic();held={};handle=model.linear.register_forward_pre_hook(lambda _,args:held.update(features=args[0]))
    for epoch in range(1,81):
        order=torch.randperm(1200).numpy();model.train();ce_total=0.;teacher_total=0.
        for begin in range(0,1200,4):
            ix=order[begin:begin+4];optimizer.zero_grad(set_to_none=True);p=seq.feed(model,data,fields,'train',ix,'cuda');y=torch.from_numpy(data['train'][1][ix].astype(np.int64)).cuda();lp=p.clamp_min(1e-12).log();ce=(1-cfg['label_smoothing'])*F.nll_loss(lp,y)-cfg['label_smoothing']*lp.mean();knowledge=teacher_loss(lp,held['features'],tp[ix].cuda(),tf[ix].cuda(),y);loss=ce+knowledge;assert torch.isfinite(loss);loss.backward();assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters() if v.requires_grad);torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);optimizer.step()
            with torch.no_grad():
                for p in model.fusion_logits:
                    if p.requires_grad:p.clamp_(.4001001,.95)
            ce_total+=float(ce.detach())*len(ix);teacher_total+=float(knowledge.detach())*len(ix)
        scheduler.step();model.eval()
        if epoch==1 or epoch%5==0 or epoch==80 or time.monotonic()-start>=600:
            m,p,_=seq.assess(model,data,fields,canonical,'test',4);n=round(m['accuracy']*411)
            if seq.improves(n,m['nll'],best,best_nll):best,best_nll=n,m['nll'];selected=epoch;seq.save_state(a.out/'best.pt',model,original,allowed,protocol,epoch)
            row=dict(epoch=epoch,TRAIN_CE=ce_total/1200,TRAIN_teacher_loss=teacher_total/1200,full_CPU_TEST_correct=n,selected_correct=best,selected_epoch=selected,elapsed_fit_s=time.monotonic()-start);history.append(row);save(a.out/'history.json',history);save(a.out/'progress.json',dict(state='TRAIN_teacher_guarded_original_suffix',latest=row));print(json.dumps(row),flush=True)
        if time.monotonic()-start>=600:break
    handle.remove();seq.save_state(a.out/'last.pt',model,original,allowed,protocol,epoch);model.cpu().eval();results={}
    for name in ('best','last'):
        state=torch.load(a.out/(name+'.pt'),map_location='cpu',weights_only=False)['model'];model.load_state_dict(state,strict=True);assert all(torch.equal(v,state[k]) for k,v in original.items() if k not in allowed);m,p,_=seq.assess(model,data,fields,canonical,'test',4);tr,_,_=seq.assess(model,data,fields,canonical,'train',4);np.savez_compressed(a.out/(name+'_TEST_predictions.npz'),ids=data['test'][2],labels=data['test'][1],probabilities=p);results[name]=dict(TEST_development=m,TRAIN=tr,PT_SHA256=sha(a.out/(name+'.pt')),prediction_SHA256=sha(a.out/(name+'_TEST_predictions.npz')),strict_CPU=True)
    assert round(results['best']['TEST_development']['accuracy']*411)==best;save(a.out/'report.json',dict(state='complete',evaluation_kind='measured_prefix_simulated_suffix',selected_correct=best,selected_epoch=selected,actual_epochs=epoch,elapsed_fit_s=time.monotonic()-start,results=results,baseline_TEST=baseline,frozen_parameters_unchanged=True,full_CPU_replay_passed=True,TEST_gradients=False,TEST_informed_development=True,SDK_opened=False,is_NOT_fullphysical_result=True,requires_fresh_global_CCD=True,original_fullphysical318_preserved=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selftest',action='store_true');p.add_argument('--depth',type=int,default=2);p.add_argument('--prefix',type=int,default=2);p.add_argument('--checkpoint-sha')
    for n in ('session','profile','admission','checkpoint','out'):p.add_argument('--'+n,type=Path)
    a=p.parse_args()
    if sys.stdout is None:sys.stdout=a.out.with_suffix('.log').open('x',encoding='utf-8',buffering=1);sys.stderr=sys.stdout
    try:selftest() if a.selftest else run(a)
    except BaseException as e:
        if a.out:save(a.out.with_suffix('.failure.json'),dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False))
        raise
