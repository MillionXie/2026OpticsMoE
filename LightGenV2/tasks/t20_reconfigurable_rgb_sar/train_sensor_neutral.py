"""Same T21 backend, corrected single-sensor input, finite TRAIN-only training.

Train optical/head jointly BEFORE the final conditioned head fit. An ill-
conditioned high-gain fitted head is not fed back into optical optimization.
"""
import argparse,json,time,subprocess,math
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .unified_native import build_unified,FRONTEND_SHA256
from .sensor_neutral import CONTRACT,frontend,amplitude
from .train_unified_static import load_arrays,balanced_batches,selection
from .refine_unified import conditioned_inputs,scores_from_logits,domain_balance
from LightGenV2.tasks.t21_continual_learning_hardware.train_native import sha,put,tensor_sha,gpu_gate


def validate(cfg):
    if cfg!=json.loads(Path(__file__).with_name('config_sensor_neutral_20261025.json').read_text()):raise ValueError('Exact published finite profile required')


@torch.no_grad()
def infer(model,data,front,device,return_features=False):
    model.eval();logits=[];features=[];routes=[]
    for start in range(0,len(data['labels']),8):
        ix=np.arange(start,min(start+8,len(data['labels'])));r=model(amplitude(data,front,ix,device));logits.append(r['logits'].cpu())
        if return_features:features.append(r['ccd_features'].cpu().numpy())
        if r['route_power'] is not None:routes.extend(r['route_power'].cpu().tolist())
    logits=torch.cat(logits);report=scores_from_logits(logits,data);report.update(ids=data['ids'].tolist(),labels=data['labels'].tolist(),domains=data['domains'].tolist(),predictions=logits.argmax(1).tolist(),probabilities=logits.softmax(1).tolist())
    if routes:
        q=np.asarray(routes);report['routing_by_domain']={name:dict(mean_power=q[data['domains']==d].mean(0).tolist(),winner_fraction=(np.bincount(q[data['domains']==d].argmax(1),minlength=16)/(data['domains']==d).sum()).tolist()) for d,name in ((0,'RGB'),(1,'SAR'))}
    return (report,np.concatenate(features)) if return_features else report


@torch.no_grad()
def collect_augmented(model,data,front,device):
    rows=[]
    for start in range(0,len(data['labels']),8):
        ix=np.arange(start,min(start+8,len(data['labels'])));rows.append(model(amplitude(data,front,ix,device))['ccd_features'].cpu().numpy())
    return np.concatenate(rows)


def main():
    p=argparse.ArgumentParser(__doc__)
    for n in ('parent','data','frontend','profile','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--gpu-uuid',required=True);p.add_argument('--preflight-only',action='store_true');a=p.parse_args();cfg=json.loads(a.profile.read_text());validate(cfg);torch.set_num_threads(2);torch.manual_seed(cfg['seed']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    gpu=gpu_gate(cfg,a.gpu_uuid);device=torch.device('cuda:0');receipt=json.loads((a.parent/'completion.json').read_text());prior=json.loads((a.parent/'protocol.json').read_text());parent=a.parent/'best_checkpoint.pt';parent_sha=sha(parent)
    if parent_sha!=receipt['best_sha256'] or prior['selection_uses_TEST']:raise ValueError('Locked validation-selected parent required')
    saved=torch.load(parent,map_location='cpu',weights_only=False);architecture=saved['architecture'];domains=prior['training_domains'];case=prior['case'];model=build_unified(architecture).to(device);model.load_state_dict(saved['model'],strict=True)
    train=load_arrays(a.data,'train');tf=frontend(train,a.frontend);cal=np.concatenate([np.flatnonzero((train['domains']==d)&(train['labels']==c))[:2] for c in range(10) for d in domains])
    with torch.no_grad():gains=model.calibrate_gains(amplitude(train,tf,cal,device))
    gain_sha=tensor_sha(model.display_gains);frozen={n:tensor_sha(v) for n,v in model.named_parameters() if not v.requires_grad};ix=next(balanced_batches(train,domains,8,42));r=model(amplitude(train,tf,ix,device));cost=F.cross_entropy(r['logits'],torch.as_tensor(train['labels'][ix],device=device));cost.backward()
    if any(v.grad is None or not torch.isfinite(v.grad).all() for v in model.parameters() if v.requires_grad):raise RuntimeError('Nonfinite real TRAIN preflight')
    if a.preflight_only:
        print(json.dumps(dict(passed=True,case=case,loss=float(cost.detach()),gain_calibration=gains,TEST_loaded=False,SDK=False)));return
    model.zero_grad(set_to_none=True);a.out.mkdir(parents=True,exist_ok=False);started=time.monotonic();val=load_arrays(a.data,'validation');vf=frontend(val,a.frontend);flipped=dict(train,images=train['images'][:,:,::-1,:].copy());ff=frontend(flipped,a.frontend)
    paths=[Path(__file__),Path(__file__).with_name('sensor_neutral.py'),Path(__file__).with_name('unified_native.py'),Path(__file__).with_name('refine_unified.py'),Path('LightGenV2/tasks/t21_continual_learning_hardware/model.py'),Path(__file__).with_name('hardware_aware.py')];sources={str(v):sha(v) for v in paths}
    put(a.out/'protocol.json',dict(contract=CONTRACT,case=case,architecture=architecture,config=cfg,config_sha256=sha(a.profile),parent_run=str(a.parent),parent_SHA256=parent_sha,training_domains=domains,frontend_sha256=FRONTEND_SHA256,fixed_gain_sha256=gain_sha,gain_calibration=gains,calibration_TRAIN_ids=train['ids'][cal].tolist(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256=sources,dataset_manifest_sha256=sha(a.data/'manifest.json'),input_contract='Same three observed sensor channels /same frozen128 feature strip /same positions forRGB andSAR; no sensor/label/ID selector',physical_graph='Exact T21 native8 A graph,16slots/4activeMoE, original7840 no-biashead',SDK=False,selection_uses_TEST=False,TEST_gradients=False,TEST_teacher=False,gpu=gpu))
    baseline=infer(model,val,vf,device);put(a.out/'baseline_validation.json',baseline);best=selection(baseline,domains);best_epoch=0;best_head_step=0;history=[];last_improvement=0
    def save(role,epoch,step=0):
        torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},architecture=architecture,stage=1,epoch=epoch,readout_step=step,contract=CONTRACT,case=case,frontend_sha256=FRONTEND_SHA256,config_sha256=sha(a.profile),parent_SHA256=parent_sha,TEST_gradients=False),a.out/(role+'_checkpoint.pt'))
    def consider(report,epoch,step=0):
        nonlocal best,best_epoch,best_head_step
        score=selection(report,domains)
        if score>best:
            best=score;best_epoch=epoch;best_head_step=step;save('best',epoch,step);put(a.out/'selected_validation.json',report);return True
        return False
    def guard():
        if tensor_sha(model.display_gains)!=gain_sha or any(tensor_sha(dict(model.named_parameters())[n])!=v for n,v in frozen.items()):raise RuntimeError('Frozen state changed')
        if sha(parent)!=parent_sha or any(sha(v)!=sources[str(v)] for v in paths):raise RuntimeError('Parent or numerical source changed')
    save('best',0);save('last',0);optics=[v for n,v in model.named_parameters() if v.requires_grad and not n.startswith('shared_head.')];optimizer=torch.optim.AdamW([dict(params=optics,lr=cfg['optical_lr'],weight_decay=0.),dict(params=model.shared_head.parameters(),lr=cfg['head_lr'],weight_decay=cfg['head_weight_decay'])])
    for epoch in range(1,cfg['epochs']+1):
        model.train();total=0.;seen=0;correct={d:0 for d in domains};counts={d:0 for d in domains};rng=np.random.default_rng(cfg['seed']+epoch);fraction=.1+.9*.5*(1+math.cos(math.pi*(epoch-1)/max(1,cfg['epochs']-1)))
        for group,lr in zip(optimizer.param_groups,(cfg['optical_lr'],cfg['head_lr'])):group['lr']=lr*fraction
        for ix in balanced_batches(train,domains,8,cfg['seed']+epoch):
            field=amplitude(train,tf,ix,device);flip=rng.random(len(ix))<cfg['TRAIN_horizontal_flip_probability']
            if flip.any():field[torch.as_tensor(flip,device=device)]=amplitude(flipped,ff,ix[flip],device)
            optimizer.zero_grad(set_to_none=True);r=model(field);y=torch.as_tensor(train['labels'][ix],device=device);d=torch.as_tensor(train['domains'][ix],device=device);loss=F.cross_entropy(r['logits'],y,label_smoothing=cfg['label_smoothing'])
            if architecture=='moe':loss+=cfg['domain_balance_weight']*domain_balance(r['route_power'],d,model.active_indices[:4])
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite sensor-neutral loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(optics,cfg['optical_clip_norm']);torch.nn.utils.clip_grad_norm_(model.shared_head.parameters(),cfg['head_clip_norm']);optimizer.step();total+=float(loss.detach())*len(ix);seen+=len(ix)
            for domain in domains:
                mask=d==domain;correct[domain]+=int(((r['logits'].argmax(1)==y)&mask).sum());counts[domain]+=int(mask.sum())
            if seen%1000<len(ix):put(a.out/'status.json',dict(state='optical_training',epoch=epoch,TRAIN_seen=seen,loss=total/seen,updated=time.time()))
        guard();report=infer(model,val,vf,device)
        if consider(report,epoch):last_improvement=epoch
        save('last',epoch);row=dict(stage='optical_training',epoch=epoch,TRAIN_loss=total/seen,TRAIN_accuracy_by_domain={str(d):correct[d]/counts[d] for d in domains},validation=report['scores'],routing_by_domain=report.get('routing_by_domain'),best_epoch=best_epoch);history.append(row);put(a.out/'history.json',history);put(a.out/'status.json',dict(state='optical_training',**row,updated=time.time()))
        if time.monotonic()-started>cfg['max_case_seconds']:raise RuntimeError('Finite full-epoch wall limit')
        if epoch>=cfg['minimum_epochs'] and epoch-last_improvement>=cfg['patience']:break
    optical_epochs=epoch;cp=torch.load(a.out/'best_checkpoint.pt',map_location=device,weights_only=False);base_epoch=cp['epoch'];model.load_state_dict(cp['model'],strict=True);model.eval();report,tx=infer(model,train,tf,device,True);unused,vx=infer(model,val,vf,device,True);fx=collect_augmented(model,flipped,ff,device);mask=np.isin(train['domains'],domains);x=np.concatenate((tx[mask],fx[mask]));y=torch.as_tensor(np.tile(train['labels'][mask],2));z,zv,t=conditioned_inputs(x,vx,cfg['conditioner_ridge']);original=model.shared_head.weight.detach().cpu().double();param=torch.nn.Parameter(torch.linalg.solve(t,original.T).T);headopt=torch.optim.LBFGS([param],lr=1.,max_iter=5,history_size=10,line_search_fn='strong_wolfe')
    for step in range(1,cfg['readout_fit_steps']+1):
        def closure():
            headopt.zero_grad();loss=F.cross_entropy(z@param.T,y,label_smoothing=cfg['label_smoothing'])+cfg['readout_regularization']*param.square().sum()/10;loss.backward();return loss
        loss=float(headopt.step(closure).detach());w=(param.detach()@t.T).float()
        if not math.isfinite(loss) or not torch.isfinite(w).all():raise RuntimeError('Nonfinite final head fit')
        with torch.no_grad():model.shared_head.weight.copy_(w.to(device))
        report=scores_from_logits(vx@w.numpy().T,val);consider(report,base_epoch,step);save('last',base_epoch,step);history.append(dict(stage='fixed_optics_readout',step=step,base_optical_epoch=base_epoch,TRAIN_loss=loss,validation=report['scores']));put(a.out/'history.json',history);put(a.out/'status.json',dict(state='fixed_optics_readout',step=step,validation=report['scores'],updated=time.time()))
    guard();locked=dict(best_sha256=sha(a.out/'best_checkpoint.pt'),last_sha256=sha(a.out/'last_checkpoint.pt'),TEST_loaded=False);put(a.out/'locked_before_TEST.json',locked);cp=torch.load(a.out/'best_checkpoint.pt',map_location=device,weights_only=False);model.load_state_dict(cp['model'],strict=True);test=load_arrays(a.data,'test');testfront=frontend(test,a.frontend);report=infer(model,test,testfront,device);put(a.out/'selected_test.json',report)
    if sha(a.out/'best_checkpoint.pt')!=locked['best_sha256']:raise RuntimeError('Locked weights changed')
    put(a.out/'completion.json',dict(passed=True,contract=CONTRACT,case=case,architecture=architecture,best_epoch=best_epoch,best_head_step=best_head_step,optical_epochs=optical_epochs,test=report['scores'],**{k:v for k,v in locked.items() if k!='TEST_loaded'},selection_uses_TEST=False,TEST_gradients=False,TEST_teacher=False,SDK=False,independent_new_blind_TEST_claim=False,original_graph_CPU_audit_pending=True,seconds=time.monotonic()-started));put(a.out/'status.json',dict(state='complete_CPU_audit_pending',updated=time.time()))


if __name__=='__main__':main()
