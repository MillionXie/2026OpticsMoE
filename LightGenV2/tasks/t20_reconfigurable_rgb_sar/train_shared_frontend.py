"""Small-step joint optimization of existing optics/head, fixed shared frontend.

Same native graph and7840 head. No TEST fitting or score constraints. Starts from
the CPU-audited robust-readout parent; frozen frontend has TRAIN-only pretraining.
"""
import argparse,json,time,subprocess,math
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .unified_native import build_unified
from .static_four import build_static_four,geometry,CONTRACT as FOUR_CONTRACT
from .domain_frontend_probe import features,FRONTEND_SHA,CONTRACT
from .train_unified_static import load_arrays,balanced_batches,selection
from .sensor_neutral import amplitude
from .train_sensor_neutral import infer,collect_augmented
from .robust_readout import raw_region_dropout
from .refine_unified import conditioned_inputs,scores_from_logits,domain_balance
from LightGenV2.tasks.t21_continual_learning_hardware.train_native import sha,put,tensor_sha,gpu_gate


def validate(cfg):
    profiles=('config_shared_frontend_joint_20261025.json','config_static_four_20261025.json')
    if not any(cfg==json.loads(Path(__file__).with_name(n).read_text()) for n in profiles):raise ValueError('Exact finite common training profile required')


def main():
    p=argparse.ArgumentParser(__doc__)
    for n in ('data','frontend','profile','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--parent',type=Path);p.add_argument('--case',choices=('MoE_AB','D2NN_RGB','D2NN_AB_control'))
    p.add_argument('--gpu-uuid',required=True);p.add_argument('--preflight-only',action='store_true');a=p.parse_args();cfg=json.loads(a.profile.read_text());validate(cfg);torch.set_num_threads(2);torch.manual_seed(cfg['seed']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;gpu=gpu_gate(cfg,a.gpu_uuid);device=torch.device('cuda:0')
    fresh=cfg.get('contract')==FOUR_CONTRACT;contract=FOUR_CONTRACT if fresh else CONTRACT
    train=load_arrays(a.data,'train');tf=features(train,a.frontend)
    if fresh:
        if a.parent is not None or a.case not in cfg['cases']:raise ValueError('Fresh physical4 requires case, no old geometry parent')
        case=a.case;domains=[0] if case=='D2NN_RGB' else [0,1];saved=dict(architecture='moe' if case=='MoE_AB' else 'd2nn');parent=None;digest=None
        model=build_static_four(saved['architecture'],cfg['seed']).to(device)
        calibration=np.concatenate([np.flatnonzero((train['domains']==d)&(train['labels']==c))[:2] for c in range(10) for d in domains])
        with torch.no_grad():gain_report=model.calibrate_gains(amplitude(train,tf,calibration,device))
    else:
        if a.parent is None:raise ValueError('Audited parent required for historical continuation')
        parent=a.parent/'best_checkpoint.pt';digest=sha(parent);receipt=json.loads((a.parent/'completion.json').read_text());prior=json.loads((a.parent/'protocol.json').read_text());audit=json.loads((a.parent.parent/(a.parent.name+'_CPU')/'report.json').read_text());assert receipt['best_sha256']==digest and receipt['frontend_SHA256']==cfg['frontend_SHA256']==FRONTEND_SHA and not prior['selection_uses_TEST'];best_audit=next(v for v in audit['rows'] if v['role']=='best');assert best_audit['checkpoint_SHA256']==digest and best_audit['comparison_to_GPU']['legacy_parity_passed'],'Parent original numerical gate required'
        saved=torch.load(parent,map_location='cpu',weights_only=False);model=build_unified(saved['architecture']).to(device);model.load_state_dict(saved['model'],strict=True);domains=prior['training_domains'];case=prior['case'];gain_report=None
    gain=tensor_sha(model.display_gains);frozen={n:tensor_sha(v) for n,v in model.state_dict().items() if n=='display_gains' or (n in dict(model.named_parameters()) and not dict(model.named_parameters())[n].requires_grad)};optics=[v for n,v in model.named_parameters() if v.requires_grad and not n.startswith('shared_head.')];ix=next(balanced_batches(train,domains,cfg['batch_size'],42));r=model(amplitude(train,tf,ix,device));loss=F.cross_entropy(r['logits'],torch.as_tensor(train['labels'][ix],device=device));loss.backward()
    if any(v.grad is None or not torch.isfinite(v.grad).all() for v in model.parameters() if v.requires_grad):raise RuntimeError('Nonfinite real TRAIN gradients')
    if a.preflight_only:
        print(json.dumps(dict(passed=True,case=case,loss=float(loss.detach()),parent_SHA256=digest,fresh_optical_initialization=fresh,geometry=geometry(model) if fresh else None,gain_calibration=gain_report,TEST_loaded=False,SDK=False)));return
    model.zero_grad(set_to_none=True);a.out.mkdir(parents=True,exist_ok=False);started=time.monotonic();val=load_arrays(a.data,'validation');vf=features(val,a.frontend);flip=dict(train,images=train['images'][:,:,::-1,:].copy());ff=features(flip,a.frontend);head=torch.load(a.frontend,map_location='cpu',weights_only=False)['model'];teacher=[F.linear(f.float(),head['head.weight'][:10],head['head.bias'][:10]).to(device) for f in (tf,ff)];baseline=infer(model,val,vf,device);best=selection(baseline,domains);best_epoch=0;best_step=0;history=[];last_improvement=0
    sources={str(v):sha(v) for v in [Path(__file__),a.profile,Path(__file__).with_name('static_four.py'),Path(__file__).with_name('domain_frontend_probe.py'),Path(__file__).with_name('sensor_neutral.py'),Path(__file__).with_name('train_sensor_neutral.py'),Path(__file__).with_name('refine_unified.py'),Path('LightGenV2/tasks/t21_continual_learning_hardware/model.py'),Path(__file__).with_name('hardware_aware.py')]}
    put(a.out/'protocol.json',dict(contract=contract,config=cfg,case=case,architecture=saved['architecture'],parent_run=str(a.parent) if a.parent else None,parent_SHA256=digest,fresh_optical_initialization=fresh,geometry=geometry(model) if fresh else None,gain_calibration=gain_report,training_domains=domains,frontend_sha256=FRONTEND_SHA,fixed_gain_sha256=gain,frozen_state_sha256=frozen,phase_updates=True,original_head_parameters=7840,teacher_scope='Original TRAIN and TRAIN flips only; shared frontend pretrained on both TRAIN domains; RGB-only means optical/student fitting only',dataset_manifest_sha256=sha(a.data/'manifest.json'),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256=sources,SDK=False,selection_uses_TEST=False,TEST_gradients=False,TEST_teacher=False,gpu=gpu));put(a.out/'baseline_validation.json',baseline)
    def save(role,epoch,step=0):
        torch.save(dict(model={n:v.detach().cpu() for n,v in model.state_dict().items()},architecture=saved['architecture'],stage=1,epoch=epoch,readout_step=step,training_stage='head_warm' if step<0 else ('fixed_optics_readout' if step else 'joint_training'),case=case,contract=contract,frontend_sha256=FRONTEND_SHA,parent_SHA256=digest,TEST_gradients=False),a.out/(role+'_checkpoint.pt'))
    def consider(report,epoch,step=0):
        nonlocal best,best_epoch,best_step
        score=selection(report,domains)
        if score>best:best=score;best_epoch=epoch;best_step=step;save('best',epoch,step);put(a.out/'selected_validation.json',report);return True
        return False
    def guard():
        if (parent is not None and sha(parent)!=digest) or tensor_sha(model.display_gains)!=gain or any(tensor_sha(model.state_dict()[n])!=v for n,v in frozen.items()) or any(sha(n)!=v for n,v in sources.items()):raise RuntimeError('Frozen parent/gain/source changed')
    save('best',0);save('last',0)
    if fresh:
        model.eval();unused,tx=infer(model,train,tf,device,True);fx=collect_augmented(model,flip,ff,device);unused,vx=infer(model,val,vf,device,True);mask=np.isin(train['domains'],domains);warmx=torch.as_tensor(np.concatenate((tx[mask],fx[mask])),device=device);warmy=torch.as_tensor(np.tile(train['labels'][mask],2),device=device);warmopt=torch.optim.Adam(model.shared_head.parameters(),lr=cfg['head_warm_lr'])
        for step in range(1,cfg['head_warm_epochs']+1):
            order=torch.randperm(len(warmy),device=device);total=0.
            for rows in order.split(128):
                warmopt.zero_grad();cost=F.cross_entropy(model.shared_head(warmx[rows]),warmy[rows],label_smoothing=cfg['label_smoothing'])+cfg['weight_regularization']*model.shared_head.weight.square().sum()/10;cost.backward();warmopt.step();total+=float(cost.detach())*len(rows)
            report=scores_from_logits(vx@model.shared_head.weight.detach().cpu().numpy().T,val);consider(report,0,-step);save('last',0,-step);history.append(dict(stage='head_warm',step=step,TRAIN_loss=total/len(warmy),validation=report['scores']));put(a.out/'history.json',history);put(a.out/'status.json',dict(state='head_warm',step=step,updated=time.time()));guard()
        del warmx,warmy
    optimizer=torch.optim.AdamW([dict(params=optics,lr=cfg['optical_lr']),dict(params=model.shared_head.parameters(),lr=cfg['head_lr'])],weight_decay=0.)
    for epoch in range(1,cfg['epochs']+1):
        model.train();total=0.;seen=0;correct={d:0 for d in domains};counts={d:0 for d in domains};rng=np.random.default_rng(cfg['seed']+epoch);factor=.1+.9*.5*(1+math.cos(math.pi*(epoch-1)/max(1,cfg['epochs']-1)))
        for group,lr in zip(optimizer.param_groups,(cfg['optical_lr'],cfg['head_lr'])):group['lr']=lr*factor
        for ix in balanced_batches(train,domains,cfg['batch_size'],cfg['seed']+epoch):
            field=amplitude(train,tf,ix,device);fl=rng.random(len(ix))<cfg['TRAIN_flip_probability'];target=teacher[0][ix].clone()
            if fl.any():field[torch.as_tensor(fl,device=device)]=amplitude(flip,ff,ix[fl],device);target[torch.as_tensor(fl,device=device)]=teacher[1][ix[fl]]
            drop=rng.random(len(ix))<cfg['TRAIN_raw_dropout_probability']
            if drop.any():field[torch.as_tensor(drop,device=device)]=raw_region_dropout(field[torch.as_tensor(drop,device=device)])
            optimizer.zero_grad(set_to_none=True);r=model(field);y=torch.as_tensor(train['labels'][ix],device=device);d=torch.as_tensor(train['domains'][ix],device=device);temp=cfg['teacher_temperature'];cost=F.cross_entropy(r['logits'],y,label_smoothing=cfg['label_smoothing'])+cfg['teacher_KD_weight']*F.kl_div(F.log_softmax(r['logits']/temp,1),(target/temp).softmax(1),reduction='batchmean')*temp**2+cfg['weight_regularization']*model.shared_head.weight.square().sum()/10
            if saved['architecture']=='moe':cost+=cfg['domain_balance_weight']*domain_balance(r['route_power'],d,model.active_indices[:4])
            if not torch.isfinite(cost):raise RuntimeError('Nonfinite shared-frontend TRAIN objective')
            cost.backward();torch.nn.utils.clip_grad_norm_(optics,1.);torch.nn.utils.clip_grad_norm_(model.shared_head.parameters(),5.);optimizer.step();total+=float(cost.detach())*len(ix);seen+=len(ix)
            for domain in domains:correct[domain]+=int(((r['logits'].argmax(1)==y)&(d==domain)).sum());counts[domain]+=int((d==domain).sum())
            if seen%1000<len(ix):put(a.out/'status.json',dict(state='joint_training',epoch=epoch,TRAIN_seen=seen,loss=total/seen,updated=time.time()))
        guard();report=infer(model,val,vf,device)
        if consider(report,epoch):last_improvement=epoch
        save('last',epoch);row=dict(stage='joint_training',epoch=epoch,TRAIN_loss=total/seen,TRAIN_accuracy_by_domain={str(d):correct[d]/counts[d] for d in domains},validation=report['scores'],best_epoch=best_epoch,head_norm=float(model.shared_head.weight.detach().norm()));history.append(row);put(a.out/'history.json',history);put(a.out/'status.json',dict(state='joint_training',**row,updated=time.time()))
        if time.monotonic()-started>cfg['max_case_seconds']:raise RuntimeError('Finite full-epoch time bound')
        if epoch>=cfg['minimum_epochs'] and epoch-last_improvement>=cfg['patience']:break
    epochs_completed=epoch;chosen=torch.load(a.out/'best_checkpoint.pt',map_location=device,weights_only=False);model.load_state_dict(chosen['model'],strict=True);base_epoch=chosen['epoch'];model.eval();unused,tx=infer(model,train,tf,device,True);unused,vx=infer(model,val,vf,device,True);fx=collect_augmented(model,flip,ff,device);mask=np.isin(train['domains'],domains);z,zv,t=conditioned_inputs(np.concatenate((tx[mask],fx[mask])),vx,cfg['conditioner_ridge']);labels=torch.as_tensor(np.tile(train['labels'][mask],2));param=torch.nn.Parameter(torch.linalg.solve(t,model.shared_head.weight.detach().cpu().double().T).T);headopt=torch.optim.LBFGS([param],lr=1.,max_iter=5,history_size=10,line_search_fn='strong_wolfe')
    for step in range(1,cfg['readout_fit_steps']+1):
        def closure():
            headopt.zero_grad();weight=param@t.T;cost=F.cross_entropy(z@param.T,labels,label_smoothing=cfg['label_smoothing'])+cfg['coordinate_regularization']*param.square().sum()/10+cfg['weight_regularization']*weight.square().sum()/10;cost.backward();return cost
        loss=float(headopt.step(closure).detach());w=(param.detach()@t.T).float()
        if not np.isfinite(loss) or not torch.isfinite(w).all():raise RuntimeError('Nonfinite final original head fit')
        with torch.no_grad():model.shared_head.weight.copy_(w.to(device))
        report=scores_from_logits(vx@w.numpy().T,val);consider(report,base_epoch,step);save('last',base_epoch,step);history.append(dict(stage='fixed_optics_readout',step=step,TRAIN_loss=loss,validation=report['scores']));put(a.out/'history.json',history);put(a.out/'status.json',dict(state='fixed_optics_readout',step=step,validation=report['scores'],updated=time.time()))
    guard();locked=dict(best_sha256=sha(a.out/'best_checkpoint.pt'),last_sha256=sha(a.out/'last_checkpoint.pt'),TEST_loaded=False);put(a.out/'locked_before_TEST.json',locked);chosen=torch.load(a.out/'best_checkpoint.pt',map_location=device,weights_only=False);model.load_state_dict(chosen['model'],strict=True);test=load_arrays(a.data,'test');testfront=features(test,a.frontend);report=infer(model,test,testfront,device);put(a.out/'selected_test.json',report);assert sha(a.out/'best_checkpoint.pt')==locked['best_sha256']
    put(a.out/'completion.json',dict(passed=True,contract=contract,case=case,architecture=saved['architecture'],best_epoch=best_epoch,best_step=best_step,epochs_completed=epochs_completed,test=report['scores'],best_sha256=locked['best_sha256'],last_sha256=locked['last_sha256'],frontend_SHA256=FRONTEND_SHA,phase_updates=True,TEST_gradients=False,TEST_teacher=False,selection_uses_TEST=False,SDK=False,independent_new_blind_TEST_claim=False,seconds=time.monotonic()-started));put(a.out/'status.json',dict(state='complete_CPU_audit_pending',updated=time.time()))


if __name__=='__main__':main()
