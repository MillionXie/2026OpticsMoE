"""Small-step joint optimization of existing optics/head, fixed shared frontend.

Same native graph and7840 head. No TEST fitting or score constraints. Starts from
the CPU-audited robust-readout parent; frozen frontend has TRAIN-only pretraining.
"""
import argparse,json,time,subprocess,math
from functools import partial
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .unified_native import build_unified
from .static_four import build_static_four,geometry,CONTRACT as FOUR_CONTRACT
from .domain_frontend_probe import features,FRONTEND_SHA,CONTRACT
from .diagnose_frontend_transfer import checked_features
from .train_unified_static import load_arrays,balanced_batches,selection
from .sensor_neutral import amplitude
from .train_sensor_neutral import infer,collect_augmented
from .robust_readout import raw_region_dropout
from .diagnose_frontend_train import pair_batches
from .augmented_teacher import load_teacher,teacher_logits
from .relational_distillation import teacher_representations,relational_losses
from .refine_unified import conditioned_inputs,scores_from_logits,domain_balance
from LightGenV2.tasks.t21_continual_learning_hardware.train_native import sha,put,tensor_sha,gpu_gate


def validate(cfg):
    profiles=('config_shared_frontend_joint_20261025.json','config_static_four_20261025.json','config_static_four_aligned_20261025.json','config_static_four_phase_projection_20261025.json','config_static_four_teacher_KD_20261025.json','config_static_four_teacher_continuation_20261025.json','config_static_four_expert_specialization_20261025.json','config_static_four_relational_KD_20261026.json')
    if not any(cfg==json.loads(Path(__file__).with_name(n).read_text()) for n in profiles):raise ValueError('Exact finite common training profile required')


def merge_phase_gradients(rgb,sar):
    """Symmetric two-domain projection within ONE existing phase block."""
    dot=(rgb*sar).sum();conflict=bool(dot<0)
    if conflict:
        r=rgb-dot/sar.square().sum().clamp_min(1e-20)*sar
        s=sar-dot/rgb.square().sum().clamp_min(1e-20)*rgb
        return (r+s)*.5,True
    return (rgb+sar)*.5,False


def rgb_priority_selection(report,cfg,domains):
    """One PT, fixed VAL-only SAR floor; never clip scores to a requested gap."""
    if not cfg.get('RGB_priority_selection',False):return selection(report,domains)
    rgb=report['scores']['RGB'];sar=report['scores']['SAR']
    return (int(sar['accuracy']>=cfg['SAR_validation_floor']),rgb['accuracy'],sar['accuracy'],-rgb['loss'])


def d4_views(data):
    return [dict(data,images=np.ascontiguousarray(np.rot90(data['images'],k,(1,2))[:,:,::-1,:] if flip else np.rot90(data['images'],k,(1,2)))) for k in range(4) for flip in (False,True)]


def teacher_KD_rows(logits,target,y,cfg):
    temp=cfg['teacher_temperature'];kd=F.kl_div(F.log_softmax(logits/temp,1),(target/temp).softmax(1),reduction='none').sum(1)*temp**2
    if cfg.get('teacher_correct_only',False):kd=kd*(target.argmax(1)==y)
    return F.cross_entropy(logits,y,label_smoothing=cfg['label_smoothing'],reduction='none')+cfg['teacher_KD_weight']*kd


def restrict_phase_updates(model,prefixes):
    """Freeze existing phase blocks only; never change the inference graph."""
    if prefixes:
        for name,p in model.named_parameters():
            if 'phase' in name:p.requires_grad_(name.startswith(tuple(prefixes)))


def main():
    p=argparse.ArgumentParser(__doc__)
    for n in ('data','frontend','profile','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--parent',type=Path);p.add_argument('--teacher',type=Path);p.add_argument('--case',choices=('MoE_AB','D2NN_RGB','D2NN_AB_control'))
    p.add_argument('--gpu-uuid',required=True);p.add_argument('--preflight-only',action='store_true');a=p.parse_args();cfg=json.loads(a.profile.read_text());validate(cfg);torch.set_num_threads(2);torch.manual_seed(cfg['seed']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;gpu=gpu_gate(cfg,a.gpu_uuid);device=torch.device('cuda:0')
    external=cfg.get('external_augmented_teacher',False)
    relational=cfg.get('relational_distillation',False)
    if external!=(a.teacher is not None):raise ValueError('Explicit matching external teacher/profile required')
    four=cfg.get('contract')==FOUR_CONTRACT;fresh=four and not cfg.get('continuation',False);contract=FOUR_CONTRACT if four else CONTRACT;front_sha=cfg['frontend_SHA256'];extract_features=partial(checked_features,expected_sha256=front_sha) if four else features
    train=load_arrays(a.data,'train');tf=extract_features(train,a.frontend)
    if fresh:
        if a.parent is not None or a.case not in cfg['cases']:raise ValueError('Fresh physical4 requires case, no old geometry parent')
        case=a.case;domains=[0] if case=='D2NN_RGB' else [0,1];saved=dict(architecture='moe' if case=='MoE_AB' else 'd2nn');parent=None;digest=None
        model=build_static_four(saved['architecture'],cfg['seed']).to(device)
        calibration=np.concatenate([np.flatnonzero((train['domains']==d)&(train['labels']==c))[:2] for c in range(10) for d in domains])
        with torch.no_grad():gain_report=model.calibrate_gains(amplitude(train,tf,calibration,device))
    else:
        if a.parent is None:raise ValueError('Audited parent required for historical continuation')
        parent=a.parent/'best_checkpoint.pt';digest=sha(parent);receipt=json.loads((a.parent/'completion.json').read_text());prior=json.loads((a.parent/'protocol.json').read_text());audit=json.loads((a.parent.parent/(a.parent.name+'_CPU')/'report.json').read_text());assert receipt['best_sha256']==digest and receipt['frontend_SHA256']==cfg.get('parent_frontend_SHA256',front_sha) and not prior['selection_uses_TEST'];best_audit=next(v for v in audit['rows'] if v['role']=='best');assert best_audit['checkpoint_SHA256']==digest and best_audit['comparison_to_GPU']['legacy_parity_passed'],'Parent original numerical gate required'
        if cfg.get('parent_expected_SHA256'):assert digest==cfg['parent_expected_SHA256']
        saved=torch.load(parent,map_location='cpu',weights_only=False);assert saved['contract']==contract;model=(build_static_four if four else build_unified)(saved['architecture']).to(device);model.load_state_dict(saved['model'],strict=True);domains=prior['training_domains'];case=prior['case'];gain_report=None
        if four and cfg.get('recalibrate_gains',True):
            calibration=np.concatenate([np.flatnonzero((train['domains']==d)&(train['labels']==c))[:2] for c in range(10) for d in domains])
            with torch.no_grad():gain_report=model.calibrate_gains(amplitude(train,tf,calibration,device))
    if cfg.get('phase_gradient_projection',False):assert case=='MoE_AB' and domains==[0,1]
    if external:assert four and case=='MoE_AB' and domains==[0,1] and not cfg.get('recalibrate_gains',True)
    restrict_phase_updates(model,cfg.get('trainable_phase_prefixes'))
    gain=tensor_sha(model.display_gains);frozen={n:tensor_sha(v) for n,v in model.state_dict().items() if n=='display_gains' or (n in dict(model.named_parameters()) and not dict(model.named_parameters())[n].requires_grad)};optics=[v for n,v in model.named_parameters() if v.requires_grad and not n.startswith('shared_head.')];ix=next(pair_batches(train,cfg['batch_size'],42) if cfg.get('paired_TRAIN_batches',False) else balanced_batches(train,domains,cfg['batch_size'],42));r=model(amplitude(train,tf,ix,device));loss=F.cross_entropy(r['logits'],torch.as_tensor(train['labels'][ix],device=device))
    if cfg.get('phase_gradient_projection',False):
        named=[(n,p) for n,p in model.named_parameters() if p.requires_grad];labels=torch.as_tensor(train['labels'][ix],device=device);d=torch.as_tensor(train['domains'][ix],device=device);gr=torch.autograd.grad(F.cross_entropy(r['logits'][d==0],labels[d==0]),[p for n,p in named],retain_graph=True);gs=torch.autograd.grad(F.cross_entropy(r['logits'][d==1],labels[d==1]),[p for n,p in named])
        for (name,p),rgb_grad,sar_grad in zip(named,gr,gs):p.grad=(merge_phase_gradients(rgb_grad,sar_grad)[0] if 'phase' in name else (rgb_grad+sar_grad)*.5).detach()
    else:
        if relational:
            tm,_=load_teacher(a.teacher,a.data,device)
            _,emb=teacher_representations(tm,dict(images=train['images'][ix],labels=train['labels'][ix]),device)
            rd,ra=relational_losses(r['ccd_features'],emb.to(device),torch.as_tensor(train['domains'][ix],device=device))
            loss=loss+cfg['relation_distance_weight']*rd+cfg['relation_angle_weight']*ra
            del tm
        loss.backward()
    if any(v.grad is None or not torch.isfinite(v.grad).all() for v in model.parameters() if v.requires_grad):raise RuntimeError('Nonfinite real TRAIN gradients')
    if a.preflight_only:
        print(json.dumps(dict(passed=True,case=case,loss=float(loss.detach()),parent_SHA256=digest,fresh_optical_initialization=fresh,geometry=geometry(model) if four else None,gain_calibration=gain_report,TEST_loaded=False,SDK=False)));return
    model.zero_grad(set_to_none=True);a.out.mkdir(parents=True,exist_ok=False);started=time.monotonic();val=load_arrays(a.data,'validation');vf=extract_features(val,a.frontend);flip=dict(train,images=train['images'][:,:,::-1,:].copy());ff=extract_features(flip,a.frontend);head=torch.load(a.frontend,map_location='cpu',weights_only=False)['model'];teacher=[F.linear(f.float(),head['head.weight'][:10],head['head.bias'][:10]).to(device) for f in (tf,ff)];baseline=infer(model,val,vf,device);best=rgb_priority_selection(baseline,cfg,domains);best_epoch=0;best_step=0;history=[];last_improvement=0;teacher_receipt=None;teacher_path=None;teacher_digest=None
    if external:
        tm,teacher_receipt=load_teacher(a.teacher,a.data,device);teacher_path=a.teacher/'best_checkpoint.pt';teacher_digest=sha(teacher_path)
        assert teacher_receipt['validation']['RGB']['accuracy']>=cfg['teacher_minimum_RGB_validation'] and teacher_receipt['validation']['SAR']['accuracy']>=cfg['teacher_minimum_SAR_validation'],'Teacher must be strong on held-out validation before distillation'
        views=d4_views(train);view_fronts=[tf,ff]+[extract_features(v,a.frontend) for v in views[2:]];teacher=[];teacher_embeddings=[]
        for v,data_view in enumerate(views):
            if relational:
                logits,embedding=teacher_representations(tm,data_view,device);teacher.append(logits.to(device));teacher_embeddings.append(embedding.to(device))
            else:teacher.append(teacher_logits(tm,data_view,device).to(device))
            put(a.out/'status.json',dict(state='TRAIN_teacher_view_cache',view=v+1,total_views=8,updated=time.time()))
        del tm;torch.cuda.empty_cache()
    sources={str(v):sha(v) for v in [Path(__file__),a.profile,Path(__file__).with_name('augmented_teacher.py'),Path(__file__).with_name('static_four.py'),Path(__file__).with_name('diagnose_frontend_transfer.py'),Path(__file__).with_name('diagnose_frontend_train.py'),Path(__file__).with_name('domain_frontend_probe.py'),Path(__file__).with_name('sensor_neutral.py'),Path(__file__).with_name('train_sensor_neutral.py'),Path(__file__).with_name('refine_unified.py'),Path('LightGenV2/tasks/t21_continual_learning_hardware/model.py'),Path(__file__).with_name('hardware_aware.py')]}
    put(a.out/'protocol.json',dict(contract=contract,config=cfg,case=case,architecture=saved['architecture'],parent_run=str(a.parent) if a.parent else None,parent_SHA256=digest,fresh_optical_initialization=fresh,geometry=geometry(model) if four else None,gain_calibration=gain_report,training_domains=domains,frontend_sha256=front_sha,fixed_gain_sha256=gain,frozen_state_sha256=frozen,phase_updates=True,original_head_parameters=7840,teacher_scope='Original TRAIN and TRAIN flips only; shared frontend pretrained on both TRAIN domains; RGB-only means optical/student fitting only',dataset_manifest_sha256=sha(a.data/'manifest.json'),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256=sources,SDK=False,selection_uses_TEST=False,TEST_gradients=False,TEST_teacher=False,gpu=gpu));put(a.out/'baseline_validation.json',baseline)
    if relational:
        helper=Path(__file__).with_name('relational_distillation.py');sources[str(helper)]=sha(helper)
        proto=json.loads((a.out/'protocol.json').read_text());proto.update(source_sha256=sources,relational_feature='Original student final CCD features; offline teacher pooled 256; within sensor-domain distance and angle',new_student_parameters=0);put(a.out/'protocol.json',proto)
    if external:
        proto=json.loads((a.out/'protocol.json').read_text());proto.update(external_teacher_run=str(a.teacher),external_teacher_SHA256=teacher_digest,external_teacher_validation=teacher_receipt['validation'],teacher_scope='Larger offline image CNN; TRAIN and TRAIN D4 only; validation selected; no TEST access; original student inference unchanged',RGB_priority_selection=True,SAR_validation_floor=cfg['SAR_validation_floor'],trainable_parameters={n:p.numel() for n,p in model.named_parameters() if p.requires_grad});put(a.out/'protocol.json',proto)
    def save(role,epoch,step=0):
        torch.save(dict(model={n:v.detach().cpu() for n,v in model.state_dict().items()},architecture=saved['architecture'],stage=1,epoch=epoch,readout_step=step,training_stage='head_warm' if step<0 else ('fixed_optics_readout' if step else 'joint_training'),case=case,contract=contract,frontend_sha256=front_sha,parent_SHA256=digest,TEST_gradients=False),a.out/(role+'_checkpoint.pt'))
    def consider(report,epoch,step=0):
        nonlocal best,best_epoch,best_step
        score=rgb_priority_selection(report,cfg,domains)
        if score>best:best=score;best_epoch=epoch;best_step=step;save('best',epoch,step);put(a.out/'selected_validation.json',report);return True
        return False
    def guard():
        if (parent is not None and sha(parent)!=digest) or (teacher_path is not None and sha(teacher_path)!=teacher_digest) or tensor_sha(model.display_gains)!=gain or any(tensor_sha(model.state_dict()[n])!=v for n,v in frozen.items()) or any(sha(n)!=v for n,v in sources.items()):raise RuntimeError('Frozen parent/teacher/gain/source changed')
    save('best',0);save('last',0);put(a.out/'selected_validation.json',baseline)
    if cfg.get('head_warm_epochs',0):
        model.eval();unused,tx=infer(model,train,tf,device,True);fx=collect_augmented(model,flip,ff,device);unused,vx=infer(model,val,vf,device,True);mask=np.isin(train['domains'],domains);warmx=torch.as_tensor(np.concatenate((tx[mask],fx[mask])),device=device);warmy=torch.as_tensor(np.tile(train['labels'][mask],2),device=device);warmopt=torch.optim.Adam(model.shared_head.parameters(),lr=cfg['head_warm_lr'])
        for step in range(1,cfg['head_warm_epochs']+1):
            order=torch.randperm(len(warmy),device=device);total=0.
            for rows in order.split(128):
                warmopt.zero_grad();cost=F.cross_entropy(model.shared_head(warmx[rows]),warmy[rows],label_smoothing=cfg['label_smoothing'])+cfg['weight_regularization']*model.shared_head.weight.square().sum()/10;cost.backward();warmopt.step();total+=float(cost.detach())*len(rows)
            report=scores_from_logits(vx@model.shared_head.weight.detach().cpu().numpy().T,val);consider(report,0,-step);save('last',0,-step);history.append(dict(stage='head_warm',step=step,TRAIN_loss=total/len(warmy),validation=report['scores']));put(a.out/'history.json',history);put(a.out/'status.json',dict(state='head_warm',step=step,updated=time.time()));guard()
        del warmx,warmy
    optimizer=torch.optim.AdamW([dict(params=optics,lr=cfg['optical_lr']),dict(params=model.shared_head.parameters(),lr=cfg['head_lr'])],weight_decay=0.)
    for epoch in range(1,cfg['epochs']+1):
        model.train();total=0.;seen=0;relation_totals=[0.,0.];correct={d:0 for d in domains};counts={d:0 for d in domains};conflicts={n:0 for n,p in model.named_parameters() if 'phase' in n and p.requires_grad};projection_batches=0;rng=np.random.default_rng(cfg['seed']+epoch);factor=.1+.9*.5*(1+math.cos(math.pi*(epoch-1)/max(1,cfg['epochs']-1)))
        for group,lr in zip(optimizer.param_groups,(cfg['optical_lr'],cfg['head_lr'])):group['lr']=lr*factor
        for ix in (pair_batches(train,cfg['batch_size'],cfg['seed']+epoch) if cfg.get('paired_TRAIN_batches',False) else balanced_batches(train,domains,cfg['batch_size'],cfg['seed']+epoch)):
            if external:
                view=0 if rng.random()<cfg.get('TRAIN_original_probability',0.) else int(rng.integers(1 if cfg.get('TRAIN_original_probability',0.) else 0,8));field=amplitude(views[view],view_fronts[view],ix,device);target=teacher[view][ix]
            else:
                field=amplitude(train,tf,ix,device);fl=rng.random(len(ix))<cfg['TRAIN_flip_probability'];target=teacher[0][ix].clone()
                if fl.any():field[torch.as_tensor(fl,device=device)]=amplitude(flip,ff,ix[fl],device);target[torch.as_tensor(fl,device=device)]=teacher[1][ix[fl]]
            drop=rng.random(len(ix))<cfg['TRAIN_raw_dropout_probability']
            if drop.any():field[torch.as_tensor(drop,device=device)]=raw_region_dropout(field[torch.as_tensor(drop,device=device)])
            optimizer.zero_grad(set_to_none=True);r=model(field);y=torch.as_tensor(train['labels'][ix],device=device);d=torch.as_tensor(train['domains'][ix],device=device);temp=cfg['teacher_temperature'];per_sample=teacher_KD_rows(r['logits'],target,y,cfg);sample_weight=torch.where(d==0,cfg.get('RGB_loss_weight',1.),1.);cost=(per_sample*sample_weight).sum()/sample_weight.sum()+cfg['weight_regularization']*model.shared_head.weight.square().sum()/10
            if relational:
                rd,ra=relational_losses(r['ccd_features'],teacher_embeddings[view][ix],d)
                cost+=cfg['relation_distance_weight']*rd+cfg['relation_angle_weight']*ra
                relation_totals[0]+=float(rd.detach())*len(ix);relation_totals[1]+=float(ra.detach())*len(ix)
            if saved['architecture']=='moe':cost+=cfg['domain_balance_weight']*domain_balance(r['route_power'],d,model.active_indices[:4])
            if not torch.isfinite(cost):raise RuntimeError('Nonfinite shared-frontend TRAIN objective')
            if cfg.get('phase_gradient_projection',False):
                assert domains==[0,1] and saved['architecture']=='moe';parts=[];named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
                for domain in domains:
                    rows=d==domain;part=F.cross_entropy(r['logits'][rows],y[rows],label_smoothing=cfg['label_smoothing'])+cfg['teacher_KD_weight']*F.kl_div(F.log_softmax(r['logits'][rows]/temp,1),(target[rows]/temp).softmax(1),reduction='batchmean')*temp**2+cfg['weight_regularization']*model.shared_head.weight.square().sum()/10+cfg['domain_balance_weight']*domain_balance(r['route_power'][rows],d[rows],model.active_indices[:4]);parts.append(part)
                gr=torch.autograd.grad(parts[0],[p for n,p in named],retain_graph=True);gs=torch.autograd.grad(parts[1],[p for n,p in named])
                for (name,p),rgb_grad,sar_grad in zip(named,gr,gs):
                    if 'phase' in name:
                        merged,opposed=merge_phase_gradients(rgb_grad,sar_grad);conflicts[name]+=int(opposed)
                    else:merged=(rgb_grad+sar_grad)*.5
                    if not torch.isfinite(merged).all():raise RuntimeError('Nonfinite projected TRAIN gradient')
                    p.grad=merged.detach()
                projection_batches+=1
            else:cost.backward()
            torch.nn.utils.clip_grad_norm_(optics,1.);torch.nn.utils.clip_grad_norm_(model.shared_head.parameters(),5.);optimizer.step();total+=float(cost.detach())*len(ix);seen+=len(ix)
            for domain in domains:correct[domain]+=int(((r['logits'].argmax(1)==y)&(d==domain)).sum());counts[domain]+=int((d==domain).sum())
            if seen%1000<len(ix):put(a.out/'status.json',dict(state='joint_training',epoch=epoch,TRAIN_seen=seen,loss=total/seen,updated=time.time()))
        guard();report=infer(model,val,vf,device)
        if consider(report,epoch):last_improvement=epoch
        save('last',epoch);row=dict(stage='joint_training',epoch=epoch,TRAIN_loss=total/seen,TRAIN_accuracy_by_domain={str(d):correct[d]/counts[d] for d in domains},validation=report['scores'],best_epoch=best_epoch,head_norm=float(model.shared_head.weight.detach().norm()),TRAIN_phase_conflict_fraction={n:v/projection_batches for n,v in conflicts.items()} if projection_batches else None);history.append(row);put(a.out/'history.json',history);put(a.out/'status.json',dict(state='joint_training',**row,updated=time.time()))
        if cfg.get('trainable_phase_prefixes'):
            row['validation_routing_by_domain']=report.get('routing_by_domain');put(a.out/'history.json',history)
        if relational:
            row['TRAIN_relational_distance']=relation_totals[0]/seen;row['TRAIN_relational_angle']=relation_totals[1]/seen;put(a.out/'history.json',history)
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
    guard();locked=dict(best_sha256=sha(a.out/'best_checkpoint.pt'),last_sha256=sha(a.out/'last_checkpoint.pt'),TEST_loaded=False);put(a.out/'locked_before_TEST.json',locked);chosen=torch.load(a.out/'best_checkpoint.pt',map_location=device,weights_only=False);model.load_state_dict(chosen['model'],strict=True);test=load_arrays(a.data,'test');testfront=extract_features(test,a.frontend);report=infer(model,test,testfront,device);put(a.out/'selected_test.json',report);assert sha(a.out/'best_checkpoint.pt')==locked['best_sha256']
    put(a.out/'completion.json',dict(passed=True,contract=contract,case=case,architecture=saved['architecture'],best_epoch=best_epoch,best_step=best_step,epochs_completed=epochs_completed,test=report['scores'],best_sha256=locked['best_sha256'],last_sha256=locked['last_sha256'],frontend_SHA256=front_sha,phase_updates=True,TEST_gradients=False,TEST_teacher=False,selection_uses_TEST=False,SDK=False,independent_new_blind_TEST_claim=False,seconds=time.monotonic()-started));put(a.out/'status.json',dict(state='complete_CPU_audit_pending',updated=time.time()))


if __name__=='__main__':main()
