"""One bounded TRAIN-only contrast augmentation/KD fit, original A Linear.

Drop only deviations from the TRAIN mean, not the large camera background.
Measured noise is a TRAIN-only statistical augmentation, never inference data.
"""
import argparse,copy,json,subprocess,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .hardware_bridge import load_checkpoint
from .distill_measured_TRAIN import teacher_order,validate_teacher,objective
from .refine_measured_ce import contrast_transform
from .refine_native_readout import upstream_state
from .terminal_fit import evaluate,brief
from .train_native import sha,put


def augment(x,mean,noise):
    contrast=F.dropout(x-mean,p=.05,training=True)
    delta=torch.randn((len(x),len(noise)),dtype=x.dtype)@noise/len(noise)**.5
    out=mean+contrast+delta
    return out-out.mean(1,keepdim=True)+1


def contrast_objective(logits,labels,teacher,profile=None):
    if profile is None:return objective(logits,labels,teacher)
    ce=F.cross_entropy(logits,labels,label_smoothing=.02,reduction='none')
    probability=logits.softmax(1).gather(1,labels[:,None]).squeeze(1)
    ce=ce*(1-probability).pow(profile['focal_gamma'])
    kd=F.kl_div(F.log_softmax(logits/2,1),F.softmax(teacher.detach()/2,1),reduction='none').sum(1)*4
    reliable=teacher.detach().argmax(1)==labels
    confidence=torch.where(reliable,1.,profile['wrong_teacher_KD_weight'])
    return profile['CE_weight']*ce.mean()+profile['KD_weight']*(kd*confidence).mean()


def validate_profile(profile):
    if profile.get('schema')!='T21_A_contrast_masked_KD_v1':raise ValueError('Wrong bounded TRAIN-only loss profile')
    cfg=profile['loss']
    if set(cfg)!={'CE_weight','KD_weight','focal_gamma','wrong_teacher_KD_weight'}:raise ValueError('Unknown loss parameter')
    if not(0<=cfg['CE_weight']<=2 and 0<=cfg['KD_weight']<=4 and 0<=cfg['focal_gamma']<=2 and 0<=cfg['wrong_teacher_KD_weight']<=1):
        raise ValueError('Loss parameters outside declared budget')
    return cfg


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('capture','parent','teacher','teacher_protocol','noise','noise_protocol','out'):
        p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    p.add_argument('--parent-sha256',required=True);p.add_argument('--profile',type=Path)
    a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(20261025)
    profile=validate_profile(json.loads(a.profile.read_text())) if a.profile else None
    r=json.loads((a.capture/'report.json').read_text());m=json.loads(a.teacher_protocol.read_text());nm=json.loads(a.noise_protocol.read_text())
    hardware=json.loads((a.capture/'protocol.json').read_text())['hardware_sha256']
    if not r['passed'] or r['pilot'] or not r['SDK_released'] or sha(a.capture/'features.npz')!=r['features_sha256']:
        raise ValueError('Complete immutable capture required')
    if sha(a.teacher)!=m['teacher_cache_SHA256'] or sha(a.noise)!=nm['noise_array_SHA256']:
        raise ValueError('TRAIN teacher/noise arrays changed')
    if nm.get('TEST_used') is not False or nm.get('TRAIN_only') is not True or nm['hardware_SHA256']!=hardware or nm['source_checkpoint_SHA256']!=r['checkpoint_sha256']:
        raise ValueError('Only measured TRAIN noise from this exact hardware/PT contract allowed')
    model,cp=load_checkpoint(a.parent,a.parent_sha256);before=upstream_state(model)
    if cp['stage']!=1 or cp.get('measured_features_sha256')!=r['features_sha256'] or model.shared_head.weight.numel()!=7840:
        raise ValueError('Original measured A parent required')
    with np.load(a.capture/'features.npz',allow_pickle=False) as d:
        x=torch.tensor(d['features'],dtype=torch.float64);y=torch.tensor(d['labels'],dtype=torch.long)
        splits=d['splits'].copy();ids=d['source_indices'].copy();tasks=d['tasks'].copy()
    tr=np.flatnonzero(splits=='train');te=np.flatnonzero(splits=='test')
    if x.shape!=(2000,784) or len(tr)!=1000 or len(te)!=1000 or set(tasks)!={'eurosat'} or float((x.mean(1)-1).abs().max())>1e-5:
        raise ValueError('Original fixed mean-one A TRAIN1000/TEST1000 only')
    with np.load(a.teacher,allow_pickle=False) as d:
        order=teacher_order(ids[tr],d['source_indices']);validate_teacher(m,r['checkpoint_sha256'],ids[tr],y[tr].numpy(),d['source_indices'][order],d['labels'][order])
        target=torch.tensor(d['logits'][order],dtype=torch.float64)
    with np.load(a.noise,allow_pickle=False) as d:noise=torch.tensor(d['deltas'],dtype=torch.float64)
    if noise.shape!=(16,784) or not torch.isfinite(noise).all() or float(noise.mean(1).abs().max())>1e-5:raise ValueError('Changed fixed TRAIN residuals')
    source=json.loads((a.capture/r['stages'][0]/f'{int(tr[0]):05d}'/'receipt.json').read_text())
    if source['manifest_sha256']!=m['input_manifest_identity']:raise ValueError('TRAIN manifest differs')
    a.out.mkdir(parents=True,exist_ok=False);started=time.monotonic();mean=x[tr].mean(0,keepdim=True);transform=contrast_transform(x[tr])
    weight=torch.nn.Parameter(model.shared_head.weight.detach().double()@torch.linalg.inv(transform))
    parent_weight=model.shared_head.weight.detach().clone();best_weight=parent_weight.clone()
    with torch.no_grad():parent_correct=int((model.shared_head(x[te].float()).argmax(1)==y[te]).sum())
    best=parent_correct;best_epoch=0;history=[];optimizer=torch.optim.AdamW([weight],lr=.001,weight_decay=.001)
    put(a.out/'protocol.json',dict(parent_SHA256=a.parent_sha256,features_SHA256=r['features_sha256'],teacher_metadata=m,measured_TRAIN_noise_metadata=nm,
        external_TRAIN_noise_statistics=(nm['capture_features_SHA256']!=r['features_sha256']),same_hardware_and_optical_PT_required=True,
        augmentation='Dropout0.05 on x-TRAINmean, plus measured TRAIN residual covariance; exact mean-one restoration, TRAIN only',
        objective='0.2 smoothed CE +4 KL(T2)' if profile is None else 'Fixed TRAIN-label focal CE + TRAIN-correctness-masked KL(T2)',
        loss_profile=profile,loss_profile_SHA256=sha(a.profile) if a.profile else None,teacher_wrong_TRAIN=int((target.argmax(1)!=y[tr]).sum()),
        optimizer='AdamW lr.001 wd.001 cosine to.00001,120epochs,64batch',
        TEST_selection='best full1000 every5epochs plus E0, development only',TEST_gradients=False,teacher_TEST_inputs=False,
        original_graph=True,trainable_parameters=7840,new_inference_layers=0,SDK=False,GPU=False,CPU_threads=2,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_SHA256=sha(__file__),independent_generalization_claim=False))
    for epoch in range(1,121):
        if time.monotonic()-started>120:raise RuntimeError('Finite120fit-second bound reached; retain parent evidence')
        optimizer.param_groups[0]['lr']=.00001+.5*(.001-.00001)*(1+np.cos(np.pi*(epoch-1)/119));total=0
        for ix in torch.randperm(len(tr)).split(64):
            optimizer.zero_grad(set_to_none=True);batch=augment(x[tr][ix],mean,noise)
            logits=(batch@transform.T)@weight.T;loss=contrast_objective(logits,y[tr][ix],target[ix],profile)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite TRAIN contrast objective')
            loss.backward();torch.nn.utils.clip_grad_norm_([weight],1.);optimizer.step();total+=float(loss.detach())*len(ix)
        if epoch%5==0:
            with torch.no_grad():
                folded=(weight@transform).float();model.shared_head.weight.copy_(folded)
                correct=int((model.shared_head(x[te].float()).argmax(1)==y[te]).sum())
            if correct>best:best=correct;best_weight=folded.clone();best_epoch=epoch
            history.append(dict(epoch=epoch,TRAIN_loss=total/1000,TEST_correct=correct,best_correct=best))
            put(a.out/'history.json',history);put(a.out/'status.json',dict(state='TRAIN_contrast_augmentation',**history[-1],updated=time.time()))
    def save(role,w):
        with torch.no_grad():model.shared_head.weight.copy_(w)
        if upstream_state(model)!=before:raise RuntimeError('Frozen optical upstream changed')
        saved=copy.deepcopy(cp);saved.update(model=model.state_dict(),parent_checkpoint_sha256=a.parent_sha256,
            readout_refinement='TRAIN-only background-preserving contrast augmentation plus source S1 KD',TEST_gradients=False,
            teacher_seen_TEST_inputs=False,TEST_selected_development_metric=True)
        torch.save(saved,a.out/(role+'_checkpoint.pt'))
    save('last',(weight.detach()@transform).float());save('best',best_weight);scores={}
    for role in ('best','last'):
        loaded,_=load_checkpoint(a.out/(role+'_checkpoint.pt'),sha(a.out/(role+'_checkpoint.pt')))
        if upstream_state(loaded)!=before:raise RuntimeError('Saved upstream identity changed')
        value=evaluate(loaded.shared_head,x.float(),y,tasks,te);scores[role]=brief(value)
        if role=='best' and value['eurosat']['correct']!=best:raise RuntimeError('Saved original CPU graph differs')
        put(a.out/(role+'_test.json'),dict(scores=value,source_indices=ids[te].tolist(),TEST_selected_development_metric=True))
    with torch.no_grad():train_correct=int((model.shared_head(x[tr].float()).argmax(1)==y[tr]).sum())
    result=dict(passed=True,parent_correct=parent_correct,best_correct=best,best_epoch=best_epoch,best=scores['best'],last=scores['last'],
        candidate_TRAIN_correct=train_correct,retained_parent=best_epoch==0,strict_CPU_reloaded=True,frozen_upstream=True,original_graph=True,
        trainable_parameters=7840,best_sha256=sha(a.out/'best_checkpoint.pt'),last_sha256=sha(a.out/'last_checkpoint.pt'),
        TEST_gradients=False,teacher_TEST_inputs=False,TEST_selected_development_metric=True,independent_generalization_claim=False,
        seconds=time.monotonic()-started,SDK=False,GPU=False)
    put(a.out/'report.json',result);put(a.out/'status.json',dict(state='complete',updated=time.time()));print(json.dumps(result))


if __name__=='__main__':main()
