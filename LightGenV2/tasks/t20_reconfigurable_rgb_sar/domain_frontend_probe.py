"""Controlled shared-frontend transfer through frozen native optical phases.

Same32128 CNN structure as T21, but TRAIN-adapted weights; both optical models
share that exact checkpoint. RGB-only refers to optical/readout fitting only:
the shared frontend has seen both TRAIN domains. Not an unseen-SAR system claim.
"""
import argparse,json,time,subprocess
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from LightGenV2.tasks.t09_multimodal_matching.vision import VisionEncoder
from LightGenV2.tasks.t21_continual_learning_hardware.train_native import sha,put,gpu_gate,tensor_sha
from .unified_native import build_unified
from .sensor_neutral import amplitude
from .train_unified_static import load_arrays,selection
from .train_sensor_neutral import infer,collect_augmented
from .refine_unified import conditioned_inputs,scores_from_logits

FRONTEND_SHA='fb7d2d52e8ca94fb92d5ee9b81cabb7c7a573d21e405fe96e973140f6803436d'
CONTRACT='T20_sensor_neutral_domain_frontend_native8_v3'


@torch.no_grad()
def features(data,path):
    if sha(path)!=FRONTEND_SHA:raise ValueError('Exact locked TRAIN-only shared frontend required')
    cp=torch.load(path,map_location='cpu',weights_only=False)
    if cp['TEST_access'] is not False or cp['config']['feature_parameters']!=32128:raise ValueError('Wrong auxiliary architecture/provenance')
    model=VisionEncoder();model.load_state_dict(cp['model'],strict=True);model.eval().requires_grad_(False);result=[]
    for start in range(0,len(data['labels']),128):result.append(model(torch.from_numpy(data['images'][start:start+128]))[0].to(torch.float16))
    return torch.cat(result)


def configuration(profile=None):
    profile=Path(__file__).with_name('config_domain_frontend_probe_20261025.json') if profile is None else Path(profile)
    if profile.resolve() not in {Path(__file__).with_name(n).resolve() for n in ('config_domain_frontend_probe_20261025.json','config_domain_frontend_weightnorm_20261025.json')}:raise ValueError('Published controlled profile required')
    cfg=json.loads(profile.read_text())
    if cfg['frontend_SHA256']!=FRONTEND_SHA or cfg['TEST_gradients'] or cfg['TEST_teacher'] or cfg['phase_updates'] or cfg['fit_steps']!=40:raise ValueError('Published finite shared-frontend control required')
    return cfg


def main():
    p=argparse.ArgumentParser(__doc__)
    for n in ('parent','data','frontend','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--profile',type=Path);p.add_argument('--gpu-uuid',required=True);a=p.parse_args();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;cfg=configuration(a.profile);gpu=gpu_gate(cfg,a.gpu_uuid);device=torch.device('cuda:0');parent=a.parent/'best_checkpoint.pt';digest=sha(parent);receipt=json.loads((a.parent/'completion.json').read_text());old=json.loads((a.parent/'protocol.json').read_text())
    if digest!=receipt['best_sha256'] or old['selection_uses_TEST']:raise ValueError('Locked parent required')
    cp=torch.load(parent,map_location='cpu',weights_only=False);model=build_unified(cp['architecture']).to(device);model.load_state_dict(cp['model'],strict=True);model.eval();domains=old['training_domains'];case=old['case'];train=load_arrays(a.data,'train');val=load_arrays(a.data,'validation');tf=features(train,a.frontend);vf=features(val,a.frontend);cal=np.concatenate([np.flatnonzero((train['domains']==d)&(train['labels']==c))[:2] for c in range(10) for d in domains])
    with torch.no_grad():gains=model.calibrate_gains(amplitude(train,tf,cal,device))
    frozen={n:tensor_sha(v) for n,v in model.state_dict().items() if n!='shared_head.weight'};model.requires_grad_(False);a.out.mkdir(parents=True,exist_ok=False);started=time.monotonic();put(a.out/'protocol.json',dict(config=cfg,contract=CONTRACT,case=case,architecture=cp['architecture'],parent_SHA256=digest,parent_run=str(a.parent),frontend_sha256=FRONTEND_SHA,shared_frontend_pretraining='Both audited TRAIN RGB+SAR; no TEST or TEST augmentation',RGB_only_scope='Optical/readout TRAIN domain only, not entire system pretraining',training_domains=domains,fixed_gain_sha256=tensor_sha(model.display_gains),gain_calibration=gains,frozen_state_sha256=frozen,dataset_manifest_sha256=sha(a.data/'manifest.json'),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256=sha(__file__),SDK=False,selection_uses_TEST=False,TEST_gradients=False,TEST_teacher=False,gpu=gpu))
    baseline,tx=infer(model,train,tf,device,True);vr,vx=infer(model,val,vf,device,True);flip=dict(train,images=train['images'][:,:,::-1,:].copy());ff=features(flip,a.frontend);fx=collect_augmented(model,flip,ff,device);mask=np.isin(train['domains'],domains);x=np.concatenate((tx[mask],fx[mask]));labels=torch.as_tensor(np.tile(train['labels'][mask],2));z,zv,t=conditioned_inputs(x,vx,cfg['conditioner_ridge']);weight=model.shared_head.weight.detach().cpu().double();param=torch.nn.Parameter(torch.linalg.solve(t,weight.T).T);optimizer=torch.optim.LBFGS([param],lr=1.,max_iter=5,history_size=10,line_search_fn='strong_wolfe');best=selection(vr,domains);best_step=0;history=[]
    def save(role,step):
        torch.save(dict(model={n:v.detach().cpu() for n,v in model.state_dict().items()},architecture=cp['architecture'],stage=1,epoch=cp['epoch'],readout_step=step,case=case,contract=CONTRACT,frontend_sha256=FRONTEND_SHA,parent_SHA256=digest,TEST_gradients=False),a.out/(role+'_checkpoint.pt'))
    save('best',0);put(a.out/'baseline_validation.json',vr)
    for step in range(1,cfg['fit_steps']+1):
        def closure():
            optimizer.zero_grad()
            cost=F.cross_entropy(z@param.T,labels,label_smoothing=cfg['label_smoothing'])+cfg['regularization']*param.square().sum()/10
            cost=cost+cfg.get('weight_regularization',0.)*(param@t.T).square().sum()/10
            cost.backward();return cost
        loss=float(optimizer.step(closure).detach());w=(param.detach()@t.T).float()
        if not np.isfinite(loss) or not torch.isfinite(w).all():raise RuntimeError('Nonfinite readout fit')
        with torch.no_grad():model.shared_head.weight.copy_(w.to(device))
        report=scores_from_logits(vx@w.numpy().T,val);score=selection(report,domains)
        if score>best:best=score;best_step=step;save('best',step);put(a.out/'selected_validation.json',report)
        save('last',step);history.append(dict(step=step,TRAIN_loss=loss,validation=report['scores'],best_step=best_step));put(a.out/'history.json',history);put(a.out/'status.json',dict(state='readout_fit',**history[-1],updated=time.time()))
    if any(tensor_sha(model.state_dict()[n])!=v for n,v in frozen.items()) or sha(parent)!=digest:raise RuntimeError('Frozen optical parent changed')
    locked=dict(best_sha256=sha(a.out/'best_checkpoint.pt'),last_sha256=sha(a.out/'last_checkpoint.pt'),TEST_loaded=False);put(a.out/'locked_before_TEST.json',locked);saved=torch.load(a.out/'best_checkpoint.pt',map_location=device,weights_only=False);model.load_state_dict(saved['model'],strict=True);test=load_arrays(a.data,'test');testfront=features(test,a.frontend);report=infer(model,test,testfront,device);put(a.out/'selected_test.json',report)
    if sha(a.out/'best_checkpoint.pt')!=locked['best_sha256']:raise RuntimeError('Locked weight changed')
    put(a.out/'completion.json',dict(passed=True,case=case,contract=CONTRACT,architecture=cp['architecture'],best_step=best_step,test=report['scores'],best_sha256=locked['best_sha256'],last_sha256=locked['last_sha256'],frontend_SHA256=FRONTEND_SHA,phase_updates=False,TEST_gradients=False,TEST_teacher=False,selection_uses_TEST=False,SDK=False,independent_new_blind_TEST_claim=False,seconds=time.monotonic()-started));put(a.out/'status.json',dict(state='complete_CPU_audit_pending',updated=time.time()))


if __name__=='__main__':main()
