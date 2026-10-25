"""TRAIN-only shared-CNN distillation and input-region dropout, original head.

No SAR optical/readout fitting in RGB-only case, no TEST teacher/augmentation.
The pretraining teacher has seen TRAIN from both sensors, explicitly disclosed.
TRAIN raw-region dropout is removed at inference; no new inference modules.
"""
import argparse,json,time,subprocess
from pathlib import Path
import torch
import numpy as np
from torch.nn import functional as F
from .domain_frontend_probe import features,FRONTEND_SHA,CONTRACT
from .unified_native import build_unified
from .train_unified_static import load_arrays,selection
from .train_sensor_neutral import infer,collect_augmented
from .sensor_neutral import amplitude
from .refine_unified import conditioned_inputs,scores_from_logits
from LightGenV2.tasks.t21_continual_learning_hardware.train_native import sha,put,tensor_sha,gpu_gate


def raw_region_dropout(field):
    x=field.clone();x[:,:84,:]=0;x[:,112:196,:112]=0
    return x/x.square().sum((-2,-1),keepdim=True).clamp_min(1e-20).sqrt()


@torch.no_grad()
def dropout_cache(model,data,front,device):
    rows=[]
    for start in range(0,len(data['labels']),8):
        ix=np.arange(start,min(start+8,len(data['labels'])));rows.append(model(raw_region_dropout(amplitude(data,front,ix,device)))['ccd_features'].cpu().numpy())
    return np.concatenate(rows)


def main():
    p=argparse.ArgumentParser(__doc__)
    for n in ('parent','data','frontend','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--gpu-uuid',required=True);a=p.parse_args();cfg=json.loads(Path(__file__).with_name('config_robust_readout_20261025.json').read_text());assert cfg['fit_steps']==40 and not cfg['TEST_gradients'] and not cfg['TEST_teacher'] and not cfg['phase_updates'];torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;gpu=gpu_gate(cfg,a.gpu_uuid);device=torch.device('cuda:0');path=a.parent/'best_checkpoint.pt';digest=sha(path);receipt=json.loads((a.parent/'completion.json').read_text());old=json.loads((a.parent/'protocol.json').read_text());assert digest==receipt['best_sha256'] and receipt['frontend_SHA256']==FRONTEND_SHA and not old['selection_uses_TEST'];cp=torch.load(path,map_location='cpu',weights_only=False);model=build_unified(cp['architecture']).to(device);model.load_state_dict(cp['model'],strict=True);model.eval().requires_grad_(False);domains=old['training_domains'];case=old['case'];frozen={n:tensor_sha(v) for n,v in model.state_dict().items() if n!='shared_head.weight'};a.out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    put(a.out/'protocol.json',dict(contract=CONTRACT,method='TRAIN_same_CNN_teacher_and_raw_region_dropout',config=cfg,case=case,architecture=cp['architecture'],parent_run=str(a.parent),parent_SHA256=digest,frontend_sha256=FRONTEND_SHA,fixed_gain_sha256=tensor_sha(model.display_gains),frozen_state_sha256=frozen,training_domains=domains,teacher_scope='Only original TRAIN RGB+SAR auxiliary frontend pretraining, student optical/readout own declared domains only',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256=sha(__file__),data_manifest_sha256=sha(a.data/'manifest.json'),SDK=False,TEST_gradients=False,TEST_teacher=False,selection_uses_TEST=False,gpu=gpu))
    train=load_arrays(a.data,'train');val=load_arrays(a.data,'validation');tf=features(train,a.frontend);vf=features(val,a.frontend);flip=dict(train,images=train['images'][:,:,::-1,:].copy());ff=features(flip,a.frontend);unused,tx=infer(model,train,tf,device,True);vr,vx=infer(model,val,vf,device,True);fx=collect_augmented(model,flip,ff,device);dx=dropout_cache(model,train,tf,device);dfx=dropout_cache(model,flip,ff,device);mask=np.isin(train['domains'],domains);x=np.concatenate([v[mask] for v in (tx,fx,dx,dfx)]);labels=torch.as_tensor(np.tile(train['labels'][mask],4));z,zv,t=conditioned_inputs(x,vx,cfg['conditioner_ridge']);frontcp=torch.load(a.frontend,map_location='cpu',weights_only=False)['model'];teacher=[F.linear(f.float(),frontcp['head.weight'][:10],frontcp['head.bias'][:10])[mask].double() for f in (tf,ff,tf,ff)];teacher=torch.cat(teacher);temp=cfg['teacher_temperature'];target=(teacher/temp).softmax(1);initial=model.shared_head.weight.detach().cpu().double();param=torch.nn.Parameter(torch.linalg.solve(t,initial.T).T);optimizer=torch.optim.LBFGS([param],lr=1.,max_iter=5,history_size=10,line_search_fn='strong_wolfe');best=selection(vr,domains);best_step=0;history=[]
    def save(role,step):
        torch.save(dict(model={n:v.detach().cpu() for n,v in model.state_dict().items()},architecture=cp['architecture'],stage=1,epoch=cp['epoch'],readout_step=step,case=case,contract=CONTRACT,frontend_sha256=FRONTEND_SHA,parent_SHA256=digest,TEST_gradients=False),a.out/(role+'_checkpoint.pt'))
    save('best',0);put(a.out/'baseline_validation.json',vr)
    for step in range(1,cfg['fit_steps']+1):
        def closure():
            optimizer.zero_grad();logits=z@param.T;weight=param@t.T;cost=F.cross_entropy(logits,labels,label_smoothing=cfg['label_smoothing'])+cfg['teacher_KD_weight']*F.kl_div(F.log_softmax(logits/temp,1),target,reduction='batchmean')*temp**2+cfg['coordinate_regularization']*param.square().sum()/10+cfg['weight_regularization']*weight.square().sum()/10;cost.backward();return cost
        loss=float(optimizer.step(closure).detach());w=(param.detach()@t.T).float()
        if not np.isfinite(loss) or not torch.isfinite(w).all():raise RuntimeError('Nonfinite robust readout')
        with torch.no_grad():model.shared_head.weight.copy_(w.to(device))
        report=scores_from_logits(vx@w.numpy().T,val);score=selection(report,domains)
        if score>best:best=score;best_step=step;save('best',step);put(a.out/'selected_validation.json',report)
        save('last',step);history.append(dict(step=step,TRAIN_loss=loss,validation=report['scores'],best_step=best_step));put(a.out/'history.json',history);put(a.out/'status.json',dict(state='robust_readout_fit',**history[-1],updated=time.time()))
    if sha(path)!=digest or any(tensor_sha(model.state_dict()[n])!=v for n,v in frozen.items()):raise RuntimeError('Frozen parent changed')
    locked=dict(best_sha256=sha(a.out/'best_checkpoint.pt'),last_sha256=sha(a.out/'last_checkpoint.pt'),TEST_loaded=False);put(a.out/'locked_before_TEST.json',locked);selected=torch.load(a.out/'best_checkpoint.pt',map_location=device,weights_only=False);model.load_state_dict(selected['model'],strict=True);test=load_arrays(a.data,'test');testfront=features(test,a.frontend);report=infer(model,test,testfront,device);put(a.out/'selected_test.json',report);assert sha(a.out/'best_checkpoint.pt')==locked['best_sha256']
    put(a.out/'completion.json',dict(passed=True,contract=CONTRACT,method='TRAIN_only_distillation_and_raw_region_dropout',case=case,architecture=cp['architecture'],best_step=best_step,test=report['scores'],best_sha256=locked['best_sha256'],last_sha256=locked['last_sha256'],frontend_SHA256=FRONTEND_SHA,phase_updates=False,TEST_gradients=False,TEST_teacher=False,selection_uses_TEST=False,SDK=False,independent_new_blind_TEST_claim=False,seconds=time.monotonic()-started));put(a.out/'status.json',dict(state='complete_CPU_audit_pending',updated=time.time()))


if __name__=='__main__':main()
