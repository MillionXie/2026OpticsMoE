"""Convex TRAIN D4 teacher fit, folded into the original7840 readout.

Independent fixed audited phase control. A teacher-wrong TRAIN target is not
distilled, but its original TRAIN label still contributes CE. No inference change.
"""
import argparse,json,time,subprocess
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .static_four import build_static_four,geometry,CONTRACT
from .augmented_teacher import load_teacher,teacher_logits
from .decoupled_distillation import supervised_decoupled
from .diagnose_frontend_transfer import checked_features
from .train_shared_frontend import d4_views,rgb_priority_selection
from .train_unified_static import load_arrays
from .train_sensor_neutral import infer,collect_augmented
from .refine_unified import conditioned_inputs,scores_from_logits
from LightGenV2.tasks.t21_continual_learning_hardware.train_native import sha,put,tensor_sha,gpu_gate


def supervised_KD(logits,targets,labels,temperature):
    ce=F.cross_entropy(logits,labels,reduction='none',label_smoothing=.02)
    correct=(targets.argmax(1)==labels).to(logits.dtype)
    kd=F.kl_div(F.log_softmax(logits/temperature,1),(targets/temperature).softmax(1),reduction='none').sum(1)*temperature**2
    return ce+correct*kd


def main():
    p=argparse.ArgumentParser(__doc__)
    for n in ('parent','teacher','data','frontend','profile','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--gpu-uuid',required=True);a=p.parse_args();cfg=json.loads(a.profile.read_text());assert any(cfg==json.loads(Path(__file__).with_name(n).read_text()) for n in ('config_static_four_teacher_readout_20261025.json','config_static_four_decoupled_readout_20261026.json'));decoupled=cfg.get('decoupled_distillation',False)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;gpu=gpu_gate(cfg,a.gpu_uuid);device=torch.device('cuda:0');path=a.parent/'best_checkpoint.pt';digest=sha(path);receipt=json.loads((a.parent/'completion.json').read_text());prior=json.loads((a.parent/'protocol.json').read_text());assert digest==receipt['best_sha256'] and not prior['selection_uses_TEST'];audit=json.loads((a.parent.parent/(a.parent.name+'_CPU')/'report.json').read_text());br=next(v for v in audit['rows'] if v['role']=='best');assert br['checkpoint_SHA256']==digest and br['comparison_to_GPU']['legacy_parity_passed']
    cp=torch.load(path,map_location='cpu',weights_only=False);assert cp['contract']==CONTRACT and cp['architecture']=='moe' and cp['frontend_sha256']==cfg['frontend_SHA256'];model=build_static_four('moe').to(device);model.load_state_dict(cp['model'],strict=True);model.eval().requires_grad_(False);frozen={n:tensor_sha(v) for n,v in model.state_dict().items() if n!='shared_head.weight'}
    if cfg.get('parent_expected_SHA256'):assert digest==cfg['parent_expected_SHA256']
    a.out.mkdir(parents=True,exist_ok=False);start=time.monotonic();tr=load_arrays(a.data,'train');va=load_arrays(a.data,'validation');tm,tcr=load_teacher(a.teacher,a.data,device);teacher_path=a.teacher/'best_checkpoint.pt';teacher_digest=sha(teacher_path)
    assert tcr['validation']['RGB']['accuracy']>=.85 and tcr['validation']['SAR']['accuracy']>=.72
    files=[Path(__file__),a.profile,Path(__file__).with_name('train_shared_frontend.py'),Path(__file__).with_name('augmented_teacher.py'),Path(__file__).with_name('static_four.py'),Path(__file__).with_name('hardware_aware.py'),Path(__file__).with_name('train_sensor_neutral.py'),Path('LightGenV2/tasks/t21_continual_learning_hardware/model.py')];sources={str(p):sha(p) for p in files}
    if decoupled:
        helper=Path(__file__).with_name('decoupled_distillation.py');sources[str(helper)]=sha(helper)
    proto=dict(contract=CONTRACT,config=cfg,method='Original7840_convex_TRAIN_D4_teacher_distillation',case='MoE_AB',architecture='moe',parent_run=str(a.parent),parent_SHA256=digest,teacher_run=str(a.teacher),teacher_SHA256=teacher_digest,teacher_scope='TRAIN/D4 only, discard teacher-wrong TRAIN KL; no TEST access',frontend_sha256=cfg['frontend_SHA256'],geometry=geometry(model),fixed_gain_sha256=tensor_sha(model.display_gains),frozen_state_sha256=frozen,training_domains=[0,1],phase_updates=False,original_head_parameters=7840,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256=sources,data_manifest_sha256=sha(a.data/'manifest.json'),selection_uses_TEST=False,TEST_gradients=False,TEST_teacher=False,SDK=False,gpu=gpu)
    put(a.out/'protocol.json',proto);views=d4_views(tr);xx=[];targets=[]
    if decoupled:
        proto.update(method='Original7840_decoupled_TRAIN_D4_teacher_distillation',DKD_target_weight=cfg['DKD_target_weight'],DKD_other_weight=cfg['DKD_other_weight']);put(a.out/'protocol.json',proto)
    for i,view in enumerate(views):
        front=checked_features(view,a.frontend,cfg['frontend_SHA256']);xx.append(collect_augmented(model,view,front,device));targets.append(teacher_logits(tm,view,device).double());put(a.out/'status.json',dict(state='TRAIN_D4_cache',view=i+1,total_views=8,updated=time.time()))
    del views,tm;torch.cuda.empty_cache();vf=checked_features(va,a.frontend,cfg['frontend_SHA256']);vr,vx=infer(model,va,vf,device,True);put(a.out/'baseline_validation.json',vr);x=np.concatenate(xx);del xx;z,zv,transform=conditioned_inputs(x,vx,cfg['conditioner_ridge']);del x;targets=torch.cat(targets);labels=torch.as_tensor(np.tile(tr['labels'],8));weights=torch.as_tensor(np.tile(np.where(tr['domains']==0,2.,1.),8));initial=model.shared_head.weight.detach().cpu().double();param=torch.nn.Parameter(torch.linalg.solve(transform,initial.T).T);opt=torch.optim.LBFGS([param],lr=1.,max_iter=5,history_size=10,line_search_fn='strong_wolfe');best=rgb_priority_selection(vr,cfg,[0,1]);best_step=0;history=[]
    def save(role,step):torch.save(dict(model={n:v.detach().cpu() for n,v in model.state_dict().items()},architecture='moe',stage=1,epoch=cp['epoch'],readout_step=step,case='MoE_AB',contract=CONTRACT,frontend_sha256=cfg['frontend_SHA256'],parent_SHA256=digest,TEST_gradients=False),a.out/(role+'_checkpoint.pt'))
    def guard():
        assert sha(path)==digest and sha(teacher_path)==teacher_digest and all(sha(n)==s for n,s in sources.items())
        assert all(tensor_sha(model.state_dict()[n])==s for n,s in frozen.items())
    save('best',0);save('last',0);put(a.out/'selected_validation.json',vr)
    for step in range(1,cfg['fit_steps']+1):
        def closure():
            opt.zero_grad();logits=z@param.T;v=(supervised_decoupled(logits,targets,labels,cfg['temperature'],cfg['DKD_target_weight'],cfg['DKD_other_weight']) if decoupled else supervised_KD(logits,targets,labels,cfg['temperature']));w=param@transform.T;cost=(v*weights).sum()/weights.sum()+cfg['coordinate_regularization']*param.square().sum()/10+cfg['weight_regularization']*w.square().sum()/10;cost.backward();return cost
        loss=float(opt.step(closure).detach());w=(param.detach()@transform.T).float();assert np.isfinite(loss) and torch.isfinite(w).all()
        with torch.no_grad():model.shared_head.weight.copy_(w.to(device))
        report=scores_from_logits(vx@w.numpy().T,va);key=rgb_priority_selection(report,cfg,[0,1])
        if key>best:best=key;best_step=step;save('best',step);put(a.out/'selected_validation.json',report)
        save('last',step);row=dict(step=step,TRAIN_loss=loss,validation=report['scores'],best_step=best_step);history.append(row);put(a.out/'history.json',history);put(a.out/'status.json',dict(state='fit',**row,updated=time.time()));guard()
        if time.monotonic()-start>cfg['max_seconds']:break
    locked=dict(best_sha256=sha(a.out/'best_checkpoint.pt'),last_sha256=sha(a.out/'last_checkpoint.pt'),TEST_loaded=False);put(a.out/'locked_before_TEST.json',locked);chosen=torch.load(a.out/'best_checkpoint.pt',map_location=device,weights_only=False);model.load_state_dict(chosen['model'],strict=True);test=load_arrays(a.data,'test');tf=checked_features(test,a.frontend,cfg['frontend_SHA256']);report=infer(model,test,tf,device);put(a.out/'selected_test.json',report);guard()
    put(a.out/'completion.json',dict(passed=True,contract=CONTRACT,case='MoE_AB',architecture='moe',best_step=best_step,test=report['scores'],best_sha256=locked['best_sha256'],last_sha256=locked['last_sha256'],frontend_SHA256=cfg['frontend_SHA256'],phase_updates=False,selection_uses_TEST=False,TEST_gradients=False,TEST_teacher=False,SDK=False,independent_new_blind_TEST_claim=False,seconds=time.monotonic()-start));put(a.out/'status.json',dict(state='complete_CPU_audit_pending',updated=time.time()))


if __name__=='__main__':main()
