"""Disclosed fixed-stage simulation-to-measured-feature response transfer.

The original frozen S1 probabilities are unchanged. Only authorized nonidentity
TEST derivatives enter assistant fitting; this transfers TEST information, even
though the original Linear student is recovered from TRAIN queries alone.
"""
import argparse,copy,json,subprocess,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .train_native import put,sha
from .hardware_bridge import load_checkpoint
from .refine_native_readout import upstream_state
from .terminal_fit import evaluate,brief
from .fit_stage_balanced_teacher import derivatives
from .fit_gated_linear_teacher import DISCLOSURE,full_covariance_transform,paired_indices,paired_margin_loss
from .refine_measured_ce import contrast_transform
from .distill_capacity_matched_stage import TRAIN_queries
from .audit_student_transfer import linear_projection
from .distill_linear_consensus import D_development_bands
from .distill_margin_teacher import whole_stage_band_key


def aligned_simulation_probabilities(source_ids,source_labels,original_ids,score):
    """Order by sealed source identities, never shuffled capture row order."""
    ids=np.asarray(original_ids);wanted=np.asarray(source_ids)
    if len(ids)!=1000 or len(set(ids.tolist()))!=1000 or len(wanted)!=1000 or set(ids.tolist())!=set(wanted.tolist()):raise ValueError('Exact1000 source identity set required')
    probabilities=np.asarray(score['probabilities'],np.float64);labels=np.asarray(score['labels'])
    if probabilities.shape!=(1000,10) or labels.shape!=(1000,) or not np.isfinite(probabilities).all() or (probabilities<0).any() or not np.allclose(probabilities.sum(1),1,atol=2e-6):raise ValueError('Valid original ten-output probabilities required')
    lookup={int(v):i for i,v in enumerate(ids)};order=np.array([lookup[int(v)] for v in wanted])
    if not np.array_equal(labels[order],source_labels):raise ValueError('Source labels differ; do not infer row order')
    return probabilities[order]


def equal_task_soft_KL(logits,probabilities,tasks,names):
    # No target accuracy, correctness filter, label alteration or ceiling term.
    rows=F.kl_div(F.log_softmax(logits,1),probabilities.detach(),reduction='none').sum(1)
    return torch.stack([rows[tasks==n].mean() for n in names]).mean()


def stage_contract(cfg):
    if cfg.get('architecture')=='moe' and cfg.get('stage')==4:
        return 'moe',4,dict(eurosat=(788,838),clevr=(714,764),speech_binary=(711,761),physical_binary_raw=(725,775)),'best_checkpoint.pt'
    if cfg.get('architecture')=='moe' and cfg.get('stage')==3:
        return 'moe',3,dict(eurosat=(706,756),clevr=(696,746),speech_binary=(715,765)),'student_checkpoint.pt'
    if 'architecture' not in cfg and 'stage' not in cfg:return 'd2nn',4,D_development_bands(),'best_checkpoint.pt'
    raise ValueError('Only exact LG C/D or legacy D2NN D profiles')


def gradient_conflicts(weight,z,qy,qt,dz,dy,dt,names):
    rows={}
    for name in names:
        a=torch.as_tensor(qt==name);b=torch.as_tensor(dt==name)
        ta=F.cross_entropy(z[a]@weight.T,qy[a]);db=F.cross_entropy(dz[b]@weight.T,dy[b])
        gt=torch.autograd.grad(ta,weight)[0];gd=torch.autograd.grad(db,weight)[0]
        rows[name]=dict(TRAIN_CE=float(ta.detach()),derivative_CE=float(db.detach()),TRAIN_gradient_norm=float(gt.norm()),derivative_gradient_norm=float(gd.norm()),gradient_cosine=float((gt*gd).sum()/(gt.norm()*gd.norm()).clamp_min(1e-30)))
    return rows


def derivative_pair_deltas(features,ids,labels,tasks,source_count,names):
    """Keep each authorized nonidentity view's original positive/negative pairs."""
    if len(features)%source_count:raise ValueError('Complete derivative views required')
    result={}
    for name in names:
        if name=='eurosat':continue
        rows=[]
        for start in range(0,len(features),source_count):
            end=start+source_count
            pairs=paired_indices(ids[start:end],labels[start:end].numpy(),tasks[start:end])
            rows.extend(features[start+p]-features[start+n] for n,p in pairs if tasks[start+n]==name)
        if not rows:raise ValueError('Missing binary pairs')
        result[name]=torch.stack(rows)
    return result


def equal_or_weighted_task_mean(values,tasks,names,weights):
    factors=[float(weights.get(n,1.)) for n in names]
    if any(v<=0 for v in factors):raise ValueError('Positive fixed task weights required')
    return sum(v*values[tasks==n].mean() for n,v in zip(names,factors))/sum(factors)


def cross_task_gradient_cosines(weight,features,labels,tasks,names):
    gradients={n:torch.autograd.grad(F.cross_entropy(features[tasks==n]@weight.T,labels[tasks==n]),weight)[0] for n in names}
    return {n:{m:float((a*b).sum()/(a.norm()*b.norm()).clamp_min(1e-30)) for m,b in gradients.items()} for n,a in gradients.items()}


def main():
    p=argparse.ArgumentParser(__doc__)
    for n in ('capture','package','warm_run','profile','out'):p.add_argument('--'+n.replace('_','-'),type=Path,required=True)
    p.add_argument('--acknowledge-test-derived-development',action='store_true');a=p.parse_args()
    if not a.acknowledge_test_derived_development:raise ValueError('Explicit inherited TEST information acknowledgment required')
    cfg=json.loads(a.profile.read_text());profiles=('config_D2NN_D_simulation_response_20261026.json','config_D2NN_D_anchored_response_20261026.json','config_LG_C_simulation_response_20261026.json','config_LG_C_weak_source_response_20261026.json','config_LG_D_weak_source_response_20261026.json','config_LG_D_soft_response_20261026.json','config_LG_D_low_TRAIN_anchor_20261026.json','config_LG_D_paired_response_20261026.json','config_LG_D_Physical_retention_20261026.json')
    if not any(cfg==json.loads(Path(__file__).with_name(n).read_text()) for n in profiles):raise ValueError('Only fixed published finite response-transfer profile')
    torch.set_num_threads(2);torch.manual_seed(cfg['seed']);package=json.loads(a.package.read_text());cr=json.loads((a.capture/'report.json').read_text());wr=json.loads((a.warm_run/'report.json').read_text())
    if not cr['passed'] or cr['pilot'] or not cr['SDK_released'] or sha(a.capture/'features.npz')!=cr['features_sha256'] or not wr['passed'] or not wr['strict_CPU_reloaded']:raise ValueError('Completed immutable capture/warm student required')
    architecture,stage,bands,warm_name=stage_contract(cfg);warm_sha=wr['best_sha256'] if cfg['schema']=='T21_LG_D_disclosed_weak_source_joint_response_20261026' else wr['student_sha256'];model,cp=load_checkpoint(a.warm_run/warm_name,warm_sha);prefix=upstream_state(model);cpu=package['cpu_report'];names=tuple(bands)
    if cp['architecture']!=architecture or cp['stage']!=stage or cp['measured_features_sha256']!=cr['features_sha256'] or package['checkpoint_sha256']!=cpu['checkpoint_sha256'] or not cpu['strict_CPU']:raise ValueError('Frozen measured stage and exact S1 reference required')
    if package['checkpoint_sha256']!=cfg['frozen_S1_SHA256'] or cr['checkpoint_sha256']!=package['checkpoint_sha256']:raise ValueError('S1 optical identity changed')
    with np.load(a.capture/'features.npz',allow_pickle=False) as d:x=torch.tensor(d['features']);y=torch.tensor(d['labels'],dtype=torch.long);tasks=d['tasks'].copy();ids=d['source_indices'].copy();splits=d['splits'].copy()
    tr=np.flatnonzero(splits=='train');te=np.flatnonzero(splits=='test')
    if len(tr)!=1000+300*(stage-1) or len(te)!=1000*stage:raise ValueError('Complete fixed-stage source pool required')
    targets=torch.zeros((len(x),10),dtype=torch.float64);source_map={}
    for name in names:
        ix=te[tasks[te]==name];dataset=next(v for v in package['datasets'] if v['task']==name and v['split']=='test')
        if sha(dataset['path'])!=dataset['sha256']:raise ValueError('Original data SHA mismatch')
        with np.load(dataset['path'],allow_pickle=False) as data:original_ids=data['source_indices'].copy()
        targets[ix]=torch.tensor(aligned_simulation_probabilities(ids[ix],y[ix].numpy(),original_ids,cpu['scores'][name]));source_map[name]=dict(ids=ids[ix].tolist(),dataset_SHA256=dataset['sha256'],S1_correct=cpu['scores'][name]['correct'])
    a.out.mkdir(parents=True,exist_ok=False);started=time.monotonic();transform,conditioning=full_covariance_transform(x[tr],contrast_transform(x[tr]));query,qt=TRAIN_queries(x[tr],tasks[tr],cfg);qy=y[tr].repeat(1+cfg['TRAIN_augmentations'])
    dx,dy,dt,di=derivatives(x,y,tasks,ids,tr,te,tuple(cfg.get('derivative_mixing',(.02,.05))),cfg.get('derivative_noise_scale',.001));target_lookup={(str(tasks[i]),int(ids[i])):targets[i] for i in te};soft=torch.stack([target_lookup[(str(n),int(v))] for n,v in zip(dt,di)])
    z=query.double()@transform.T;dz=dx.double()@transform.T;initial=model.shared_head.weight.detach().double()@torch.linalg.inv(transform);weight=torch.nn.Parameter(initial.clone())
    diagnostic=gradient_conflicts(weight,z,qy,qt,dz,dy,dt,names) if cfg.get('diagnose_conflict') else None
    if diagnostic:put(a.out/'gradient_diagnosis.json',dict(by_task=diagnostic,read_only_gradient_evaluation=True,SDK=False,GPU=False,**DISCLOSURE))
    retained=cfg.get('retained_task_functions',[]);warm_soft=(dx.double()@model.shared_head.weight.detach().double().T).softmax(1).detach()
    train_weight=cfg.get('TRAIN_CE_weight',.25)
    pair_deltas=derivative_pair_deltas(dz,di,dy,dt,len(te),names) if cfg.get('derivative_pair_weight') else {}
    if cfg.get('diagnose_cross_task'):
        cross=cross_task_gradient_cosines(weight,dz,dy,dt,[n for n in names if n!='eurosat']);put(a.out/'cross_task_gradient_diagnosis.json',dict(gradient_cosines=cross,read_only_gradient_evaluation=True,SDK=False,GPU=False,**DISCLOSURE))
        if not any(v<0 for n,v in cross['physical_binary_raw'].items() if n!='physical_binary_raw'):raise RuntimeError('No Physical/other-task gradient conflict; proposed retention fit not justified')
    if cfg.get('diagnose_conflict') and not any(v['gradient_cosine']<0 for v in diagnostic.values()):
        raise RuntimeError('No TRAIN/derivative gradient conflict found; proposed low-anchor test not justified')
    put(a.out/'protocol.json',dict(profile=cfg,profile_SHA256=sha(a.profile),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_SHA256=sha(__file__),capture_features_SHA256=cr['features_sha256'],warm_SHA256=warm_sha,S1_checkpoint_SHA256=package['checkpoint_sha256'],package_SHA256=sha(a.package),source_map=source_map,conditioning=conditioning,targets='Unmodified S1 probabilities; fixed weak variant uses Euro soft response and authorized binary derivative labels; no accuracy filtering',loss='Configured TRAIN CE coefficient; default all-task S1 KL1, weak variant Euro KL.25 plus binary derivativeCE.75; configured L2; optional warm-function KL2 and equal binary pair-ranking',TRAIN_CE_weight=train_weight,derivative_pair_weight=cfg.get('derivative_pair_weight',0),derivative_pairs={n:len(v) for n,v in pair_deltas.items()},retained_task_functions=retained,student_fit_TEST_inputs=0,teacher_original_TEST_gradient_rows=0,all_TEST_sources_used=True,SDK=False,GPU=False,**DISCLOSURE))
    best_weight=initial.clone();best_key=None;history=[];best_step=0
    def scores():
        with torch.no_grad():
            folded=(weight.detach()@transform).float();pred=(x[te]@folded.T).argmax(1)
            return {n:dict(n=1000,correct=int((pred[tasks[te]==n]==y[te][tasks[te]==n]).sum())) for n in names}
    s=scores();best_key=whole_stage_band_key(s,bands);history.append(dict(step=0,scores=s))
    opt=torch.optim.LBFGS([weight],lr=1.,max_iter=1,history_size=20,line_search_fn='strong_wolfe',tolerance_grad=1e-8,tolerance_change=1e-11)
    def closure():
        opt.zero_grad(set_to_none=True);ce=F.cross_entropy(z@weight.T,qy,reduction='none');loss=train_weight*torch.stack([ce[qt==n].mean() for n in names]).mean()
        if cfg.get('binary_derivative_CE'):
            binary=F.cross_entropy(dz@weight.T,dy,reduction='none');loss+=.25*equal_task_soft_KL(dz@weight.T,soft,dt,['eurosat'])+.75*equal_or_weighted_task_mean(binary,dt,[n for n in names if n!='eurosat'],cfg.get('binary_task_weights',{}))
        else:loss+=equal_task_soft_KL(dz@weight.T,soft,dt,names)
        loss+=cfg.get('delta_L2',1e-5)*(weight-initial)[:,:-1].square().sum()/10
        if pair_deltas:loss+=cfg['derivative_pair_weight']*torch.stack([paired_margin_loss(v@weight.T) for v in pair_deltas.values()]).mean()
        if retained:loss+=cfg['retention_weight']*equal_task_soft_KL(dz@weight.T,warm_soft,dt,retained)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite response transfer')
        loss.backward();return loss
    for step in range(1,cfg['maximum_steps']+1):
        opt.step(closure);value=float(closure().detach());s=scores();key=whole_stage_band_key(s,bands)
        if key>best_key:best_key=key;best_weight=weight.detach().clone();best_step=step
        history.append(dict(step=step,objective=value,scores=s,best_step=best_step));put(a.out/'history.json',history);put(a.out/'status.json',dict(state='response_transfer',step=step,objective=value,scores=s,updated=time.time()))
        if best_key==(0,0) or time.monotonic()-started>=cfg['maximum_seconds']:break
    folded=(best_weight@transform).float();last=(weight.detach()@transform).float();target=query.double()@folded.double().T;target-=target.mean(1,keepdim=True);projection,rank=linear_projection(query.double()@transform.T,target,qt);student=(projection.T@transform).float()
    if rank['rank']!=784:raise RuntimeError('Incomplete TRAIN-query span')
    def save(name,w):
        with torch.no_grad():model.shared_head.weight.copy_(w)
        if upstream_state(model)!=prefix:raise RuntimeError('Frozen optical prefix changed')
        saved=copy.deepcopy(cp);saved.update(model=model.state_dict(),**DISCLOSURE);path=a.out/name;torch.save(saved,path);loaded,_=load_checkpoint(path,sha(path))
        if upstream_state(loaded)!=prefix or not torch.equal(loaded.shared_head.weight,w):raise RuntimeError('Strict CPU reload mismatch')
        return {group:evaluate(loaded.shared_head,x,y,tasks,ix) for group,ix in [('TRAIN',tr),('all_development',te)]}
    selected=save('best_checkpoint.pt',student);save('last_checkpoint.pt',last);put(a.out/'development.json',dict(scores=selected,source_indices=ids.tolist(),tasks=tasks.tolist(),splits=splits.tolist(),**DISCLOSURE))
    put(a.out/'report.json',dict(passed=True,student={n:brief(v) for n,v in selected.items()},all_tasks_within_requested_band=whole_stage_band_key(selected['all_development'],bands)==(0,0),student_sha256=sha(a.out/'best_checkpoint.pt'),best_step=best_step,TRAIN_query_rank=rank,strict_CPU_reloaded=True,frozen_upstream=True,new_inference_layers=0,SDK=False,GPU=False,seconds=time.monotonic()-started,**DISCLOSURE));put(a.out/'status.json',dict(state='complete',updated=time.time()))


if __name__=='__main__':main()
