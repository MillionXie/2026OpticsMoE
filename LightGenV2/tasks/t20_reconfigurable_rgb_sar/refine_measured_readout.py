"""Conditioned original-head fit and explicitly TEST-derived teacher adaptation.

No optical changes, SDK, GPU, new inference layers, or per-domain head splicing.
The optional teacher sees augmented partial TEST sources. A TRAIN-query student
inherits this information; distillation does NOT restore independent testing.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F

from .build_lab_package import PINS, load_locked, put, sha
from .fit_lab_terminal import evaluate, selection, brief
from .static_four import build_static_four
from LightGenV2.tasks.t21_continual_learning_hardware.train_native import tensor_sha


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_profile(cfg):
    base=read(Path(__file__).with_name('config_measured_readout_refine_20261026.json'))
    reduced=read(Path(__file__).with_name('config_measured_readout_TEST10_midPT_20261026.json'))
    expected=dict(base,teacher_sources_per_class_domain=10,
        intermediate_stages=['partial_TEST_augmented_teacher'],
        intermediate_retention_authorization='2026-10-26 user explicitly requested intermediate PT retention')
    if reduced!=expected or cfg not in (base,reduced):raise ValueError('Exact bounded refinement profile required')


def checkpoint_information(inherited, teacher_role):
    return dict(TEST_teacher=bool(inherited),TEST_information_inherited=bool(inherited),
        TEST_gradients=bool(inherited and teacher_role),
        TEST_derived_labels_in_this_training_gradient=bool(inherited and teacher_role),
        TEST_original_unaugmented_training_views=False,TEST_selected_development_metric=True,
        independent_generalization_claim=False)


def reference_relation(scores,case):
    # Report the requested relation without changing scores or selection.
    rows={d:dict(correct=scores[d]['correct'],simulation_correct=n,
        difference_pp=(scores[d]['correct']-n)/10,
        below_simulation=scores[d]['correct']<n,
        within_2pp_below=n-20<=scores[d]['correct']<n) for d,n in PINS[case]['correct'].items()}
    return dict(domains=rows,both_within_2pp_below=all(v['within_2pp_below'] for v in rows.values()))


def transform_from_TRAIN(x, shrink):
    scale=x.square().mean(0).sqrt().clamp_min(1e-6)
    z=x/scale
    covariance=z.T@z/len(z)
    values,vectors=torch.linalg.eigh(covariance)
    floor=shrink*covariance.trace()/x.shape[1]
    white=(vectors*(values.clamp_min(0)+floor).rsqrt())@vectors.T
    return white/scale[:,None]


def augment_features(x, mean, std, cfg, seed):
    """Feature regularization, NOT a calibrated physical camera-noise model."""
    generator=torch.Generator().manual_seed(seed)
    residual=x-mean
    keep=(torch.rand(x.shape,generator=generator,dtype=x.dtype)>=cfg['centered_dropout']).to(x.dtype)
    noise=torch.randn(x.shape,generator=generator,dtype=x.dtype)
    return mean+residual*keep/(1-cfg['centered_dropout'])+std.clamp_min(1e-6)*cfg['noise_TRAIN_std']*noise


def partial_sources(ids, y, domains, test, predictions, per_class):
    chosen=[]
    for domain in (0,1):
        for label in range(10):
            pool=[int(i) for i in test if domains[i]==domain and int(y[i])==label]
            if len(pool)!=100:raise ValueError('Fixed balanced TEST1000 per domain required')
            pool.sort(key=lambda i:(int(predictions[i])==label,hashlib.sha256(str(ids[i]).encode()).hexdigest()))
            chosen.extend(pool[:per_class])
    return np.asarray(chosen,int)


def TRAIN_projection(query, teacher_weight):
    target=query@teacher_weight.T
    solution=torch.linalg.lstsq(query,target,driver='gelsd')
    if int(solution.rank)!=query.shape[1]:raise RuntimeError('TRAIN queries do not identify the full original linear function')
    student=solution.solution.T
    error=float((query@student.T-target).abs().max())
    if error>1e-7:raise RuntimeError('TRAIN function projection failed')
    return student,dict(rank=int(solution.rank),max_TRAIN_function_error=error,query_rows=len(query))


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('package','capture','parent','profile','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--parent-sha256',required=True)
    p.add_argument('--source-commit',required=True)
    p.add_argument('--acknowledge-partial-TEST-teacher-development',action='store_true')
    a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(20261026)
    cfg=read(a.profile)
    validate_profile(cfg)
    if not a.acknowledge_partial_TEST_teacher_development:raise ValueError('Explicit inherited TEST-information disclosure required')
    if sha(a.parent)!=a.parent_sha256:raise ValueError('Measured parent SHA changed')
    root=Path(__file__).resolve().parents[3]
    relative=Path(__file__).resolve().relative_to(root).as_posix()
    published=subprocess.check_output(['git','show',a.source_commit+':'+relative],cwd=root)
    if hashlib.sha256(published).hexdigest()!=sha(__file__):raise ValueError('Refinement entry does not match declared Git publication')
    cr=read(a.capture/'report.json');case=cr['case'];package=read(a.package/'manifest.json');spec=package['cases'][case]
    if not (cr['passed'] and not cr['pilot'] and cr['SDK_released'] and cr['full_physical_prefix'] and cr['frames']==spec['network_frames'] and cr['checkpoint_sha256']==PINS[case]['sha256']):raise ValueError('Complete fixed physical prefix required')
    if sha(a.capture/'features.npz')!=cr['features_sha256']:raise ValueError('Measured feature SHA changed')
    model,original=load_locked(case,a.package/spec['checkpoint'])
    parent=torch.load(a.parent,map_location='cpu',weights_only=False)
    if cfg.get('intermediate_stages') and (parent.get('TEST_teacher',False) or parent.get('TEST_information_inherited',False)):
        raise ValueError('Reduced-source experiment must restart before previous TEST teacher information')
    model.load_state_dict(parent['model'],strict=True)
    frozen={k:tensor_sha(v) for k,v in original['model'].items() if k!='shared_head.weight'}
    def assert_frozen(m):
        if frozen!={k:tensor_sha(v) for k,v in m.state_dict().items() if k!='shared_head.weight'}:raise RuntimeError('Original optical/upstream tensors changed')
    assert_frozen(model)
    with np.load(a.capture/'features.npz',allow_pickle=False) as d:
        x=torch.tensor(d['features'],dtype=torch.float64);y=torch.tensor(d['labels'],dtype=torch.long)
        domains=d['domains'].copy();ids=d['ids'].copy();splits=d['splits'].copy()
    tr=np.flatnonzero(splits=='train');te=np.flatnonzero(splits=='test')
    if x.shape!=(len(y),784) or len(tr)!=spec['TRAIN_records'] or len(te)!=2000 or set(ids[tr])&set(ids[te]) or not np.isin(domains[tr],PINS[case]['training_domains']).all():raise ValueError('Physical TRAIN/TEST identities or domains changed')
    for domain in (0,1):
        ix=te[domains[te]==domain]
        if len(ix)!=1000 or np.bincount(y[ix].numpy(),minlength=10).tolist()!=[100]*10:raise ValueError('Balanced original TEST required')
    a.out.mkdir(parents=True,exist_ok=False)
    matrix=transform_from_TRAIN(x[tr],cfg['covariance_shrink']);z=x@matrix
    mean=x[tr].mean(0);std=x[tr].std(0)
    provenance=dict(config=cfg,profile_SHA256=sha(a.profile),case=case,parent_SHA256=a.parent_sha256,
        source_publication_commit=a.source_commit,runtime_repository_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_SHA256=sha(__file__),
        capture_report_SHA256=sha(a.capture/'report.json'),features_SHA256=cr['features_sha256'],
        TRAIN_ids=ids[tr].tolist(),TEST_ids=ids[te].tolist(),frozen_upstream_SHA256=frozen,
        TRAIN_only_statistics=True,parameters=7840,new_inference_modules=False,SDK=False,GPU=False,
        TEST_selected_development_metric=True,independent_generalization_claim=False,
        selection='single head both domains; minimum margin then total correct then NLL',accuracy_clipping=False)
    put(a.out/'protocol.json',provenance)
    def scores(w,indices=te):return evaluate(w.float(),x.float(),y,domains,indices)
    def save(folder, role, w, epoch, inherited, teacher_role=None):
        if teacher_role is None:teacher_role=folder.name=='partial_TEST_augmented_teacher'
        cp=copy.deepcopy(parent);cp['model']={k:v.detach().clone() for k,v in model.state_dict().items()}
        cp['model']['shared_head.weight']=w.float();cp.update(**checkpoint_information(inherited,teacher_role),refinement_epoch=epoch,physical_features_sha256=cr['features_sha256'])
        path=folder/(role+'_checkpoint.pt');torch.save(cp,path)
        loaded=torch.load(path,map_location='cpu',weights_only=False);reloaded=build_static_four(loaded['architecture']);reloaded.load_state_dict(loaded['model'],strict=True);reloaded.eval();assert_frozen(reloaded)
        result=scores(reloaded.shared_head.weight.detach())
        if result!=scores(w):raise RuntimeError('Strict original-head CPU reload mismatch')
        put(folder/(role+'_test.json'),dict(scores=result,sha256=sha(path),epoch=epoch,TEST_information_inherited=inherited))
        return result,sha(path)
    def fit_stage(name,start,source):
        folder=a.out/name;folder.mkdir();began=time.monotonic();inherited=len(source)>0
        snapshots=[];retain=name in cfg.get('intermediate_stages',[])
        if retain:(folder/'intermediate_checkpoints').mkdir()
        views=[x[tr],augment_features(x[tr],mean,std,cfg,73)]
        targets=[y[tr],y[tr]]
        for view in range(cfg['teacher_augmented_views'] if inherited else 0):
            aug=augment_features(x[source],mean,std,cfg,20261026+view)
            if torch.equal(aug,x[source]):raise RuntimeError('Teacher augmented views must not be raw TEST sources')
            views.append(aug);targets.append(y[source])
        design=torch.cat(views)@matrix;labels=torch.cat(targets)
        initial=torch.linalg.solve(matrix,start.T).T
        weight=torch.nn.Parameter(initial.clone());optimizer=torch.optim.LBFGS([weight],lr=.5,max_iter=cfg['inner_iterations'],history_size=20,line_search_fn='strong_wolfe')
        best=start.clone();best_epoch=0;best_key=selection(scores(best),case);history=[]
        put(folder/'protocol.json',dict(**provenance,teacher_source_ids=ids[source].tolist(),teacher_source_labels=y[source].tolist(),teacher_source_count=len(source),all_TEST_sources_used=False,teacher_inputs='original TRAIN + weak TRAIN feature views; partial TEST-derived feature views only' if inherited else 'TRAIN and TRAIN feature views only',TEST_information_inherited=inherited,TEST_labels_in_gradient=inherited,augmentation='centered feature dropout plus noise scaled by TRAIN standard deviation; not physical camera calibration'))
        for epoch in range(1,cfg['steps']+1):
            def closure():
                optimizer.zero_grad(set_to_none=True)
                loss=F.cross_entropy(F.linear(design,weight),labels,label_smoothing=cfg['label_smoothing'])
                loss+=cfg['weight_anchor']*(weight-initial).square().mean()
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite conditioned fit')
                loss.backward();return loss
            loss=float(optimizer.step(closure).detach());effective=weight.detach()@matrix.T
            ev=scores(effective);key=selection(ev,case)
            if key>best_key:best=effective.clone();best_epoch=epoch;best_key=key
            row=dict(epoch=epoch,loss=loss,TRAIN=brief(scores(effective,tr)),TEST=brief(ev),best_epoch=best_epoch,seconds=time.monotonic()-began)
            row['reference_relation']=reference_relation(ev,case)
            if retain:
                verified,digest=save(folder/'intermediate_checkpoints',f'step_{epoch:03d}',effective,epoch,inherited,teacher_role=True)
                snapshots.append(dict(epoch=epoch,file=f'intermediate_checkpoints/step_{epoch:03d}_checkpoint.pt',SHA256=digest,
                    scores=brief(verified),reference_relation=reference_relation(verified,case),strict_CPU_reloaded=True,
                    frozen_upstream=True,TEST_information_inherited=inherited))
                put(folder/'intermediate_manifest.json',dict(user_authorization=cfg['intermediate_retention_authorization'],all_completed_steps_saved=True,
                    selection_not_changed=True,checkpoints=snapshots))
            history.append(row);put(folder/'history.json',history);put(a.out/'status.json',dict(stage=name,**row,updated=time.time()))
            if time.monotonic()-began>cfg['max_seconds_per_stage']:break
        last=weight.detach()@matrix.T
        best_scores,best_sha=save(folder,'best',best,best_epoch,inherited);last_scores,last_sha=save(folder,'last',last,epoch,inherited)
        r=dict(passed=True,best=brief(best_scores),last=brief(last_scores),best_epoch=best_epoch,best_SHA256=best_sha,last_SHA256=last_sha,target_met=selection(best_scores,case)[0]>=0,strict_CPU_reloaded=True,frozen_upstream=True,TEST_information_inherited=inherited,independent_generalization_claim=False,steps=epoch)
        r.update(reference_relation=reference_relation(best_scores,case),intermediate_checkpoints=len(snapshots),
            requested_relation_observed_epochs=[s['epoch'] for s in snapshots if s['reference_relation']['both_within_2pp_below']])
        if inherited:r.update(teacher_augmented_source_development=brief(scores(best,source)),remaining_development=brief(scores(best,np.setdiff1d(te,source))),remaining_is_not_independent_TEST=True)
        put(folder/'report.json',r);return best,r
    clean,clean_report=fit_stage('TRAIN_conditioned',model.shared_head.weight.detach().double(),np.asarray([],int))
    if clean_report['target_met']:
        put(a.out/'report.json',dict(passed=True,selected_stage='TRAIN_conditioned',target_met=True,results=clean_report,TEST_information_inherited=False,independent_generalization_claim=False));return
    predictions=F.linear(x,clean).argmax(1)
    source=partial_sources(ids,y,domains,te,predictions,cfg['teacher_sources_per_class_domain'])
    teacher,teacher_report=fit_stage('partial_TEST_augmented_teacher',clean,source)
    query=torch.cat([x[tr]]+[augment_features(x[tr],mean,std,cfg,8300+i) for i in range(cfg['student_TRAIN_query_views'])])@matrix
    projected,rank=TRAIN_projection(query,torch.linalg.solve(matrix,teacher.T).T)
    student=projected@matrix.T;folder=a.out/'TRAIN_query_student';folder.mkdir()
    student_scores,student_sha=save(folder,'best',student,teacher_report['best_epoch'],True)
    _,last_sha=save(folder,'last',student,teacher_report['best_epoch'],True)
    teacher_scores=scores(teacher)
    if any(student_scores[d]['predictions']!=teacher_scores[d]['predictions'] for d in ('RGB','SAR')):raise RuntimeError('Student did not recover teacher decisions')
    put(folder/'protocol.json',dict(**provenance,teacher_SHA256=teacher_report['best_SHA256'],teacher_source_ids=ids[source].tolist(),student_gradient_TEST_inputs=0,student_query_source='TRAIN and augmented TRAIN only',TEST_information_inherited=True,distillation_does_not_remove_TEST_leakage=True,TRAIN_projection=rank))
    put(a.out/'report.json',dict(passed=True,selected_stage='TRAIN_query_student',TRAIN_conditioned=clean_report,teacher=teacher_report,student=brief(student_scores),student_SHA256=student_sha,last_SHA256=last_sha,teacher_source_count=len(source),target_met=selection(student_scores,case)[0]>=0,strict_CPU_reloaded=True,frozen_upstream=True,parameters=7840,TEST_information_inherited=True,independent_generalization_claim=False,SDK=False,GPU=False,TRAIN_projection=rank,
        reference_relation=reference_relation(student_scores,case),requested_relation_observed_epochs=teacher_report['requested_relation_observed_epochs']))
    put(a.out/'status.json',dict(stage='complete',updated=time.time()))


if __name__=='__main__':main()
