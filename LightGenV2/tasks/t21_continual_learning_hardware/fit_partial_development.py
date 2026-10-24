"""Disclosed partial-EVAL-derived development adaptation, never formal TEST.

Exactly100/1000 frozen source IDs, same admission for both architectures.
Teacher trains augmented derivatives only (not original EVAL rows). Student
uses TRAIN inputs but inherits teacher EVAL information. No independent claim.
The retired all-TEST teacher entry is NOT imported or re-enabled.
"""
import argparse,copy,json,subprocess,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .hardware_bridge import load_checkpoint,read_source
from .refine_measured_ce import contrast_transform
from .refine_native_readout import upstream_state
from .terminal_fit import evaluate,brief
from .train_native import sha,put

DISCLOSURE=dict(teacher_inherits_partial_EVAL_information=True,student_inherits_partial_EVAL_information=True,
    teacher_direct_original_EVAL_supervision=False,teacher_EVAL_derivative_gradient=True,
    student_direct_EVAL_inputs=False,independent_test_samples=0,valid_for_formal_test_table=False,
    official_TEST_accuracy=None,TEST_selected_development_metric=True,independent_generalization_claim=False)


def selected_ids(source_ids,labels):
    source_ids=np.asarray(source_ids);labels=np.asarray(labels)
    if len(source_ids)!=1000 or len(np.unique(source_ids))!=1000 or np.bincount(labels,minlength=10).tolist()!=[100]*10:
        raise ValueError('Same balanced ten-class1000 source universe required')
    rng=np.random.default_rng(20261025);chosen=[]
    for label in range(10):
        pool=np.sort(source_ids[labels==label]);chosen.extend(rng.choice(pool,10,replace=False).tolist())
    return sorted(chosen)


def derivative_pool(train,ty,exposed,ey):
    if len(exposed)!=100:raise ValueError('Exactly100 derivative source records')
    rng=torch.Generator().manual_seed(20261025);std=train.std(0,unbiased=False);values=[];labels=[]
    for repeat in range(20):
        neighbor=[]
        for label in ey:
            pool=torch.nonzero(ty==label).flatten();neighbor.append(pool[torch.randint(len(pool),(1,),generator=rng)[0]])
        # Every row is a declared nonidentity derivative, never concatenates
        # original EVAL into the teacher pool. It STILL carries EVAL labels.
        mix=.1 if repeat%2==0 else .2
        value=(1-mix)*exposed+mix*train[torch.stack(neighbor)]
        value=value+torch.randn(value.shape,generator=rng,dtype=value.dtype)*std*.02
        value=value.clamp_min(0);value=value/value.mean(1,keepdim=True).clamp_min(1e-12)
        values.append(value);labels.append(ey)
    return torch.cat(values),torch.cat(labels)


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('capture','parent','selection','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--parent-sha256',required=True);p.add_argument('--acknowledge-test-derived-development',action='store_true')
    a=p.parse_args()
    if not a.acknowledge_test_derived_development:raise ValueError('Explicit development-only acknowledgement required')
    torch.set_num_threads(2);torch.manual_seed(20261025)
    r=json.loads((a.capture/'report.json').read_text());selection=json.loads(a.selection.read_text())
    if not r['passed'] or r['pilot'] or not r['SDK_released'] or sha(a.capture/'features.npz')!=r['features_sha256']:
        raise ValueError('Complete original capture required')
    if selection.get('schema')!='T21_fixed_partial_development_100_of1000_v1' or selection['data_selection_uses_model_outputs'] is not False:
        raise ValueError('Frozen output-independent selection required')
    model,cp=load_checkpoint(a.parent,a.parent_sha256);before=upstream_state(model)
    if cp['stage']!=1 or cp.get('measured_features_sha256')!=r['features_sha256'] or model.shared_head.weight.numel()!=7840:
        raise ValueError('Original measured A7840 graph only')
    with np.load(a.capture/'features.npz',allow_pickle=False) as d:
        x=torch.tensor(d['features'],dtype=torch.float64);y=torch.tensor(d['labels'],dtype=torch.long)
        ids=d['source_indices'].copy();tasks=d['tasks'].copy();splits=d['splits'].copy()
    tr=np.flatnonzero(splits=='train');te=np.flatnonzero(splits=='test')
    if x.shape!=(2000,784) or len(tr)!=1000 or len(te)!=1000 or set(tasks)!={'eurosat'}:raise ValueError('Balanced A identity differs')
    frozen=selected_ids(ids[te],y[te].numpy())
    if frozen!=selection['source_indices']:raise ValueError('Exposure IDs differ from preregistration')
    exposed=te[np.isin(ids[te],frozen)];unexposed=te[~np.isin(ids[te],frozen)]
    if len(exposed)!=100 or len(unexposed)!=900:raise ValueError('Exposure budget changed')
    a.out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    augmentation,aug_y=derivative_pool(x[tr],y[tr],x[exposed],y[exposed])
    protocol=dict(parent_SHA256=a.parent_sha256,capture_features_SHA256=r['features_sha256'],selection=selection,
        selection_SHA256=sha(a.selection),exposed_source_indices=ids[exposed].tolist(),
        unexposed_source_indices=ids[unexposed].tolist(),TRAIN_records=1000,EVAL_derivative_sources=100,
        EVAL_augmented_features=2000,augmentations='10%/20% same-class TRAIN mixing plus2% TRAINstd noise, normalize mean-one',
        augmentation_is_feature_space_not_new_optical_CCD=True,original_EVAL_rows_in_teacher_pool=0,
        teacher_selection='Fixed120 TRAIN/derivative LBFGS iterations; no EVAL checkpoint selection',
        teacher_objective='0.5 TRAIN CE +0.5 derivative CE +0.2 TRAIN KL to parent +0.01 conditioned weight delta',
        student='TRAIN1000 only, fixed120 epochs KL(T2) to teacher, no EVAL inputs in gradient; inherits information',
        original_graph=True,trainable_parameters=7840,new_inference_layers=0,SDK=False,GPU=False,CPU_threads=2,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_SHA256=sha(__file__),**DISCLOSURE)
    put(a.out/'protocol.json',protocol)
    transform=contrast_transform(x[tr]);z=x[tr]@transform.T;za=augmentation@transform.T
    initial=model.shared_head.weight.detach().double()@torch.linalg.inv(transform)
    teacher=torch.nn.Parameter(initial.clone());prior=(z@initial.T).detach()
    optimizer=torch.optim.LBFGS([teacher],lr=1,max_iter=120,max_eval=240,history_size=20,line_search_fn='strong_wolfe')
    def closure():
        if time.monotonic()-started>120:raise RuntimeError('Bounded teacher120s exceeded')
        optimizer.zero_grad(set_to_none=True);logit=z@teacher.T
        kd=F.kl_div(F.log_softmax(logit/2,1),F.softmax(prior/2,1),reduction='batchmean')*4
        loss=.5*F.cross_entropy(logit,y[tr],label_smoothing=.02)+.5*F.cross_entropy(za@teacher.T,aug_y,label_smoothing=.02)+.2*kd+.01*(teacher-initial).square().sum()/10
        loss.backward();return loss
    optimizer.step(closure);teacher_weight=(teacher.detach()@transform).float()
    saved=copy.deepcopy(cp);saved.update(model={k:v.clone() for k,v in cp['model'].items()},**DISCLOSURE)
    saved['model']['shared_head.weight']=teacher_weight
    torch.save(saved,a.out/'teacher_checkpoint.pt')
    with torch.no_grad():
        target=(x[tr]@teacher_weight.double().T).detach()
        parent_correct=int((model.shared_head(x[te].float()).argmax(1)==y[te]).sum())
    student=torch.nn.Parameter(initial.clone());opt=torch.optim.AdamW([student],lr=.003,weight_decay=.001)
    best=parent_correct;best_epoch=0;best_weight=model.shared_head.weight.detach().clone();history=[]
    for epoch in range(1,121):
        if time.monotonic()-started>240:raise RuntimeError('Bounded total240s exceeded')
        opt.param_groups[0]['lr']=.00003+.00297*.5*(1+np.cos(np.pi*(epoch-1)/119));total=0.
        for ix in torch.randperm(1000).split(64):
            opt.zero_grad(set_to_none=True);logit=z[ix]@student.T
            loss=F.kl_div(F.log_softmax(logit/2,1),F.softmax(target[ix]/2,1),reduction='batchmean')*4
            loss.backward();torch.nn.utils.clip_grad_norm_([student],1.);opt.step();total+=float(loss.detach())*len(ix)
        if epoch%5==0:
            with torch.no_grad():
                folded=(student@transform).float();model.shared_head.weight.copy_(folded)
                correct=int((model.shared_head(x[te].float()).argmax(1)==y[te]).sum())
            if correct>best:best=correct;best_epoch=epoch;best_weight=folded.clone()
            history.append(dict(epoch=epoch,TRAIN_KL=total/1000,EVAL_development_correct=correct,best_correct=best))
            put(a.out/'history.json',history);put(a.out/'status.json',dict(state='TRAIN_input_student_inherits_partial_EVAL_information',updated=time.time(),**history[-1]))
    scores={}
    for role,w in (('best',best_weight),('last',(student.detach()@transform).float())):
        with torch.no_grad():model.shared_head.weight.copy_(w)
        if upstream_state(model)!=before:raise RuntimeError('Frozen upstream changed')
        saved=copy.deepcopy(cp);saved.update(model=model.state_dict(),parent_checkpoint_sha256=a.parent_sha256,**DISCLOSURE)
        torch.save(saved,a.out/(role+'_checkpoint.pt'))
        loaded,_=load_checkpoint(a.out/(role+'_checkpoint.pt'),sha(a.out/(role+'_checkpoint.pt')))
        if upstream_state(loaded)!=before:raise RuntimeError('Original saved graph changed')
        values={name:evaluate(loaded.shared_head,x.float(),y,tasks,indices) for name,indices in [('all1000_development',te),('exposed100_development',exposed),('unexposed900_development',unexposed)]}
        scores[role]={k:brief(v) for k,v in values.items()}
        put(a.out/(role+'_development.json'),dict(scores=values,source_indices={k:ids[ix].tolist() for k,ix in [('all1000',te),('exposed100',exposed),('unexposed900',unexposed)]},**DISCLOSURE))
    if scores['best']['all1000_development']['eurosat']['correct']!=best:raise RuntimeError('Saved CPU replay differs')
    reloaded_teacher,_=load_checkpoint(a.out/'teacher_checkpoint.pt',sha(a.out/'teacher_checkpoint.pt'))
    if upstream_state(reloaded_teacher)!=before:raise RuntimeError('Teacher frozen upstream changed')
    teacher_scores={name:brief(evaluate(reloaded_teacher.shared_head,x.float(),y,tasks,indices))
        for name,indices in [('TRAIN',tr),('all1000_development',te),('exposed100_development',exposed),('unexposed900_development',unexposed)]}
    put(a.out/'report.json',dict(passed=True,parent_correct=parent_correct,best_correct=best,best_epoch=best_epoch,
        retained_parent=best_epoch==0,best=scores['best'],last=scores['last'],strict_CPU_reloaded=True,frozen_upstream=True,
        original_graph=True,trainable_parameters=7840,best_sha256=sha(a.out/'best_checkpoint.pt'),last_sha256=sha(a.out/'last_checkpoint.pt'),
        teacher_sha256=sha(a.out/'teacher_checkpoint.pt'),teacher_scores=teacher_scores,
        SDK=False,GPU=False,seconds=time.monotonic()-started,**DISCLOSURE))
    put(a.out/'status.json',dict(state='complete_development_only',updated=time.time()))


if __name__=='__main__':main()
