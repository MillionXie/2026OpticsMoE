"""One TRAIN-augmentation functional projection of a sealed offline teacher.

Student remains the original7840 Linear. Four ridge strengths chosen on held
TRAIN source IDs for teacher-function approximation, not EVAL scores.
All teacher partial-EVAL information remains disclosed and inherited.
"""
import argparse,copy,json,time,subprocess
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .fit_augmented_residual_teacher import ResidualTeacher,augment_train,select_sources
from .hardware_bridge import load_checkpoint
from .refine_native_readout import upstream_state
from .refine_measured_ce import contrast_transform
from .terminal_fit import evaluate,brief,score
from .train_native import sha,put


def solve_projection(gram,cross,strength):
    penalty=torch.eye(len(gram),dtype=gram.dtype);penalty[-1,-1]=0
    return torch.linalg.solve(gram+strength*penalty,cross).T


def solve_pair_projection(gram,cross,pair_gram,pair_cross,strength):
    """Equal-weight binary-margin difference objective, same ten-row Linear."""
    fitted=solve_projection(gram,cross,strength)
    average=(fitted[0]+fitted[1])/2
    margin=solve_projection(gram+pair_gram,((cross[:,1]-cross[:,0])/2+pair_cross)[:,None],strength)[0]
    fitted[0]=average-margin;fitted[1]=average+margin
    return fitted


def binary_source_pairs(ids,labels,tasks):
    """Complete CLEVR TRAIN source pairs only, returned positive then negative."""
    lookup={int(source):i for i,source in enumerate(ids) if tasks[i]=='clevr'}
    pairs=[]
    for group in sorted({source//2 for source in lookup}):
        positions=[lookup.get(2*group),lookup.get(2*group+1)]
        if None in positions:raise ValueError('Partial TRAIN pair forbidden')
        a,b=positions
        if sorted([int(labels[a]),int(labels[b])])!=[0,1]:raise ValueError('Binary TRAIN pair labels differ')
        pairs.append((a,b) if int(labels[a])==1 else (b,a))
    if not pairs:raise ValueError('Complete CLEVR TRAIN pairs required')
    return torch.tensor(pairs,dtype=torch.long)


def centered_jacobian_mean(teacher,z,weights):
    """Exact eval-mode GELU Jacobian mean; fixed DC coordinate is excluded."""
    if teacher.training:raise ValueError('Sealed eval-mode teacher required')
    first,last=teacher.residual[0],teacher.residual[-1];pre=first(z)
    slope=.5*(1+torch.erf(pre/2**.5))+pre*torch.exp(-pre.square()/2)/(2*np.pi)**.5
    mean_slope=(slope.double()*weights.double()).sum(0)
    base=teacher.parent.double()-teacher.parent.double().mean(0,keepdim=True)
    outer=last.weight.double()-last.weight.double().mean(0,keepdim=True)
    jac=base*weights.sum()+outer@(mean_slope[:,None]*first.weight.double())
    jac[:,-1]=0
    return jac


def admit_legacy_partial(protocol,report):
    if protocol.get('selection',{}).get('schema')!='T21_fixed_partial_development_100_of1000_v1' or protocol.get('EVAL_derivative_sources')!=100:
        raise ValueError('Only sealed source-frozen100 derivative teacher; retired all-TEST teacher forbidden')
    if report.get('teacher_direct_original_EVAL_supervision') is not False or report.get('teacher_EVAL_derivative_gradient') is not True or report.get('valid_for_formal_test_table') is not False:
        raise ValueError('Legacy partial-EVAL disclosure differs')
    return dict(teacher_uses_EVAL_derivatives=True,student_inherits_EVAL_information=True,student_original_EVAL_inputs=False,
        original_EVAL_rows_in_teacher_pool=0,valid_for_formal_test_table=False,independent_test_samples=0,
        TEST_selected_development_metric=True,independent_generalization_claim=False)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(__doc__)
    for n in ('capture','parent','teacher_run','out'):p.add_argument('--'+n.replace('_','-'),type=Path,required=True)
    p.add_argument('--parent-sha256',required=True);p.add_argument('--legacy-linear-teacher',action='store_true')
    p.add_argument('--match-teacher-jacobian',action='store_true',help='One fixed weight1 TRAIN-only analytic teacher-gradient matching variant')
    p.add_argument('--match-pair-contrasts',action='store_true',help='One fixed weight1 TRAIN-pair binary-margin projection with shared weak noise')
    a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(20261025)
    if a.match_teacher_jacobian and a.legacy_linear_teacher:raise ValueError('Jacobian variant only for nonlinear offline teacher')
    if a.match_pair_contrasts and (a.match_teacher_jacobian or a.legacy_linear_teacher):raise ValueError('Pair variant is separate nonlinear-teacher method')
    capture=json.loads((a.capture/'report.json').read_text());report=json.loads((a.teacher_run/'report.json').read_text());protocol=json.loads((a.teacher_run/'protocol.json').read_text())
    teacher_path=a.teacher_run/('teacher_checkpoint.pt' if a.legacy_linear_teacher else 'offline_teacher.pt')
    if not capture['passed'] or not report['passed'] or sha(a.capture/'features.npz')!=protocol['capture_features_SHA256'] or sha(teacher_path)!=report['teacher_sha256']:
        raise ValueError('Sealed capture and exact completed teacher required')
    model,cp=load_checkpoint(a.parent,a.parent_sha256);before=upstream_state(model)
    if cp.get('measured_features_sha256')!=capture['features_sha256'] or model.shared_head.weight.numel()!=7840:raise ValueError('Measured original Linear required')
    with np.load(a.capture/'features.npz',allow_pickle=False) as d:
        x=torch.tensor(d['features'],dtype=torch.float32);y=torch.tensor(d['labels'],dtype=torch.long)
        tasks=d['tasks'].copy();ids=d['source_indices'].copy();splits=d['splits'].copy()
    tr=np.flatnonzero(splits=='train');te=np.flatnonzero(splits=='test');tt=tasks[tr];names=sorted(set(tt));rng=np.random.default_rng(20261025);held=[]
    if a.match_pair_contrasts and (cp['architecture']!='d2nn' or cp['stage']!=2):raise ValueError('Pair diagnosis variant is D2NN B only')
    if a.legacy_linear_teacher:
        disclosure=admit_legacy_partial(protocol,report)
        if protocol['parent_SHA256']!=a.parent_sha256 or cp['stage']!=1:raise ValueError('Legacy source-frozen A parent differs')
        teacher_model,_=load_checkpoint(teacher_path,report['teacher_sha256'])
        if upstream_state(teacher_model)!=before:raise ValueError('Legacy teacher upstream differs')
        teacher=teacher_model.shared_head.eval();transform=contrast_transform(x[tr]).float()
        query=lambda values:teacher(values)
        selected={'eurosat':protocol['selection']['source_indices']};teacher_scope='sealed_100_source_linear_teacher_development'
    else:
        artifact=torch.load(teacher_path,map_location='cpu',weights_only=False)
        if artifact['parent_SHA256']!=a.parent_sha256:raise ValueError('Teacher-parent identity differs')
        teacher=ResidualTeacher(artifact['state_dict']['parent']);teacher.load_state_dict(artifact['state_dict']);teacher.eval();transform=artifact['transform']
        query=lambda values:teacher(values@transform.T)
        selected=protocol['selection_source_ids'];teacher_scope=protocol['scope']
        disclosure={k:report[k] for k in ('teacher_uses_EVAL_derivatives','student_inherits_EVAL_information','student_original_EVAL_inputs','original_EVAL_rows_in_teacher_pool','valid_for_formal_test_table','independent_test_samples','TEST_selected_development_metric','independent_generalization_claim')}
    for n in names:
        pool=np.flatnonzero(tt==n)
        if n=='eurosat':
            for k in range(10):held.extend(rng.permutation(pool[y[tr][pool].numpy()==k])[:len(pool)//50].tolist())
        else:
            groups=np.unique(ids[tr][pool]//2);chosen=rng.choice(groups,len(groups)//5,replace=False);held.extend(pool[np.isin(ids[tr][pool]//2,chosen)].tolist())
    held=np.array(sorted(held));fit=np.setdiff1d(np.arange(len(tr)),held)
    if len(held)==0 or len(fit)==0:raise ValueError('TRAIN functional holdout required')
    a.out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    put(a.out/'protocol.json',dict(parent_SHA256=a.parent_sha256,teacher_SHA256=report['teacher_sha256'],capture_features_SHA256=capture['features_sha256'],
        method='TRAIN-only weak-noise function and paired binary-margin differences, fixed pair weight1, refit full TRAIN' if a.match_pair_contrasts else 'TRAIN-only centered-logit plus exact GELU-Jacobian matching, fixed weight1, refit full TRAIN' if a.match_teacher_jacobian else 'Augmented TRAIN-only function projection of fixed teacher, centered logits, four ridge strengths selected by held TRAIN KL, refit full TRAIN',
        strengths=[1.] if a.match_teacher_jacobian else [.0001,.001,.01,.1],TRAIN_augmentations_per_source=24,held_TRAIN_source_ids=ids[tr][held].tolist(),
        teacher_Jacobian_matching=a.match_teacher_jacobian,Jacobian_inputs='TRAIN and its augmentations only',
        pair_contrast_matching=a.match_pair_contrasts,pair_weight=1 if a.match_pair_contrasts else 0,
        pair_variant_augmentation='No dropout, .002 TRAINstd additive noise; same noise for positive/negative pair, mean-one renormalization' if a.match_pair_contrasts else None,
        pair_scale='Inverse RMS TRAIN paired contrast coordinate, estimated on student fit TRAIN only' if a.match_pair_contrasts else None,
        held_TRAIN_is_student_functional_approximation_not_independent_teacher_validation=True,
        original_student_parameters=7840,new_inference_layers=0,SDK=False,GPU=False,maximum_seconds=240,
        teacher_scope=teacher_scope,teacher_selection_source_ids=selected,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_SHA256=sha(__file__),**disclosure))
    def statistics(indices):
        source=x[tr][indices];st=tt[indices];mean=source.mean(0,keepdim=True);std=source.std(0,unbiased=False)
        gram=torch.zeros((784,784),dtype=torch.float64);cross=torch.zeros((784,10),dtype=torch.float64);jac=torch.zeros((10,784),dtype=torch.float64)
        weight=torch.tensor([1/(len(names)*int((st==n).sum())) for n in st],dtype=torch.float64)
        for repeat in range(24):
            if time.monotonic()-started>240:raise RuntimeError('Fixed240s projection bound reached')
            if repeat==0:values=source
            elif a.match_pair_contrasts:
                values=(source+torch.randn_like(source)*std*.002).clamp_min(0);values=values/values.mean(1,keepdim=True).clamp_min(1e-12)
            else:values=augment_train(source,mean,std)
            for ix in torch.arange(len(source)).split(256):
                z=values[ix]@transform.T;target=query(values[ix]);target=target-target.mean(1,keepdim=True);zz=z.double();ww=weight[ix,None]/24
                gram+=zz.T@(ww*zz);cross+=zz.T@(ww*target.double())
                if a.match_teacher_jacobian:jac+=centered_jacobian_mean(teacher,z,ww)
        return gram,cross,jac
    def pair_statistics(indices):
        source=x[tr][indices];pairs=binary_source_pairs(ids[tr][indices],y[tr][indices],tt[indices]);std=source.std(0,unbiased=False)
        positive,negative=source[pairs[:,0]],source[pairs[:,1]]
        delta=(positive-negative)@transform.T;scale=delta.double().square().mean().clamp_min(1e-12).rsqrt()
        gram=torch.zeros((784,784),dtype=torch.float64);cross=torch.zeros(784,dtype=torch.float64)
        for repeat in range(24):
            if time.monotonic()-started>240:raise RuntimeError('Fixed240s pair projection bound reached')
            noise=torch.zeros_like(positive) if repeat==0 else torch.randn_like(positive)*std*.002
            pa=(positive+noise).clamp_min(0);na=(negative+noise).clamp_min(0)
            pa=pa/pa.mean(1,keepdim=True).clamp_min(1e-12);na=na/na.mean(1,keepdim=True).clamp_min(1e-12)
            for ix in torch.arange(len(pa)).split(128):
                d=((pa[ix]-na[ix])@transform.T).double()*scale;tp,tn=query(pa[ix]),query(na[ix])
                target=((tp[:,1]-tp[:,0])-(tn[:,1]-tn[:,0])).double()*scale/2
                gram+=d.T@d/(24*len(pa));cross+=d.T@target/(24*len(pa))
        return gram,cross
    gram,cross,jac=statistics(fit);target=query(x[tr][held]).double();rows=[]
    if a.match_pair_contrasts:pair_gram,pair_cross=pair_statistics(fit)
    for strength in ((1.,) if a.match_teacher_jacobian else (.0001,.001,.01,.1)):
        w=solve_pair_projection(gram,cross,pair_gram,pair_cross,strength) if a.match_pair_contrasts else solve_projection(gram,cross+strength*jac.T,strength);pred=(x[tr][held]@transform.T).double()@w.T
        loss=F.kl_div(F.log_softmax(pred/2,1),F.softmax(target/2,1),reduction='none').sum(1)*4
        metric=float(np.mean([float(loss[torch.tensor(tt[held]==n)].mean()) for n in names]));rows.append(dict(strength=strength,held_TRAIN_teacher_KL=metric))
    chosen=min(rows,key=lambda r:(r['held_TRAIN_teacher_KL'],-r['strength']))['strength'];gram,cross,jac=statistics(np.arange(len(tr)))
    if a.match_pair_contrasts:pair_gram,pair_cross=pair_statistics(np.arange(len(tr)));w=solve_pair_projection(gram,cross,pair_gram,pair_cross,chosen)
    else:w=solve_projection(gram,cross+chosen*jac.T,chosen)
    folded=(w@transform.double()).float();baseline=evaluate(model.shared_head,x,y,tasks,te);model.shared_head.weight.copy_(folded);candidate=evaluate(model.shared_head,x,y,tasks,te)
    retained=score(candidate)<=score(baseline);weights={'best':cp['model']['shared_head.weight'].clone() if retained else folded,'last':folded};result={}
    partial=disclosure['teacher_uses_EVAL_derivatives'];exposed=np.array([i for i in te if partial and int(ids[i]) in selected[str(tasks[i])]],dtype=int);unexposed=np.setdiff1d(te,exposed)
    for role,weight in weights.items():
        model.shared_head.weight.copy_(weight)
        if upstream_state(model)!=before:raise RuntimeError('Frozen upstream changed')
        saved=copy.deepcopy(cp);saved.update(model=model.state_dict(),parent_checkpoint_sha256=a.parent_sha256,**disclosure)
        torch.save(saved,a.out/(role+'_checkpoint.pt'));loaded,_=load_checkpoint(a.out/(role+'_checkpoint.pt'),sha(a.out/(role+'_checkpoint.pt')))
        if upstream_state(loaded)!=before:raise RuntimeError('Saved original upstream differs')
        if not torch.equal(loaded.shared_head.weight,weight):raise RuntimeError('Saved original Linear differs')
        values={name:evaluate(loaded.shared_head,x,y,tasks,ix) for name,ix in [('TRAIN',tr),('all_development',te),('exposed_development',exposed),('unexposed_development',unexposed)] if len(ix)}
        result[role]={k:brief(v) for k,v in values.items()};put(a.out/(role+'_development.json'),dict(scores=values,**disclosure))
    put(a.out/'report.json',dict(passed=True,baseline=brief(baseline),candidate=brief(candidate),best=result['best'],last=result['last'],retained_parent=retained,
        strength_selection=rows,selected_strength=chosen,strict_CPU_reloaded=True,frozen_upstream=True,original_student_graph=True,student_trainable_parameters=7840,
        best_sha256=sha(a.out/'best_checkpoint.pt'),last_sha256=sha(a.out/'last_checkpoint.pt'),teacher_sha256=report['teacher_sha256'],seconds=time.monotonic()-started,SDK=False,GPU=False,**disclosure))
    print(json.dumps(dict(passed=True,retained_parent=retained,best=result['best'],**disclosure)))


if __name__=='__main__':main()
