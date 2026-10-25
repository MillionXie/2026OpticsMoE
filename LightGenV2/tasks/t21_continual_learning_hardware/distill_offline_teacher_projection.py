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
from .terminal_fit import evaluate,brief,score
from .train_native import sha,put


def solve_projection(gram,cross,strength):
    penalty=torch.eye(len(gram),dtype=gram.dtype);penalty[-1,-1]=0
    return torch.linalg.solve(gram+strength*penalty,cross).T


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(__doc__)
    for n in ('capture','parent','teacher_run','out'):p.add_argument('--'+n.replace('_','-'),type=Path,required=True)
    p.add_argument('--parent-sha256',required=True);a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(20261025)
    capture=json.loads((a.capture/'report.json').read_text());report=json.loads((a.teacher_run/'report.json').read_text());protocol=json.loads((a.teacher_run/'protocol.json').read_text())
    if not capture['passed'] or not report['passed'] or sha(a.capture/'features.npz')!=protocol['capture_features_SHA256'] or sha(a.teacher_run/'offline_teacher.pt')!=report['teacher_sha256']:
        raise ValueError('Sealed capture and exact completed teacher required')
    model,cp=load_checkpoint(a.parent,a.parent_sha256);before=upstream_state(model)
    if cp.get('measured_features_sha256')!=capture['features_sha256'] or model.shared_head.weight.numel()!=7840:raise ValueError('Measured original Linear required')
    artifact=torch.load(a.teacher_run/'offline_teacher.pt',map_location='cpu',weights_only=False)
    if artifact['parent_SHA256']!=a.parent_sha256:raise ValueError('Teacher-parent identity differs')
    teacher=ResidualTeacher(artifact['state_dict']['parent']);teacher.load_state_dict(artifact['state_dict']);teacher.eval();transform=artifact['transform']
    with np.load(a.capture/'features.npz',allow_pickle=False) as d:
        x=torch.tensor(d['features'],dtype=torch.float32);y=torch.tensor(d['labels'],dtype=torch.long)
        tasks=d['tasks'].copy();ids=d['source_indices'].copy();splits=d['splits'].copy()
    tr=np.flatnonzero(splits=='train');te=np.flatnonzero(splits=='test');tt=tasks[tr];names=sorted(set(tt));rng=np.random.default_rng(20261025);held=[]
    for n in names:
        pool=np.flatnonzero(tt==n)
        if n=='eurosat':
            for k in range(10):held.extend(rng.permutation(pool[y[tr][pool].numpy()==k])[:len(pool)//50].tolist())
        else:
            groups=np.unique(ids[tr][pool]//2);chosen=rng.choice(groups,len(groups)//5,replace=False);held.extend(pool[np.isin(ids[tr][pool]//2,chosen)].tolist())
    held=np.array(sorted(held));fit=np.setdiff1d(np.arange(len(tr)),held)
    if len(held)==0 or len(fit)==0:raise ValueError('TRAIN functional holdout required')
    a.out.mkdir(parents=True,exist_ok=False);started=time.monotonic();disclosure={k:report[k] for k in ('teacher_uses_EVAL_derivatives','student_inherits_EVAL_information','student_original_EVAL_inputs','original_EVAL_rows_in_teacher_pool','valid_for_formal_test_table','independent_test_samples','TEST_selected_development_metric','independent_generalization_claim')}
    put(a.out/'protocol.json',dict(parent_SHA256=a.parent_sha256,teacher_SHA256=report['teacher_sha256'],capture_features_SHA256=capture['features_sha256'],
        method='Augmented TRAIN-only function projection of fixed teacher, centered logits, four ridge strengths selected by held TRAIN KL, refit full TRAIN',
        strengths=[.0001,.001,.01,.1],TRAIN_augmentations_per_source=24,held_TRAIN_source_ids=ids[tr][held].tolist(),
        held_TRAIN_is_student_functional_approximation_not_independent_teacher_validation=True,
        original_student_parameters=7840,new_inference_layers=0,SDK=False,GPU=False,maximum_seconds=240,
        teacher_scope=protocol['scope'],source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_SHA256=sha(__file__),**disclosure))
    def statistics(indices):
        source=x[tr][indices];st=tt[indices];mean=source.mean(0,keepdim=True);std=source.std(0,unbiased=False)
        gram=torch.zeros((784,784),dtype=torch.float64);cross=torch.zeros((784,10),dtype=torch.float64)
        weight=torch.tensor([1/(len(names)*int((st==n).sum())) for n in st],dtype=torch.float64)
        for repeat in range(24):
            if time.monotonic()-started>240:raise RuntimeError('Fixed240s projection bound reached')
            values=source if repeat==0 else augment_train(source,mean,std)
            for ix in torch.arange(len(source)).split(256):
                z=values[ix]@transform.T;target=teacher(z);target=target-target.mean(1,keepdim=True);zz=z.double();ww=weight[ix,None]/24
                gram+=zz.T@(ww*zz);cross+=zz.T@(ww*target.double())
        return gram,cross
    gram,cross=statistics(fit);target=teacher(x[tr][held]@transform.T).double();rows=[]
    for strength in (.0001,.001,.01,.1):
        w=solve_projection(gram,cross,strength);pred=(x[tr][held]@transform.T).double()@w.T
        loss=F.kl_div(F.log_softmax(pred/2,1),F.softmax(target/2,1),reduction='none').sum(1)*4
        metric=float(np.mean([float(loss[torch.tensor(tt[held]==n)].mean()) for n in names]));rows.append(dict(strength=strength,held_TRAIN_teacher_KL=metric))
    chosen=min(rows,key=lambda r:(r['held_TRAIN_teacher_KL'],-r['strength']))['strength'];gram,cross=statistics(np.arange(len(tr)));w=solve_projection(gram,cross,chosen)
    folded=(w@transform.double()).float();baseline=evaluate(model.shared_head,x,y,tasks,te);model.shared_head.weight.copy_(folded);candidate=evaluate(model.shared_head,x,y,tasks,te)
    retained=score(candidate)<=score(baseline);weights={'best':cp['model']['shared_head.weight'].clone() if retained else folded,'last':folded};result={}
    partial=report['teacher_uses_EVAL_derivatives'];exposed=np.array([i for i in te if partial and int(ids[i]) in protocol['selection_source_ids'][str(tasks[i])]],dtype=int);unexposed=np.setdiff1d(te,exposed)
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
