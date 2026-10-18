"""TRAIN-only fixed teacher geometry aligned to retained optical feature axes."""
import argparse,os,subprocess
from pathlib import Path
import numpy as np
import torch
import caltech20_rich_dualview_hosthash_repair as prior
HERE,ROOT,s,engine,base=prior.HERE,prior.ROOT,prior.s,prior.engine,prior.base
resources=prior.resources
FILES=('caltech20_rich_teacher_alignment.py','caltech20_rich_teacher_alignment_profile.json','check_caltech20_rich_teacher_alignment.py','CALTECH20_RICH_TEACHER_ALIGNMENT_PROTOCOL_20261018.md')
def config():return s.read(HERE/FILES[1])
OUT=HERE/'runs/smoke'/config()['run'];CPU=HERE/'runs/smoke/caltech20_rich_teacher_alignment_cpu_20261018'
def sources():
    z=prior.sources()
    for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return z
def identity():
    z=sources();git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();engine.verify_sources(z,git);return dict(sources=z,git_commit=git)
def claim(p,phase):s.save(p,dict(identity(),pid=os.getpid(),phase=phase,automatic_retry=False),exclusive=True)
def ln(x):return (x-x.mean(1,keepdims=True))/np.sqrt(x.var(1,keepdims=True)+1e-5)
def softmax(x):x=x-x.max(1,keepdims=True);p=np.exp(x);return p/p.sum(1,keepdims=True)
def teacher():
    prior.old.kd.teacher_ready();t=prior.old.kd.teacher;path=t.OUT/'train_features.npz';saved=s.read(t.OUT/'fit_result.json');assert saved['feature_sha256']['train']==s.sha(path)
    with np.load(path,allow_pickle=False) as z:features,labels,ids=[z[k].copy() for k in ('features','labels','ids')]
    assert ids.tolist()==base.manifest()['split_ids']['train'] and len(ids)==3141
    state=torch.load(t.OUT/'selected_head.pt',map_location='cpu',weights_only=False);scaled=(features-state['mean'])/state['std'];q,valid=prior.old.kd.teacher_targets()
    return scaled.astype(np.float64),labels,ids,valid.astype(np.float64),dict(teacher_train_features_sha256=s.sha(path),teacher_head_sha256=s.sha(t.OUT/'selected_head.pt'))
def student(d):
    q,pt=prior.diagnosis.parents.parent(d);p=Path(q['path']);path=p.parent/('selected_train.npz' if d==2 else 'candidate_train.npz')
    with np.load(path,allow_pickle=False) as z:features,labels,ids=[z[k].copy() for k in ('features','labels','ids')]
    assert ids.tolist()==base.manifest()['split_ids']['train'];w=pt['model']['linear.weight'].double().numpy();b=pt['model']['linear.bias'].double().numpy()
    return features.astype(np.float64),labels,ids,w,b,dict(parent=q,TRAIN_features_sha256=s.sha(path))
def prepare():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not OUT.exists();proof=s.read(CPU/'checks.json');assert proof['passed'] and proof['sources']==sources();identity();prior.old.controls();claim(OUT.with_name(OUT.name+'.claim.json'),'TRAIN_only_fixed_teacher_targets');OUT.mkdir();torch.set_num_threads(4)
    x,y,ids,valid,ti=teacher();mean=x.mean(0);center=x-mean;cov=center.T@center/len(x);values,vectors=np.linalg.eigh(cov);order=np.argsort(values)[::-1];basis=vectors[:,order[:256]]
    for j in range(256):basis[:,j]*=1 if basis[np.argmax(np.abs(basis[:,j])),j]>=0 else -1
    projected=ln(center@basis);engine.save_pt(OUT/'teacher_basis.pt',dict(mean=mean,basis=basis,eigenvalues=values[order],sources=sources(),teacher=ti))
    checks={};counts=np.asarray(base.manifest()['supports']['train'],float);weights=(len(y)/(20*counts))**config()['class_weight_exponent'];weights*=len(y)/(counts@weights)
    for d in (2,4,6):
        f,labels,sids,w,b,parent=student(d);assert np.array_equal(y,labels) and np.array_equal(ids,sids);u,sv,vt=np.linalg.svd(projected.T@f,full_matrices=False);rotation=u@vt;target=ln(projected@rotation);assert np.max(np.abs(rotation.T@rotation-np.eye(256)))<1e-8
        p=softmax(f@w.T+b);soft=np.eye(20)[y]*(1-config()['label_smoothing'])+config()['label_smoothing']/20;weighted=soft*weights[None];gc=((p*weighted.sum(1,keepdims=True)-weighted)@w)/len(y);gf=valid[:,None]*(f-target)/len(y)
        scale=min(config()['coefficient_cap'],config()['classification_feature_gradient_fraction']*np.linalg.norm(gc)/max(np.linalg.norm(gf),1e-12));pred=softmax(target@w.T+b)
        arrays=dict(features=target.astype(np.float32),labels=y,ids=ids,valid=valid.astype(np.float32));base.save_np(OUT/f'L{d}_train_targets.npz',arrays);engine.save_pt(OUT/f'L{d}_rotation.pt',dict(rotation=rotation,singular_values=sv,scale=scale,parent=parent,sources=sources()))
        checks[str(d)]=dict(parent=parent,targets_sha256=s.sha(OUT/f'L{d}_train_targets.npz'),rotation_sha256=s.sha(OUT/f'L{d}_rotation.pt'),parent_TRAIN_accuracy=float((p.argmax(1)==y).mean()),teacher_target_parent_head_TRAIN_accuracy=float((pred.argmax(1)==y).mean()),rotation_orthogonality_maxabs=float(np.max(np.abs(rotation.T@rotation-np.eye(256)))),normalized_teacher_student_cosine=float((ln(f)*target).mean()),classification_feature_gradient_norm=float(np.linalg.norm(gc)),unscaled_alignment_gradient_norm=float(np.linalg.norm(gf)),alignment_scale=scale,valid_teacher_fraction=float(valid.mean()),effective_initial_gradient_fraction=float(scale*np.linalg.norm(gf)/np.linalg.norm(gc)))
    prior.old.controls();s.save(OUT/'result.json',dict(identity(),passed=True,teacher=ti,checks=checks,explained_teacher_variance=float(values[order[:256]].sum()/values.clip(0).sum()),teacher_basis_sha256=s.sha(OUT/'teacher_basis.pt'),TRAIN3141_only=True,VAL_images_read=0,TEST_images_read=0,new_model_forwards=0,optimizer_steps=0,new_inference_parameters=0),exclusive=True)
def audit():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not (OUT/'science').exists();ex=s.read(OUT/'producer_exit.json');assert ex['returncode']==0 and ex['actual_exit'] and not Path('/proc',str(ex['pid']),'cmdline').exists();r=s.read(OUT/'result.json');assert r['sources']==sources();engine.verify_sources(r['sources'],r['git_commit']);claim(OUT/'audit_claim.json','independent_saved_TRAIN_projection_and_derivative')
    x,y,ids,valid,ti=teacher();pt=torch.load(OUT/'teacher_basis.pt',map_location='cpu',weights_only=False);assert s.sha(OUT/'teacher_basis.pt')==r['teacher_basis_sha256'] and pt['teacher']==ti and np.allclose(pt['mean'],x.mean(0),atol=1e-12,rtol=0)
    center=x-pt['mean'];basis=pt['basis'];cov=center.T@center/len(x);assert np.allclose(basis.T@basis,np.eye(256),atol=1e-8,rtol=0) and np.allclose(cov@basis,basis*pt['eigenvalues'][:256][None],atol=1e-8,rtol=1e-7);t=ln(center@basis);checks={}
    for d in (2,4,6):
        f,labels,sids,w,b,parent=student(d);a=torch.load(OUT/f'L{d}_rotation.pt',map_location='cpu',weights_only=False);q=r['checks'][str(d)];assert a['sources']==sources() and a['parent']==parent and q['rotation_sha256']==s.sha(OUT/f'L{d}_rotation.pt') and q['targets_sha256']==s.sha(OUT/f'L{d}_train_targets.npz')
        rot=a['rotation'];cross=t.T@f;assert np.allclose(rot.T@rot,np.eye(256),atol=1e-8,rtol=0);symmetric=rot.T@cross;assert np.max(np.abs(symmetric-symmetric.T))<1e-6 and np.linalg.eigvalsh((symmetric+symmetric.T)/2).min()>-1e-6
        target=ln(t@rot)
        with np.load(OUT/f'L{d}_train_targets.npz',allow_pickle=False) as saved:assert np.array_equal(saved['ids'],ids) and np.array_equal(saved['labels'],y) and np.array_equal(saved['valid'],valid.astype(np.float32)) and np.max(np.abs(saved['features']-target))<1e-6
        counts=np.asarray(base.manifest()['supports']['train'],float);weights=(len(y)/(20*counts))**config()['class_weight_exponent'];weights*=len(y)/(counts@weights);p=softmax(f@w.T+b);soft=np.eye(20)[y]*(1-config()['label_smoothing'])+config()['label_smoothing']/20;gradient=((p*(soft*weights[None]).sum(1,keepdims=True)-soft*weights[None])@w)/len(y);feature_gradient=valid[:,None]*(f-target)/len(y);scale=min(config()['coefficient_cap'],config()['classification_feature_gradient_fraction']*np.linalg.norm(gradient)/max(np.linalg.norm(feature_gradient),1e-12));assert abs(scale-a['scale'])<1e-12;checks[str(d)]=dict(rotation_optimal_and_orthogonal=True,gradient_budget_scale=scale,target_rows=len(y))
    (OUT/'science').mkdir();s.save(OUT/'science/audit.json',dict(identity(),passed=True,checks=checks,TRAIN_only=True,model_forwards=0,TEST_images_read=0,optimizer_steps=0),exclusive=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['prepare','audit'],required=True);a=p.parse_args()
    try:globals()[a.phase]()
    except BaseException as e:
        mark=OUT/('failure.json' if a.phase=='prepare' else 'audit_failure.json') if OUT.exists() else OUT.with_name(OUT.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
