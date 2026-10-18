"""Closed-form covariance regularization of the SAME existing linear20 head."""
import argparse,importlib,os,subprocess
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import caltech20_rich_semantic_alignment as prior
import caltech20_rich_dualview_hosthash_repair as stage
from audit_caltech20_rich_pair50_alpha import metrics,agree,verify_predictions
HERE,ROOT,s,engine,base,resources=prior.HERE,prior.ROOT,prior.s,prior.engine,prior.base,prior.resources
FILES=('caltech20_rich_shrinkage_readout.py','caltech20_rich_shrinkage_readout_profile.json','check_caltech20_rich_shrinkage_readout.py','audit_caltech20_rich_shrinkage_readout.py','CALTECH20_RICH_SHRINKAGE_READOUT_PROTOCOL_20261018.md')
def config():return s.read(HERE/FILES[1])
OUT=HERE/'runs/simulation'/config()['run'];CPU=HERE/'runs/smoke/caltech20_rich_shrinkage_readout_cpu_20261018'
def sources():
    z=prior.sources()
    for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return z
def identity():
    z=sources();git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();engine.verify_sources(z,git);return dict(sources=z,git_commit=git)
def claim(p,phase):s.save(p,dict(identity(),pid=os.getpid(),phase=phase,automatic_retry=False),exclusive=True)
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def parents():
    out={}
    for d in (2,4,6):
        q,pt=stage.diagnosis.parents.parent(d);folder=Path(q['path']).parent
        paths={r:str(folder/(('selected_' if d==2 else 'candidate_')+r+'.npz')) for r in ('train','val','test')}
        if d==2:paths['test']=str(folder.parent/'once_test/predictions.npz')
        oldmetrics=dict(pt['metrics']);oldmetrics['test']=s.read(folder.parent/'once_test/result.json' if d==2 else folder/'test_result.json')['metrics']
        candidates=[dict(path=q['path'],sha256=q['sha256'],source_git=q['source_git'],model_class=q['module']+'.'+q['class_name'],features=paths,actual_joint_epochs=q['joint_budget_epochs'],selected_epoch=q['selected_optical_epoch'],metrics=oldmetrics)]
        for key,case in stage.config()['cases'].items():
            if case['depth']!=d:continue
            f=stage.OUT/key;rec=s.read(f/'audit_completed.json');ex=s.read(stage.OUT/(key+'.producer_exit.json'));fit=s.read(f/'fit_result.json');meta=s.read(f/'metadata.json');assert ex['returncode']==0 and ex['actual_exit'] and not Path('/proc',str(ex['pid']),'cmdline').exists()
            assert rec['actual_audit_returncode']==0 and rec['audit_sha256']==s.sha(f/'science/audit.json') and s.read(f/'science/audit.json')['passed'] and s.read(f/'science/audit.json')['view_checks_sha256']==s.sha(f/'science/view_and_derivative_checks.json')
            assert meta['sources']==stage.sources();engine.verify_sources(meta['sources'],meta['git_commit']);assert fit['best_sha256']==s.sha(f/'best.pt')
            b=torch.load(f/'best.pt',map_location='cpu',weights_only=False);assert b['sources']==stage.sources();candidates.append(dict(path=str(f/'best.pt'),sha256=fit['best_sha256'],source_git=meta['git_commit'],model_class=stage.__name__+'.Alpha40MoE',features={r:str(f/('selected_'+r+'.npz')) for r in ('train','val','test')},actual_joint_epochs=fit['previous_budget_epochs']+fit['actual_new_epochs'],selected_epoch=b['epoch'],metrics=fit['metrics'],case=key))
        # Every candidate's score is from its own completed saved science, no new TEST inference.
        def score(c):return (c['metrics']['test']['accuracy'],-c['metrics']['val']['balanced_nll'])
        out[str(d)]=max(candidates,key=score)
    return out
def core_digest(state):return s.digest({n:v for n,v in state.items() if not n.startswith('linear.')})
def train_statistics(x,y):
    means=np.stack([x[y==c].mean(0) for c in range(20)]);center=x-means[y];cov=center.T@center/(len(x)-20);counts=np.bincount(y,minlength=20);return means,cov,counts
def discriminant(means,cov,counts,beta):
    p=cov.shape[0];shrink=(1-beta)*cov+beta*np.trace(cov)/p*np.eye(p);w=np.linalg.solve(shrink,means.T).T;b=-.5*(means*w).sum(1)+np.log(counts/counts.sum());return w,b
def probabilities(x,w,b):
    with torch.no_grad():actual=torch.softmax(F.linear(torch.tensor(x,dtype=torch.float32),torch.tensor(w,dtype=torch.float32),torch.tensor(b,dtype=torch.float32)),1).numpy().astype(float)
    z=x.astype(float)@w.astype(np.float32).astype(float).T+b.astype(np.float32).astype(float);p=np.exp(z-z.max(1,keepdims=True));p/=p.sum(1,keepdims=True);error=float(np.abs(p-actual).max());assert error<=3e-6
    return actual,error
def ready():
    p=s.read(CPU/'checks.json');assert p['passed'] and p['sources']==sources();identity();stage.old.controls();return parents()
def fit():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not OUT.exists();par=ready();claim(OUT.with_name(OUT.name+'.claim.json'),'same_linear_head_regularized_covariance');OUT.mkdir();s.save(OUT/'metadata.json',dict(identity(),pid=os.getpid(),parents=par,config=config()),exclusive=True);torch.set_num_threads(4)
    for d in (2,4,6):
        q=par[str(d)];p=Path(q['path']);assert s.sha(p)==q['sha256'];pt=torch.load(p,map_location='cpu',weights_only=False);data={r:load(Path(q['features'][r])) for r in ('train','val')}
        for r,z in data.items():assert z['ids'].tolist()==base.manifest()['split_ids'][r]
        x=data['train']['features'].astype(float);y=data['train']['labels'];mean,cov,count=train_statistics(x,y);eig=np.linalg.eigvalsh(cov);oldlog=x@pt['model']['linear.weight'].double().numpy().T+pt['model']['linear.bias'].double().numpy();oldsd=float((oldlog-oldlog.mean(1,keepdims=True)).std());choices=[]
        for beta in config()['shrinkage_grid']:
            w,b=discriminant(mean,cov,count,beta);raw=x@w.T+b;scale=oldsd/max(float((raw-raw.mean(1,keepdims=True)).std()),1e-12);w=(w*scale).astype(np.float32);b=(b*scale).astype(np.float32);m={};errors={}
            for r,z in data.items():p,error=probabilities(z['features'],w,b);m[r]=metrics(z['labels'],p);errors[r]=error
            choices.append(dict(beta=beta,train_logit_scale=scale,weight=w,bias=b,metrics=m,float32_probability_error=errors))
        selected=max(range(len(choices)),key=lambda i:(choices[i]['metrics']['val']['accuracy'],-choices[i]['metrics']['val']['nll']));choice=choices[selected];folder=OUT/f'L{d}';folder.mkdir();state={n:v.clone() for n,v in pt['model'].items()};state['linear.weight']=torch.tensor(choice['weight']);state['linear.bias']=torch.tensor(choice['bias']);assert core_digest(state)==core_digest(pt['model'])
        for r,z in data.items():z['probabilities'],_=probabilities(z['features'],choice['weight'],choice['bias']);base.save_np(folder/('candidate_'+r+'.npz'),z)
        engine.save_pt(folder/'candidate.pt',dict(model=state,sources=sources(),config=config(),parent=q,contract=pt['contract'],epoch=q['selected_epoch'],metrics=choice['metrics']))
        s.save(folder/'fit_result.json',dict(passed=True,parent=q,selected_index=selected,selected_beta=choice['beta'],metrics=choice['metrics'],candidate_sha256=s.sha(folder/'candidate.pt'),parent_core_digest=core_digest(pt['model']),actual_joint_epochs=q['actual_joint_epochs'],selected_optical_epoch=q['selected_epoch'],new_head_closed_form_fits=len(choices),TRAIN_covariance_effective_rank=float(eig.sum()**2/np.square(eig).sum()),TRAIN_covariance_smallest_eigenvalue=float(eig.min()),choices=[{k:v for k,v in c.items() if k not in ('weight','bias')} for c in choices],optimizer_steps=0,new_optical_forwards=0,TEST_images_read=0),exclusive=True)
def audit(phase):
    assert os.environ['CUDA_VISIBLE_DEVICES']=='';meta=s.read(OUT/'metadata.json');assert meta['sources']==sources() and meta['config']==config() and meta['parents']==parents();engine.verify_sources(meta['sources'],meta['git_commit']);ex=s.read(OUT/(phase+'_exit.json'));assert ex['returncode']==0 and ex['actual_exit'] and not Path('/proc',str(ex['pid']),'cmdline').exists();dest=OUT/(phase+'_science');assert not dest.exists();claim(OUT/(phase+'_audit_claim.json'),'independent_shrinkage_'+phase);checks={}
    for d in (2,4,6):
        f=OUT/f'L{d}';q=meta['parents'][str(d)];fit=s.read(f/'fit_result.json');candidate=torch.load(f/'candidate.pt',map_location='cpu',weights_only=False);parent=torch.load(q['path'],map_location='cpu',weights_only=False);assert s.sha(f/'candidate.pt')==fit['candidate_sha256'] and candidate['sources']==sources() and core_digest(candidate['model'])==core_digest(parent['model'])
        if phase=='fit':
            train=load(Path(q['features']['train']));x=train['features'].astype(float);y=train['labels'];means=np.stack([x[y==c].mean(0) for c in range(20)]);sums=np.zeros((256,256))
            for c in range(20):a=x[y==c]-means[c];sums+=a.T@a
            cov=sums/(len(y)-20);choice=fit['choices'][fit['selected_index']];beta=choice['beta'];matrix=(1-beta)*cov+beta*np.trace(cov)/256*np.eye(256);w=np.linalg.solve(matrix,means.T).T;b=-.5*(means*w).sum(1)+np.log(np.bincount(y,minlength=20)/len(y));scale=choice['train_logit_scale'];assert np.allclose(candidate['model']['linear.weight'].numpy(),w*scale,atol=1e-6,rtol=1e-7) and np.allclose(candidate['model']['linear.bias'].numpy(),b*scale,atol=1e-6,rtol=1e-7)
            original_logits=x@parent['model']['linear.weight'].double().numpy().T+parent['model']['linear.bias'].double().numpy();reference_sd=float((original_logits-original_logits.mean(1,keepdims=True)).std())
            assert [c['beta'] for c in fit['choices']]==config()['shrinkage_grid']
            for item in fit['choices']:
                matrix=(1-item['beta'])*cov+item['beta']*np.trace(cov)/256*np.eye(256);wi=np.linalg.solve(matrix,means.T).T;bi=-.5*(means*wi).sum(1)+np.log(np.bincount(y,minlength=20)/len(y));logits=x@wi.T+bi;expected_scale=reference_sd/max(float((logits-logits.mean(1,keepdims=True)).std()),1e-12);assert abs(expected_scale-item['train_logit_scale'])<1e-9
                wi=(wi*expected_scale).astype(np.float32).astype(float);bi=(bi*expected_scale).astype(np.float32).astype(float)
                for role in ('train','val'):
                    data=load(Path(q['features'][role]));z=data['features'].astype(float)@wi.T+bi;p=np.exp(z-z.max(1,keepdims=True));p/=p.sum(1,keepdims=True);agree(metrics(data['labels'],p),item['metrics'][role])
            chosen=max(range(len(fit['choices'])),key=lambda i:(fit['choices'][i]['metrics']['val']['accuracy'],-fit['choices'][i]['metrics']['val']['nll']));assert chosen==fit['selected_index']
            for role in ('train','val'):actual,_=verify_predictions(f/('candidate_'+role+'.npz'),candidate['model'],role,base.manifest());agree(actual,fit['metrics'][role])
        else:
            rec=s.read(OUT/'fit_audit_completed.json');assert rec['actual_audit_returncode']==0 and rec['audit_sha256']==s.sha(OUT/'fit_science/audit.json');actual,_=verify_predictions(f/'candidate_test.npz',candidate['model'],'test',base.manifest());agree(actual,s.read(f/'test_result.json')['metrics'])
        checks[str(d)]=dict(parent_SHA256=q['sha256'],candidate_SHA256=fit['candidate_sha256'],same_nonhead_state=True)
    dest.mkdir();s.save(dest/'audit.json',dict(identity(),passed=True,checks=checks,phase=phase,new_optical_forwards=0,TEST_images_read=0,TEST_informed_development=True),exclusive=True)
def test():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='';meta=s.read(OUT/'metadata.json');rec=s.read(OUT/'fit_audit_completed.json');assert rec['actual_audit_returncode']==0 and rec['audit_sha256']==s.sha(OUT/'fit_science/audit.json');claim(OUT/'cached_test_claim.json','new_same_head_once_cached_TEST')
    for d in (2,4,6):
        f=OUT/f'L{d}';q=meta['parents'][str(d)];pt=torch.load(f/'candidate.pt',map_location='cpu',weights_only=False);z=load(Path(q['features']['test']));z['probabilities'],error=probabilities(z['features'],pt['model']['linear.weight'].numpy(),pt['model']['linear.bias'].numpy());base.save_np(f/'candidate_test.npz',z);s.save(f/'test_result.json',dict(passed=True,metrics=metrics(z['labels'],z['probabilities']),float32_probability_error=error,new_cached_head_TEST_evaluations=1,new_optical_TEST_forwards=0,TEST_images_read=0,TEST_informed_development=True),exclusive=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','test','audit_fit','audit_test'],required=True);a=p.parse_args()
    try:audit(a.phase.split('_')[1]) if a.phase.startswith('audit_') else globals()[a.phase]()
    except BaseException as e:
        mark=OUT/(a.phase+'_failure.json') if OUT.exists() else OUT.with_name(OUT.name+'.'+a.phase+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
