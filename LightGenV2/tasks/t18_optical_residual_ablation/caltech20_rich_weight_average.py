"""One deployable network: VALID-selected averages of verified nearby states."""
import argparse,os,subprocess
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
import caltech20_rich_postCCD_electronic_adapt as previous
import caltech20_rich_prototype_consistency as prototype
import caltech20_rich_L2_strongview_namespace_repair as strong2
import caltech20_rich_prototype_strongview as strong46
import caltech20_rich_L4_prototype_continue as continue4
import caltech20_rich_L6_shared_supervision as shared6
from audit_caltech20_rich_pair50_alpha import metrics,agree,verify_predictions
HERE,ROOT,s,base,engine,resources=previous.HERE,previous.ROOT,previous.s,previous.base,previous.engine,previous.resources
FILES=('caltech20_rich_weight_average.py','check_caltech20_rich_weight_average.py','audit_caltech20_rich_weight_average.py','CALTECH20_RICH_WEIGHT_AVERAGE_PROTOCOL_20261018.md')
def config():return dict(run='caltech20_rich20_sameNetwork_VALID_weight_average246_20261018',mixture_fractions=[0.,.25,.5,1.],selection='maximum_VALID_correct_then_minimum_balanced_NLL_then_smallest_fraction',new_optimizer_steps=0,new_joint_epochs=0,new_inference_parameters=0,prediction_ensemble=False,TEST_informed_development=True,independent_generalization=False)
OUT=HERE/'runs/simulation'/config()['run'];CPU=HERE/'runs/smoke/caltech20_rich_weight_average_CPU_20261018'
def sources():
    z=previous.sources()
    for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return z
def identity():
    z=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();engine.verify_sources(z,commit);return dict(sources=z,git_commit=commit)
def claim(p,phase):s.save(p,dict(identity(),pid=os.getpid(),phase=phase,automatic_retry=False),exclusive=True)
def parent(depth):return previous.parent(depth)
@lru_cache(None)
def bank(depth):
    q,pt,_=parent(depth);initial=previous.initial_state(depth);rows=[dict(path=q['path'],sha256=q['sha256'],role='canonical_unchanged',selected_epoch=q['selected_optical_epoch'],joint_budget=q['joint_budget_epochs'])];states=[initial]
    extra={2:[(prototype,'L2_prototype'),(strong2,'L2_prototype_strong')],4:[(continue4,'L4_prototype_continue'),(strong46,'L4_prototype_strong')],6:[(prototype,'L6_prototype'),(shared6,'L6_shared_head')]}[depth]
    for module,key in extra:
        assert key in module.config()['cases'],(module.__name__,key,module.config()['cases']);folder=module.OUT/key;f=s.read(folder/'fit_result.json');rec=s.read(folder/'audit_completed.json');ex=s.read(module.OUT/(key+'.producer_exit.json'));meta=s.read(folder/'metadata.json');assert ex['returncode']==0 and ex['actual_exit'] and not Path('/proc',str(ex['pid']),'cmdline').exists();assert rec['actual_audit_returncode']==0 and rec['audit_sha256']==s.sha(folder/'science/audit.json') and s.read(folder/'science/audit.json')['passed'];assert meta['sources']==module.sources();engine.verify_sources(meta['sources'],meta['git_commit']);p=folder/'last.pt';assert f['last_sha256']==s.sha(p);last=torch.load(p,map_location='cpu',weights_only=False);state={n:v.detach().cpu().clone() for n,v in last['ema'].items()};assert state.keys()==initial.keys() and all(state[n].shape==v.shape and state[n].dtype==v.dtype and torch.isfinite(state[n]).all() for n,v in initial.items());assert all(.4001<float(v.float())<.95 for n,v in state.items() if n.startswith('fusion_logits.'));states.append(state);rows.append(dict(path=str(p),sha256=s.sha(p),role='verified_endpoint_EMA',source_git=meta['git_commit'],science_SHA256=rec['audit_sha256'],actual_joint_budget=f['previous_budget_epochs']+f['actual_new_epochs'],epoch=last['epoch']))
    return states,rows
def state(depth,fraction):
    assert fraction in config()['mixture_fractions'];states,_=bank(depth);result={}
    for n,v in states[0].items():
        if not v.is_floating_point():assert all(torch.equal(v,z[n]) for z in states);result[n]=v.clone();continue
        result[n]=((1-fraction)*v.double()+fraction*.5*(states[1][n].double()+states[2][n].double())).to(v.dtype)
    return result
def initialize(depth,fraction):
    engine.seed(17);model=previous.physics.Alpha40MoE(depth,.3);model.load_state_dict(state(depth,fraction));model.requires_grad_(False);return model.eval()
def ready():
    q=s.read(CPU/'checks.json');assert q['passed'] and q['sources']==sources();identity();previous.closure.diagnosis.closure.old.controls();rec=s.read(previous.OUT/'test_audit_completed.json');assert rec['actual_audit_returncode']==0 and rec['audit_sha256']==s.sha(previous.OUT/'test_science/audit.json')
def configure(gpu):return previous.configure(gpu)
def score(row):return (int(np.trace(np.asarray(row['val']['confusion_matrix']))),-row['val']['balanced_nll'],-row['fraction'])
def fit(depth,gpu):
    ready();folder=OUT/f'L{depth}';assert not folder.exists();claim(folder.with_name(folder.name+'.fit_claim.json'),'single_network_weight_average_VALID');folder.mkdir(parents=True);policy=configure(gpu);torch.set_num_threads(4);q,pt,pred=parent(depth);states,parents=bank(depth);data=base.load('train');val=base.load('val');model=initialize(depth,0).cuda();mod=__import__(q['module'],fromlist=[q['class_name']]);ref=getattr(mod,q['class_name'])(depth,.3).cuda().eval();ref.load_state_dict(pt['model']);idx=np.array([int((data[1]==k).nonzero()[0]) for k in range(20)]);pe=ce=0.
    with torch.no_grad():
        for off in range(0,20,8):x=base.encode(data[0][idx[off:off+8]].cuda(),None,100);a,b=model(x);p,c=ref(x);pe=max(pe,float((a-p).abs().max()));ce=max(ce,float((b-c).abs().max()))
    assert pe<=3e-6 and ce<=3e-6;del ref;torch.cuda.empty_cache();rows=[]
    for i,fraction in enumerate(config()['mixture_fractions']):
        z=state(depth,fraction);model.load_state_dict(z);before=s.digest(model.state_dict());metric,p=base.evaluate(model,val);base.save_np(folder/f'VALID_choice{i}.npz',p);assert before==s.digest(model.state_dict());rows.append(dict(fraction=fraction,val=metric,model_digest=before))
    selected=max(range(len(rows)),key=lambda i:score(rows[i]));model.load_state_dict(state(depth,rows[selected]['fraction']));train,p=base.evaluate(model,data);base.save_np(folder/'candidate_train.npz',p);base.save_np(folder/'candidate_val.npz',previous.load_np(folder/f'VALID_choice{selected}.npz'));result=dict(train=train,val=rows[selected]['val']);meta=dict(identity(),config=config(),parents=parents,gpu_uuid=gpu,policy=policy,numeric_gate=dict(all20=True,probability_maxabs=pe,capture_maxabs=ce),new_optimizer_steps=0,new_joint_epochs=0,new_inference_parameters=0);s.save(folder/'metadata.json',meta,exclusive=True);engine.save_pt(folder/'candidate.pt',dict(model={n:v.detach().cpu() for n,v in model.state_dict().items()},metrics=result,sources=sources(),config=config(),contract=model.contract(),weight_average=True,selected_fraction=rows[selected]['fraction'],contributing_parents=parents));s.save(folder/'fit_result.json',dict(passed=True,selected_index=selected,selected_fraction=rows[selected]['fraction'],records=rows,metrics=result,model_digest=s.digest(model.state_dict()),candidate_SHA256=s.sha(folder/'candidate.pt'),parent_unchanged=selected==0,optimizer_steps=0,new_joint_epochs=0),exclusive=True)
def test(depth,gpu):
    ready();rec=s.read(OUT/'fit_audit_completed.json');assert rec['actual_audit_returncode']==0 and rec['audit_sha256']==s.sha(OUT/'fit_science/audit.json');folder=OUT/f'L{depth}';claim(folder/'once_test_claim.json','single_VALID_selected_weight_average_TEST');fit=s.read(folder/'fit_result.json');assert fit['candidate_SHA256']==s.sha(folder/'candidate.pt');configure(gpu);torch.set_num_threads(4);pt=torch.load(folder/'candidate.pt',map_location='cpu',weights_only=False);same=fit['model_digest']==s.digest(previous.initial_state(depth))
    if same:p=parent(depth)[2]['test'];metric=metrics(p['labels'],p['probabilities'])
    else:model=initialize(depth,fit['selected_fraction']).cuda();metric,p=base.evaluate(model,base.load('test'))
    base.save_np(folder/'candidate_test.npz',p);s.save(folder/'test_result.json',dict(passed=True,metrics=metric,model_digest=fit['model_digest'],new_TEST_models=0 if same else 1,parent_TEST_reused=same,TEST_used_to_select_fraction=False,TEST_informed_development=True,independent_generalization=False),exclusive=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','test'],required=True);p.add_argument('--depth',choices=[2,4,6],type=int,required=True);p.add_argument('--gpu',required=True);a=p.parse_args()
    try:globals()[a.phase](a.depth,a.gpu)
    except BaseException as e:
        mark=OUT/(f'L{a.depth}.'+a.phase+'_failure.json') if OUT.exists() else OUT.with_name(OUT.name+'.'+a.phase+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
