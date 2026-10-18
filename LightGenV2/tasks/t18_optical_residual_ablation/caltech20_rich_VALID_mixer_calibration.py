"""VALID-selected one-stage alpha calibration; same physical hybrid graph."""
import argparse,os,subprocess
from pathlib import Path
import numpy as np
import torch
import caltech20_rich_stagewise_electronic_diagnosis as diagnosis
from audit_caltech20_rich_pair50_alpha import metrics,agree,verify_predictions
HERE,ROOT,s,base,engine,resources=diagnosis.HERE,diagnosis.ROOT,diagnosis.s,diagnosis.base,diagnosis.engine,diagnosis.resources
FILES=('caltech20_rich_VALID_mixer_calibration.py','check_caltech20_rich_VALID_mixer_calibration.py','audit_caltech20_rich_VALID_mixer_calibration.py','CALTECH20_RICH_VALID_MIXER_CALIBRATION_PROTOCOL_20261018.md')
def config():return dict(run='caltech20_rich20_VALID_one_stage_alpha_calibration_s17_20261018',actions={'4':dict(stage=1,alpha=.85),'6':dict(stage=5,alpha=.75)},new_inference_parameters=0,optimizer_steps=0,VAL_selected=True,TEST_informed_development=True)
OUT=HERE/'runs/simulation'/config()['run'];CPU=HERE/'runs/smoke/caltech20_rich_VALID_mixer_calibration_cpu_20261018'
def sources():
    z=diagnosis.sources()
    for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return z
def identity():
    z=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();engine.verify_sources(z,commit);return dict(sources=z,git_commit=commit)
def claim(p,phase):s.save(p,dict(identity(),pid=os.getpid(),phase=phase,automatic_retry=False),exclusive=True)
def parent(depth):return diagnosis.closure.incumbent(depth)
def decision(depth):
    f=diagnosis.OUT/f'L{depth}/diagnosis.json';q=s.read(f);rec=s.read(diagnosis.OUT/f'L{depth}.completed.json');ex=s.read(diagnosis.OUT/f'L{depth}.producer_exit.json');assert rec['passed'] and rec['diagnosis_SHA256']==s.sha(f) and ex['returncode']==0 and ex['actual_exit'] and not Path('/proc',str(ex['pid']),'cmdline').exists();assert q['passed'] and q['sources']==diagnosis.sources() and q['state_unchanged'] and q['TEST_forward_calls']==q['optimizer_steps']==0 and q['VAL_full383'];p,_,_=parent(depth);assert q['parent']==p
    def score(k):v=q['records'][k]['val'];return (int(round(383*v['accuracy'])),-v['balanced_nll'])
    best=max(q['records'],key=score);action=config()['actions'][str(depth)];expected=f"stage{action['stage']}_alpha{action['alpha']:.2f}";assert best==expected and score(best)[0]>score('retained')[0];return best,q['records'][best],s.sha(f)
def state(depth):
    q,pt,_=parent(depth);z=diagnosis.closure.remap(pt['model'],q['alpha_low']);a=config()['actions'][str(depth)];z[f"fusion_logits.{a['stage']}"]=torch.full_like(z[f"fusion_logits.{a['stage']}"],a['alpha']);return q,pt,z
def ready():
    q=s.read(CPU/'checks.json');assert q['passed'] and q['sources']==sources();identity();diagnosis.closure.old.controls()
    for d in (4,6):decision(d)
def configure(gpu):
    assert gpu in resources.GPU.values() and os.environ['CUDA_VISIBLE_DEVICES']==gpu;slots=resources.slots();assert slots['total']<=5 and slots['counts'][gpu]==1 and any(j['pid']==os.getpid() and j['gpu_uuid']==gpu for j in slots['jobs']);s.gpu_budget.ALLOWED.update(resources.GPU.values());return s.gpu_budget.configure(torch,gpu,shared=True,memory_gib=2)
def fit(depth,gpu):
    ready();folder=OUT/f'L{depth}';assert not folder.exists();claim(folder.with_name(folder.name+'.fit_claim.json'),'single_VALID_selected_alpha');folder.mkdir(parents=True);policy=configure(gpu);torch.set_num_threads(4);q,pt,z=state(depth);choice,expected,diagsha=decision(depth);model=diagnosis.physics.Alpha40MoE(depth).cuda().eval();model.load_state_dict(z);model.requires_grad_(False);before=s.digest(model.state_dict());data={r:base.load(r) for r in ('train','val')};reference=diagnosis.Probe(depth).cuda().eval();reference.load_state_dict(diagnosis.closure.remap(pt['model'],q['alpha_low']));reference.force=[None]*depth;action=config()['actions'][str(depth)];reference.force[action['stage']]=action['alpha'];indices=np.array([int((data['train'][1]==k).nonzero()[0]) for k in range(20)]);pe=ce=0.
    with torch.no_grad():
        for off in range(0,20,8):x=base.encode(data['train'][0][indices[off:off+8]].cuda(),None,100);p,c=model(x);a,b=reference(x);pe=max(pe,float((p-a).abs().max()));ce=max(ce,float((c-b).abs().max()))
    assert pe<=3e-6 and ce<=3e-6;del reference;torch.cuda.empty_cache();result={}
    for role in ('train','val'):result[role],pred=base.evaluate(model,data[role]);base.save_np(folder/('candidate_'+role+'.npz'),pred)
    assert result['val']['accuracy']==expected['val']['accuracy'] and abs(result['val']['balanced_nll']-expected['val']['balanced_nll'])<=3e-6 and before==s.digest(model.state_dict());metadata=dict(identity(),parent=q,config=config(),chosen_scenario=choice,diagnosis_SHA256=diagsha,gpu_uuid=gpu,policy=policy,optimizer_steps=0,new_joint_epochs=0,actual_joint_budget=q['joint_budget_epochs'],selected_optical_epoch=q['selected_optical_epoch'],numeric_gate=dict(all20=True,probability_maxabs=pe,capture_maxabs=ce,optimizer_steps=0,TEST_forward_calls=0));s.save(folder/'metadata.json',metadata,exclusive=True);engine.save_pt(folder/'candidate.pt',dict(model={n:v.detach().cpu() for n,v in model.state_dict().items()},metrics=result,contract=model.contract(),sources=sources(),config=config(),epoch=q['selected_optical_epoch'],VAL_selected_alpha_calibration=True,parent=q));s.save(folder/'fit_result.json',dict(passed=True,metrics=result,candidate_sha256=s.sha(folder/'candidate.pt'),model_digest=before,optimizer_steps=0,new_joint_epochs=0,actual_joint_budget=q['joint_budget_epochs'],selected_optical_epoch=q['selected_optical_epoch']),exclusive=True)
def test(depth,gpu):
    ready();rec=s.read(OUT/'fit_audit_completed.json');assert rec['actual_audit_returncode']==0 and rec['audit_sha256']==s.sha(OUT/'fit_science/audit.json');folder=OUT/f'L{depth}';claim(folder/'once_test_claim.json','one_new_parameter_digest_TEST');fit=s.read(folder/'fit_result.json');assert fit['candidate_sha256']==s.sha(folder/'candidate.pt');configure(gpu);torch.set_num_threads(4);pt=torch.load(folder/'candidate.pt',map_location='cpu',weights_only=False);model=diagnosis.physics.Alpha40MoE(depth).cuda().eval();model.load_state_dict(pt['model']);model.requires_grad_(False);assert s.digest(model.state_dict())==fit['model_digest'];result,pred=base.evaluate(model,base.load('test'));base.save_np(folder/'candidate_test.npz',pred);s.save(folder/'test_result.json',dict(passed=True,metrics=result,model_digest=fit['model_digest'],new_TEST_evaluations=1,used_to_choose_alpha=False,TEST_informed_development=True),exclusive=True)
def audit(phase):
    assert os.environ['CUDA_VISIBLE_DEVICES']=='';torch.set_num_threads(4);dest=OUT/(phase+'_science');assert not dest.exists();claim(OUT/(phase+'_audit_claim.json'),'independent_single_alpha_saved_'+phase);checks={}
    for depth in (4,6):
        f=OUT/f'L{depth}';ex=s.read(OUT/f'L{depth}.{phase}_exit.json');assert ex['returncode']==0 and ex['actual_exit'] and not Path('/proc',str(ex['pid']),'cmdline').exists();meta=s.read(f/'metadata.json');fit=s.read(f/'fit_result.json');pt=torch.load(f/'candidate.pt',map_location='cpu',weights_only=False);q,old,z=state(depth);key,expected,sha=decision(depth);assert meta['sources']==pt['sources']==sources() and meta['config']==pt['config']==config() and meta['parent']==q and meta['chosen_scenario']==key and meta['diagnosis_SHA256']==sha;engine.verify_sources(meta['sources'],meta['git_commit']);assert fit['candidate_sha256']==s.sha(f/'candidate.pt') and s.digest(pt['model'])==fit['model_digest'];assert all(torch.equal(pt['model'][n],v) for n,v in z.items())
        probe=s.read(diagnosis.OUT/f'L{depth}/diagnosis.json');rank=[]
        for index,(label,row) in enumerate(probe['records'].items()):rank.append((int(np.trace(np.asarray(row['val']['confusion_matrix']))),-row['val']['balanced_nll'],-index,label))
        assert max(rank)[3]==key and max(rank)[0]>int(np.trace(np.asarray(probe['records']['retained']['val']['confusion_matrix'])))
        original=diagnosis.closure.remap(old['model'],q['alpha_low']);a=config()['actions'][str(depth)];changed={n for n in pt['model'] if not torch.equal(original[n],pt['model'][n])};assert changed=={f"fusion_logits.{a['stage']}"} and float(pt['model'][f"fusion_logits.{a['stage']}"])==a['alpha'];assert sum(v.numel() for n,v in z.items() if n.startswith('linear.'))==5140 and fit['optimizer_steps']==fit['new_joint_epochs']==0
        gate=meta['numeric_gate'];assert gate['all20'] and gate['probability_maxabs']<=3e-6 and gate['capture_maxabs']<=3e-6
        for role in (('train','val') if phase=='fit' else ('test',)):
            actual,detail=verify_predictions(f/('candidate_'+role+'.npz'),pt['model'],role,base.manifest());agree(actual,fit['metrics'][role] if role!='test' else s.read(f/'test_result.json')['metrics'])
        if phase=='fit':assert fit['metrics']['val']['accuracy']==expected['val']['accuracy'] and abs(fit['metrics']['val']['balanced_nll']-expected['val']['balanced_nll'])<=3e-6
        else:assert s.read(f/'test_result.json')['new_TEST_evaluations']==1 and not s.read(f/'test_result.json')['used_to_choose_alpha']
        checks[str(depth)]=dict(PT_SHA256=fit['candidate_sha256'],parent_SHA256=q['sha256'],only_changed_state_key=next(iter(changed)),selected_alpha=a['alpha'],no_other_weight_change=True)
    diagnosis.closure.old.controls();dest.mkdir();s.save(dest/'audit.json',dict(identity(),passed=True,checks=checks,phase=phase,optical_model_forward_calls=0,TEST_images_read=0,independent_saved_head_math=True,new_inference_parameters=0,TEST_informed_development=True),exclusive=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','test','audit_fit','audit_test'],required=True);p.add_argument('--depth',type=int,choices=[4,6]);p.add_argument('--gpu');a=p.parse_args()
    try:audit(a.phase.split('_')[1]) if a.phase.startswith('audit_') else globals()[a.phase](a.depth,a.gpu)
    except BaseException as e:
        mark=OUT/(f'L{a.depth}.'+a.phase+'_failure.json') if OUT.exists() else OUT.with_name(OUT.name+'.'+a.phase+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
