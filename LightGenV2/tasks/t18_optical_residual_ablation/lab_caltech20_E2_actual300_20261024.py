"""Human-authorized one E2 physical test; actual CPU >=300 alone permits G2.

Prediction is an identity check, NOT a prerequisite for the fresh E2 test.
All response models remain TRAIN-only; acquisition uses the native graph.
"""
import argparse,datetime,json,shutil,traceback
from pathlib import Path
import lab_caltech20_TRAIN_native_mask_20261024 as native
base=native.base


def validate_policy(p):
    assert p['schema']=='TUCam_L4_single_E2_actual300_v1'
    assert p['actual_E2_target']==300 and p['final_target']==340
    assert p['predictive_accuracy_prerequisite'] is False
    assert p['TRAIN']==1600 and p['TEST']==411 and p['VALID']==0
    assert p['TEST_gradient'] is False and p['scheduler'] is False
    assert p['automatic_retry'] is False and p['new_inference_layers']==0
    assert p['candidate_PT_SHA256']=='99c5538aeea0436f96df9c3ae4263b9693ab39b0c75d57d85630365a3d386e41'
    assert p['candidate_report_SHA256']=='b2fab6ce90d76922f8f76a80b819789280cc02315c275f687c68d8eb810feaba'
    assert p['native_mask_PT_SHA256']=='0cf1e41fc11b9d4c24e95ec9079aee5a118373e9af5279ce0b33116267d72da2'
    assert p['native_response_map_SHA256']=='2e8a1aa53f0acc4df02f4ef82609462fa2fba06e14709cce1168d9abb3c47e96'
    assert p['deadline']=='2026-10-24T08:00:00+08:00'
    assert p['observed_recipe']==dict(name='actual300_E2_native_suffix_CE',epochs=36,seconds=1200,lr_scale=1,TRAIN_class_emphasis=.75)
    assert p['final_recipe']==dict(name='actual300_full_physical_original5150',epochs=120,seconds=1200,lr_scale=1,TRAIN_class_emphasis=.75)


def seconds_left(p):
    return (datetime.datetime.fromisoformat(p['deadline'])-datetime.datetime.now(datetime.timezone.utc)).total_seconds()


def can_start_G2(result,target):
    return bool(result['strict_original_CPU'] and result['selected_correct']>=target
                and result['results']['best']['test']['correct']>=target)


def run(a):
    np,torch,m,loop=base.numerical_modules()
    import lab_caltech20_tucam_capture_20261023 as capture
    from lab_caltech20_tucam_E2_mask_retake_20261023 import phase_probe
    p=m.read(a.profile);validate_policy(p)
    old_policy=base.HERE/p['source_admission_policy'];assert m.sha(old_policy)==p['source_admission_policy_SHA256']
    old=m.read(old_policy);native.parent.validate_policy(old)
    for name,digest in p['source_SHA256'].items():assert m.sha(base.HERE/name)==digest
    assert m.sha(base.HERE/old['base_profile'])==old['base_profile_SHA256']
    c=m.read(base.HERE/old['base_profile']);base.validate_config(c);c['intermediate_target']=300
    assert seconds_left(p)>=3600 and not a.out.exists() and not m.LOCK.exists()
    a.out.mkdir();torch.set_num_threads(2);torch.manual_seed(73)
    m.save(a.out/'protocol.json',dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),policy=p,policy_SHA256=m.sha(a.profile),
        actual_E2_target=300,predictive_accuracy_prerequisite=False,TEST_development=True,TEST_gradient=False,scheduler=False,automatic_retry=False))
    old_model,data,dm,fields,paths,_,reference,_,_,old_phase=native.parent.admit(old,c,a.out)
    original=m.state_cpu(old_model);assert list(fields)==['router','0','1'];lineage=base.identities(m,paths)
    folder=base.ROOT/p['candidate_fit'];r=m.read(folder/'report.json');audit=m.read(folder/'independent_audit.json');pt=folder/'best.pt'
    assert m.sha(folder/'report.json')==audit['report_SHA256']==p['candidate_report_SHA256'] and audit['passed']
    assert r['strict_original_state_and_CPU'] and r['all_non_readout_bitwise_frozen'] and r['selected_predictive_correct']==269
    assert m.read(folder.parent.with_suffix('.return.json'))['returncode']==0
    assert m.read(folder/'local_SHA_backup.json')['all_local_remote_SHA_matched']
    assert m.sha(pt)==r['results']['best']['PT_SHA256']==p['candidate_PT_SHA256']
    model,_=m.model_for(pt,p['candidate_PT_SHA256'],4,reference)
    mask=base.ROOT/p['native_mask_fit'];mr=m.read(mask/'report.json');ma=m.read(mask/'independent_audit.json')
    assert m.sha(mask/'report.json')==ma['report_SHA256']==p['native_mask_report_SHA256'] and ma['passed']
    assert mr['actual_epochs']==24 and not mr['selected_E0'] and mr['strict_original_state_and_CPU']
    mask_model,_=m.model_for(mask/'best.pt',p['native_mask_PT_SHA256'],4,reference)
    loop.validate_state(m,model,m.state_cpu(mask_model),{'linear.weight','linear.bias'})
    loop.validate_state(m,model,original,m.allowed_keys(model,3))
    response=base.ROOT/p['native_response_fit'];rr=m.read(response/'report.json');ra=m.read(response/'independent_audit.json')
    assert m.sha(response/'report.json')==ra['report_SHA256']==p['native_response_report_SHA256'] and ra['passed'] and rr['native_response_admitted'] and not rr['TEST_used']
    assert m.sha(response/'TRAIN_native_response.npz')==p['native_response_map_SHA256']
    with np.load(response/'TRAIN_native_response.npz',allow_pickle=False) as z:
        g=torch.from_numpy(z['gain'].copy());b=torch.from_numpy(z['background'].copy());sigma=float(z['sigma_pixels'])
    metric,_=native.score(model,g,b,sigma,data,fields,'test');assert metric['correct']==269
    new_phase=phase_probe(model,data,fields,'2',a.out/'candidate_E2_phase.bmp')
    changed=[k for k in ('phase_sets.5','phase_sets.6','phase_sets.7','phase_sets.8') if not torch.equal(original[k],model.state_dict()[k])]
    assert changed and new_phase!=old_phase and base.identities(m,paths)==lineage
    m.save(a.out/'candidate_admission.json',dict(passed=True,PT_SHA256=m.sha(pt),calibrated_predictive_correct=269,not_physical_accuracy=True,
        predictive_accuracy_prerequisite=False,changed_E2_phase_keys=changed,new_phase_SHA256=new_phase,old_phase_SHA256=old_phase,
        all_measured_ancestors_bitwise_frozen=True,prefix_SHA256=lineage,response_map_not_deployed=True))
    actual={**c,'phase_lr':.0002,'electronic_lr':.001,'alpha_lr':.001,'head_lr':.002}
    for stage,target,recipe in [('2',300,p['observed_recipe']),('3',340,p['final_recipe'])]:
        if stage=='3':
            assert can_start_G2(result,300)
            assert sum(v.numel() for n,v in model.named_parameters() if n in m.allowed_keys(model,5))==5150
            if seconds_left(p)<2700:
                m.save(a.out/'report.json',dict(state='actual_E2_passed_deadline_boundary',actual_E2_correct=result['selected_correct'],G2_not_started=True,SDK_released=not m.LOCK.exists(),full_physical=False));return
        out=a.out/('s_'+stage);out.mkdir();hardware=base.ROOT/c['hardware_by_stage'][stage]['path']
        assert m.sha(hardware)==c['hardware_by_stage'][stage]['SHA256'] and shutil.disk_usage(base.ROOT).free>=120*1024**3
        fields=capture.acquire(model,data,dm,fields,paths,hardware,stage,out/'capture',pt,interstage_fit=True);paths[stage]=out/'capture'
        assert m.read(paths[stage]/'independent_audit.json')['passed']
        if stage=='2':assert m.read(paths[stage]/'report.json')['phase_SHA256']==new_phase
        model,pt,result=base.fit_suffix(model,data,dm,fields,paths,pt,reference,actual,recipe,out/'fit',target)
        m.save(out/'endpoint_report.json',dict(actual_correct=result['selected_correct'],strict_original_CPU=result['strict_original_CPU'],full_physical=stage=='3',PT_SHA256=m.sha(pt)))
        if stage=='2' and not can_start_G2(result,300):
            m.save(a.out/'report.json',dict(state='actual_E2_gate_unmet',actual_E2_correct=result['selected_correct'],G2_not_started=True,strict_original_CPU=result['strict_original_CPU'],SDK_released=not m.LOCK.exists(),full_physical=False));return
    m.save(a.out/'report.json',dict(state='complete',full_physical=True,full_physical_correct=result['selected_correct'],target_met=result['selected_correct']>=340,
        strict_original_CPU=True,TEST_development=True,SDK_released=not m.LOCK.exists(),PT_SHA256=m.sha(pt),new_inference_layers=0,response_map_not_deployed=True,prefix_SHA256=base.identities(m,paths)))


if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('--profile',type=Path,required=True);q.add_argument('--out',type=Path,required=True);a=q.parse_args()
    try:run(a)
    except BaseException as e:
        if a.out.exists():
            _,_,m,_=base.numerical_modules();m.save(a.out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False,all_data_preserved=True))
        raise
