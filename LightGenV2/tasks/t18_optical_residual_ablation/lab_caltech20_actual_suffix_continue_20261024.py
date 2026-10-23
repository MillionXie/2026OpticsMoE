"""Finite continuation on one immutable, newly measured E2; no old CCD mixing.

Compose the existing native suffix / OAS / affine engines, not a new model.
Every continuation is separately configured and reviewed before execution.
"""
import argparse,shutil,traceback
from pathlib import Path
import lab_caltech20_E2_actual300_20261024 as admission
base=admission.base


def validate_policy(p):
    assert p['schema']=='TUCam_L4_actual300_suffix_continuation_v1'
    assert p['actual_E2_target']==300 and p['final_target']==340
    assert p['TRAIN']==1600 and p['TEST']==411 and p['VALID']==0
    assert not p['TEST_gradient'] and not p['scheduler'] and not p['automatic_retry']
    assert p['new_inference_layers']==0 and p['method'] in ('OAS','affine_LBFGS','native_CE')
    assert 0<=p['expected_parent_correct']<300
    assert p['source_fit'].startswith(p['capture_run']+'/')
    assert p['capture_run'].endswith('/e2_actual300_20261024')
    assert p['capture_PT_SHA256']=='99c5538aeea0436f96df9c3ae4263b9693ab39b0c75d57d85630365a3d386e41'
    assert p['E2_phase_SHA256']=='2e469eb91c507b777348d157346b14cdb657c2d09138385bedae2a6985458151'
    assert len(p['source_PT_SHA256'])==len(p['source_report_SHA256'])==64
    assert p['deadline']=='2026-10-24T08:00:00+08:00'
    assert 1<=p['recipe']['epochs']<=80 and 0<p['recipe']['seconds']<=1800


def admit(p,c,out):
    np,torch,m,loop=base.numerical_modules()
    run=base.ROOT/p['capture_run'];original_policy=m.read(run/'protocol.json')['policy']
    admission.validate_policy(original_policy);assert m.read(run/'candidate_admission.json')['passed']
    source=base.ROOT/p['source_fit'];sr=m.read(source/'report.json');sa=m.read(source/'independent_audit.json');pt=source/'best.pt'
    assert m.sha(source/'report.json')==sa['report_SHA256']==p['source_report_SHA256'] and sa['passed'] and sr['strict_original_CPU']
    assert m.sha(pt)==sr['results']['best']['PT_SHA256']==p['source_PT_SHA256'] and sr['selected_correct']==p['expected_parent_correct']
    assert m.read(base.ROOT/p['completed_controller'])['returncode']==0
    data,dm=m.data_for(base.ROOT/c['sequence']);assert dm['sha256']==c['data_SHA256'];fields={};paths={}
    for stage,relative in p['captures'].items():
        assert stage==base.STAGES[len(fields)];cap=base.ROOT/relative;r=m.read(cap/'report.json');audit=m.read(cap/'independent_audit.json');cp=m.read(cap/'protocol.json')
        assert r['state']=='complete' and r['physical_ccd']==2011 and r['SDK_released'] and r['data_SHA256']==dm['sha256']
        assert audit['passed'] and audit['matched_samples']==2011 and audit['report_SHA256']==m.sha(cap/'report.json')
        assert audit['protocol_SHA256']==r['protocol_SHA256']==m.sha(cap/'protocol.json')
        assert r['hardware_SHA256']==c['hardware_by_stage'][stage]['SHA256'] and r['phase_SHA256']==m.sha(cap/'phase.bmp')
        assert cp['prefix_SHA256']==base.identities(m,paths)
        if stage=='2':assert r['PT_SHA256']==p['capture_PT_SHA256'] and r['phase_SHA256']==p['E2_phase_SHA256'] and cap==run/'s_2/capture'
        fields[stage]={}
        for role,n in (('train',1600),('test',411)):
            f=cap/role/'ccd.npy';assert m.sha(f)==r['arrays'][role]['SHA256'];values=np.load(f,mmap_mode='r');assert values.shape==(n,478,478) and values.dtype==np.uint8;fields[stage][role]=values
        paths[stage]=cap
    assert list(fields)==['router','0','1','2'] and m.read(source/'protocol.json')['prefix_SHA256']==base.identities(m,paths)
    reference=base.ROOT/c['reference'];assert m.sha(reference)==c['reference_SHA256'];model,_=m.model_for(pt,p['source_PT_SHA256'],4,reference)
    initial=base.ROOT/original_policy['candidate_fit']/'best.pt';capture_model,_=m.model_for(initial,p['capture_PT_SHA256'],4,reference)
    loop.validate_state(m,model,m.state_cpu(capture_model),m.allowed_keys(model,4))
    metric,_=base.evaluate(model,data,fields,'test');assert metric['correct']==p['expected_parent_correct']
    m.save(out/'admission.json',dict(passed=True,source_correct=metric['correct'],prefix_SHA256=base.identities(m,paths),all_captured_input_ancestors_bitwise_frozen=True,
        TRAIN=1600,TEST=411,VALID=0,new_CCD=0,SDK_used=False,full_physical=False,actual_E2_with_simulated_G2=True))
    return model,data,dm,fields,paths,pt,reference


def run(a):
    np,torch,m,_=base.numerical_modules();p=m.read(a.profile);validate_policy(p)
    for name,digest in p['source_SHA256'].items():assert m.sha(base.HERE/name)==digest
    bp=base.HERE/p['base_profile'];assert m.sha(bp)==p['base_profile_SHA256'];c=m.read(bp);base.validate_config(c);c['intermediate_target']=300
    assert admission.seconds_left(p)>1800 and not a.out.exists() and not m.LOCK.exists();a.out.mkdir();torch.set_num_threads(2)
    m.save(a.out/'protocol.json',dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),profile_SHA256=m.sha(a.profile),policy=p,TEST_development=True,TEST_gradient=False,new_inference_layers=0,scheduler=False))
    model,data,dm,fields,paths,pt,reference=admit(p,c,a.out)
    if p['method']=='native_CE':
        config={**c,**p['learning_rates']};engine=base
    else:
        import lab_caltech20_tucam_TRAIN_linear_20261023 as engine
        config={**c}
        if p['method']=='affine_LBFGS':
            config.update(TRAIN_readout='cached_TRAIN_affine_LBFGS',native_feature_source=p['source_fit'],fresh_native_TRAIN_features=True,
                TRAIN_affine_optimizer=dict(l2_parent_TRAIN_NLL_fraction=.1,label_smoothing=.03,outer_steps=80))
    model,pt,result=engine.fit_suffix(model,data,dm,fields,paths,pt,reference,config,p['recipe'],a.out/'fit',300)
    if not admission.can_start_G2(result,300):
        m.save(a.out/'report.json',dict(state='actual_E2_gate_unmet',actual_E2_correct=result['selected_correct'],strict_original_CPU=True,SDK_released=not m.LOCK.exists(),G2_not_started=True,full_physical=False));return
    if admission.seconds_left(p)<2700:
        m.save(a.out/'report.json',dict(state='actual_E2_passed_deadline_boundary',actual_E2_correct=result['selected_correct'],strict_original_CPU=True,SDK_released=not m.LOCK.exists(),G2_not_started=True,full_physical=False));return
    assert admission.can_start_G2(result,300)
    import lab_caltech20_tucam_capture_20261023 as capture
    out=a.out/'s_3';out.mkdir();hardware=base.ROOT/c['hardware_by_stage']['3']['path']
    assert m.sha(hardware)==c['hardware_by_stage']['3']['SHA256'] and shutil.disk_usage(base.ROOT).free>=120*1024**3
    assert sum(v.numel() for n,v in model.named_parameters() if n in m.allowed_keys(model,5))==5150
    fields=capture.acquire(model,data,dm,fields,paths,hardware,'3',out/'capture',pt,interstage_fit=True);paths['3']=out/'capture'
    recipe=dict(name='actual300_full_physical_original5150',epochs=120,seconds=1200,lr_scale=1,TRAIN_class_emphasis=.75)
    model,pt,result=base.fit_suffix(model,data,dm,fields,paths,pt,reference,c,recipe,out/'fit',340)
    m.save(a.out/'report.json',dict(state='complete',full_physical=True,full_physical_correct=result['selected_correct'],target_met=result['selected_correct']>=340,
        strict_original_CPU=True,TEST_development=True,SDK_released=not m.LOCK.exists(),PT_SHA256=m.sha(pt),new_inference_layers=0,prefix_SHA256=base.identities(m,paths)))


if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('--profile',type=Path,required=True);q.add_argument('--out',type=Path,required=True);a=q.parse_args()
    try:run(a)
    except BaseException as e:
        if a.out.exists():
            _,_,m,_=base.numerical_modules();m.save(a.out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False,all_data_preserved=True))
        raise
