"""Finite E1 underfit response and gated continuation of the same measured run.

Only orchestration/configuration changes. Original suffix training, original
inference and exact captured-prefix audits are reused without runtime patches.
"""
import argparse,json,traceback
from pathlib import Path
import lab_caltech20_tucam_layerwise_20261023 as base
import lab_caltech20_tucam_gap10_20261023 as gap


def validate_policy(p):
    assert p['schema'] in ('TUCam_L4_E1_underfit_finite_v1','TUCam_L4_E1_teacher_finite_v1')
    assert p['intermediate_target']==305 and p['final_target']==340
    assert p['parent_stage']=='0'
    assert 1<=p['recipe']['epochs']<=48 and 0<p['recipe']['seconds']<=1500
    assert p['recipe']['lr_scale']==1 and p['recipe']['TRAIN_class_emphasis']==.75
    if p['schema']=='TUCam_L4_E1_underfit_finite_v1':
        assert p['maximum_parent_TRAIN_accuracy']==.75 and p['minimum_parent_correct']==255
        assert p['learning_rates']==dict(phase_lr=.0002,electronic_lr=.001,alpha_lr=.001,head_lr=.002)
    else:
        assert p['maximum_parent_TRAIN_accuracy']==1. and p['minimum_parent_correct']==261
        assert p['minimum_parent_TRAIN_TEST_gap']==.10 and p['TRAIN_teacher']==dict(temperature=2,mixture=.3)
        assert p['learning_rates']==dict(phase_lr=.00005,electronic_lr=.0002,alpha_lr=.0002,head_lr=.0005)
        assert p['weight_decay']==.01 and len(p['teacher_engine_SHA256'])==64
    assert p['maximum_parent_correct']==304
    assert p['TEST_gradient'] is False and p['new_inference_layers']==0


def admit(policy,c,out):
    np,torch,m,loop=base.numerical_modules()
    old=base.ROOT/policy['completed_controller'];captures=base.ROOT/policy['capture_root']
    assert m.read(old.with_suffix('.return.json'))['returncode']==0 and not m.LOCK.exists()
    r=m.read(old/'report.json');assert r['state'] in ('stage_target_not_met_after_diagnostic_fit','E1_underfit_budget_complete_gate_unmet')
    assert r['next_layer_not_started'] and r['SDK_released']
    assert [e['stage'] for e in r['endpoints']]==['router','0']
    data,dm=m.data_for(base.ROOT/c['sequence']);assert dm['sha256']==c['data_SHA256']
    fields={};paths={}
    for e in r['endpoints']:
        cap=captures/('s_'+e['stage'])/'capture';audit=m.read(cap/'independent_audit.json');q=m.read(cap/'report.json')
        assert audit['passed'] and audit['matched_samples']==2011 and audit['report_SHA256']==m.sha(cap/'report.json')
        assert q['physical_ccd']==2011 and q['data_SHA256']==dm['sha256']
        assert q['PT_SHA256']==m.read(cap/'protocol.json')['PT_SHA256']
        fields[e['stage']]={};paths[e['stage']]=cap
        for role,n in (('train',1600),('test',411)):
            p=cap/role/'ccd.npy';assert m.sha(p)==q['arrays'][role]['SHA256']
            a=np.load(p,mmap_mode='r');assert a.shape==(n,478,478) and a.dtype==np.uint8;fields[e['stage']][role]=a
    e=r['endpoints'][-1];parent=Path(e['PT']);assert m.sha(parent)==e['PT_SHA256']
    assert base.identities(m,paths)==e['prefix_SHA256']
    fit=m.read(parent.parent/'report.json');audit=m.read(parent.parent/'independent_audit.json')
    assert fit['state']=='complete' and fit['strict_original_CPU'] and audit['passed']
    assert audit['report_SHA256']==m.sha(parent.parent/'report.json')
    assert fit['results']['best']['PT_SHA256']==e['PT_SHA256']
    assert policy['minimum_parent_correct']<=e['selected_correct']<=policy['maximum_parent_correct']
    assert fit['results']['best']['train']['accuracy']<policy['maximum_parent_TRAIN_accuracy']
    if 'minimum_parent_TRAIN_TEST_gap' in policy:
        b=fit['results']['best'];assert b['train']['accuracy']-b['test']['accuracy']>=policy['minimum_parent_TRAIN_TEST_gap']
    reference=base.ROOT/c['reference'];assert m.sha(reference)==c['reference_SHA256']
    model,_=m.model_for(parent,e['PT_SHA256'],4,reference)
    score,_=base.evaluate(model,data,fields,'test');assert score['correct']==e['selected_correct']==fit['selected_correct']
    m.save(out/'admission.json',dict(parent_PT=str(parent),parent_SHA256=e['PT_SHA256'],parent_report_SHA256=m.sha(parent.parent/'report.json'),
        parent_correct=score['correct'],parent_TRAIN=fit['results']['best']['train']['correct'],data_SHA256=dm['sha256'],prefix_SHA256=base.identities(m,paths),
        captured_masks_and_input_ancestors_frozen=True,new_CCD=0,SDK_used=False,TEST_gradient=False))
    return model,data,dm,fields,paths,parent,reference,list(r['endpoints'])


def run(a):
    _,torch,m,_=base.numerical_modules();p=m.read(a.profile);validate_policy(p)
    assert m.sha(base.HERE/p['base_profile'])==p['base_profile_SHA256']
    c=m.read(base.HERE/p['base_profile']);base.validate_config(c);c['intermediate_target']=305
    assert m.sha(base.__file__)==p['suffix_engine_SHA256'] and m.sha(gap.__file__)==p['continuation_engine_SHA256']
    assert not a.out.exists() and not m.LOCK.exists();a.out.mkdir();torch.set_num_threads(2);torch.manual_seed(c['seed'])
    m.save(a.out/'protocol.json',dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),policy=p,profile_SHA256=m.sha(a.profile),
        scheduler=False,automatic_retry=False,original_model=True,new_inference_layers=0,TEST_gradient=False))
    model,data,dm,fields,paths,parent,reference,endpoints=admit(p,c,a.out)
    tuned={**c,**p['learning_rates']};folder=a.out/'s_0';folder.mkdir()
    m.save(a.out/'progress.json',dict(state='E1_underfit_suffix_fit',stage='0',target=305,parent_PT_SHA256=m.sha(parent)))
    if 'TRAIN_teacher' in p:
        import lab_caltech20_tucam_TRAIN_distill_20261023 as teacher
        assert m.sha(teacher.__file__)==p['teacher_engine_SHA256']
        tuned.update(TRAIN_teacher=p['TRAIN_teacher'],weight_decay=p['weight_decay'])
        model,parent,fit=teacher.fit_suffix(model,data,dm,fields,paths,parent,reference,tuned,p['recipe'],folder/'fit',305)
    else:
        model,parent,fit=base.fit_suffix(model,data,dm,fields,paths,parent,reference,tuned,p['recipe'],folder/'fit',305)
    def endpoint(stage,parent,fit):
        target=340 if stage=='3' else 305
        return dict(stage=stage,PT=str(parent),PT_SHA256=m.sha(parent),selected_correct=fit['selected_correct'],target=target,target_met=fit['selected_correct']>=target,
            prefix_SHA256=base.identities(m,paths),strict_original_CPU=True,full_physical=stage=='3',TEST_development=True)
    e=endpoint('0',parent,fit);endpoints[-1]=e;m.save(folder/'endpoint_report.json',e)
    if not e['target_met']:
        state='E1_TRAIN_teacher_budget_complete_gate_unmet' if 'TRAIN_teacher' in p else 'E1_underfit_budget_complete_gate_unmet'
        m.save(a.out/'report.json',dict(state=state,endpoints=endpoints,next_layer_not_started=True,SDK_released=not m.LOCK.exists()));return
    from lab_caltech20_tucam_capture_20261023 import acquire
    for stage in base.STAGES[len(fields):]:
        folder=a.out/('s_'+stage);folder.mkdir();target=340 if stage=='3' else 305
        m.save(a.out/'progress.json',dict(state='capturing',stage=stage,parent_PT_SHA256=m.sha(parent),completed_stages=len(fields)))
        fields=acquire(model,data,dm,fields,paths,base.ROOT/c['hardware_by_stage'][stage]['path'],stage,folder/'capture',parent,interstage_fit=True);paths[stage]=folder/'capture'
        for i,recipe in enumerate(c['recipes']):
            model,parent,fit=base.fit_suffix(model,data,dm,fields,paths,parent,reference,c,recipe,folder/('f'+str(i)),target)
            if fit['selected_correct']>=target:break
        if fit['selected_correct']<target:
            extra,recipe,observed_gap=gap.diagnostic_recipe(fit,c)
            m.save(folder/'error_diagnosis.json',dict(TRAIN_TEST_gap=observed_gap,recipe=recipe,TEST_gradient=False))
            model,parent,fit=base.fit_suffix(model,data,dm,fields,paths,parent,reference,extra,recipe,folder/'extra',target)
        e=endpoint(stage,parent,fit);endpoints.append(e);m.save(folder/'endpoint_report.json',e)
        if not e['target_met']:
            m.save(a.out/'report.json',dict(state='stage_target_not_met_after_diagnostic_fit',endpoints=endpoints,next_layer_not_started=True,SDK_released=not m.LOCK.exists()));return
    m.save(a.out/'report.json',dict(state='complete',endpoints=endpoints,full_physical=True,strict_original_CPU=True,TEST_development=True,SDK_released=not m.LOCK.exists()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--profile',type=Path,default=base.HERE/'lab_caltech20_tucam_E1_underfit_profile_20261023.json');parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    try:
        run(args);_,_,m,_=base.numerical_modules();m.save(args.out.with_suffix('.return.json'),dict(returncode=0))
    except BaseException as error:
        _,_,m,_=base.numerical_modules();m.save(args.out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False,all_data_preserved=True));m.save(args.out.with_suffix('.return.json'),dict(returncode=1,error=repr(error)));raise
