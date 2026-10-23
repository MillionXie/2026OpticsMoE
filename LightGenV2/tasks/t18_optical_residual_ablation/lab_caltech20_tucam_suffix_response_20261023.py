"""Finite, audited response to underfitting after an existing measured prefix.

Reuses the published original suffix fitter. No inference algorithm is added,
no active controller is patched, and no captured-input ancestor is unfrozen.
"""
import argparse,traceback
from pathlib import Path
import lab_caltech20_tucam_layerwise_20261023 as base
import lab_caltech20_tucam_gap10_20261023 as gap


def validate_policy(p):
    assert p['schema']=='TUCam_L4_G1_underfit_response_v1'
    assert p['parent_stage']=='1' and list(p['captures'])==['router','0','1']
    assert p['intermediate_target']==305 and p['final_target']==340
    assert p['minimum_parent_correct']==250 and p['maximum_parent_correct']==304
    assert p['maximum_parent_TRAIN_accuracy']==.80
    assert p['learning_rates']==dict(phase_lr=.0002,electronic_lr=.001,alpha_lr=.001,head_lr=.002)
    assert p['recipe']==dict(name='G1_observed_underfit_original_suffix',epochs=48,seconds=1500,lr_scale=1,TRAIN_class_emphasis=.75)
    assert p['TEST_gradient'] is False and p['new_inference_layers']==0
    assert p['new_CCD_before_gate']==0 and p['scheduler'] is False
    for key in ('base_profile_SHA256','suffix_engine_SHA256','continuation_engine_SHA256'):
        assert len(p[key])==64


def admit(p,c,out):
    np,_,m,loop=base.numerical_modules();old=base.ROOT/p['completed_controller']
    assert m.read(old.with_suffix('.return.json'))['returncode']==0 and not m.LOCK.exists()
    r=m.read(old/'report.json')
    assert r['state']=='stage_target_not_met_after_diagnostic_fit'
    assert r['next_layer_not_started'] and r['SDK_released']
    endpoints=r['endpoints'];assert [e['stage'] for e in endpoints]==list(p['captures'])
    assert all(e['target_met'] for e in endpoints[:-1]) and not endpoints[-1]['target_met']
    data,dm=m.data_for(base.ROOT/c['sequence']);assert dm['sha256']==c['data_SHA256']
    fields={};paths={}
    for stage,name in p['captures'].items():
        cap=base.ROOT/name;audit=m.read(cap/'independent_audit.json');q=m.read(cap/'report.json');protocol=m.read(cap/'protocol.json')
        assert audit['passed'] and audit['matched_samples']==2011
        assert audit['report_SHA256']==m.sha(cap/'report.json')
        assert q['physical_ccd']==2011 and q['data_SHA256']==dm['sha256']
        assert q['PT_SHA256']==protocol['PT_SHA256']
        if 'prefix_SHA256' in protocol:assert protocol['prefix_SHA256']==base.identities(m,paths)
        fields[stage]={}
        for role,n in (('train',1600),('test',411)):
            array=cap/role/'ccd.npy';assert m.sha(array)==q['arrays'][role]['SHA256']
            values=np.load(array,mmap_mode='r');assert values.shape==(n,478,478) and values.dtype==np.uint8
            fields[stage][role]=values
        paths[stage]=cap
    e=endpoints[-1];parent=Path(e['PT']);assert m.sha(parent)==e['PT_SHA256']
    assert base.identities(m,paths)==e['prefix_SHA256']
    fit=m.read(parent.parent/'report.json');audit=m.read(parent.parent/'independent_audit.json')
    assert fit['state']=='complete' and fit['strict_original_CPU'] and audit['passed']
    assert audit['report_SHA256']==m.sha(parent.parent/'report.json')
    assert fit['results']['best']['PT_SHA256']==e['PT_SHA256']
    assert p['minimum_parent_correct']<=e['selected_correct']<=p['maximum_parent_correct']
    assert fit['results']['best']['train']['accuracy']<p['maximum_parent_TRAIN_accuracy']
    reference=base.ROOT/c['reference'];assert m.sha(reference)==c['reference_SHA256']
    model,_=m.model_for(parent,e['PT_SHA256'],4,reference)
    # Compare to the EXACT PT that produced G1, not a timestamp-selected model.
    cap_protocol=m.read(paths['1']/'protocol.json');captured_parent=base.ROOT/p['capture_parent_PT']
    assert m.sha(captured_parent)==cap_protocol['PT_SHA256']==p['capture_parent_SHA256']
    origin,_=m.model_for(captured_parent,cap_protocol['PT_SHA256'],4,reference)
    loop.validate_state(m,model,m.state_cpu(origin),m.allowed_keys(model,len(fields)))
    score,_=base.evaluate(model,data,fields,'test')
    assert score['correct']==e['selected_correct']==fit['selected_correct']
    m.save(out/'admission.json',dict(parent_PT=str(parent),parent_SHA256=e['PT_SHA256'],parent_report_SHA256=m.sha(parent.parent/'report.json'),
        parent_correct=score['correct'],parent_TRAIN=fit['results']['best']['train']['correct'],data_SHA256=dm['sha256'],
        prefix_SHA256=base.identities(m,paths),captured_masks_and_input_ancestors_frozen=True,new_CCD=0,SDK_used=False,TEST_gradient=False))
    return model,data,dm,fields,paths,parent,reference,list(endpoints)


def endpoint(m,stage,parent,fit,paths):
    target=340 if stage=='3' else 305
    return dict(stage=stage,PT=str(parent),PT_SHA256=m.sha(parent),selected_correct=fit['selected_correct'],target=target,
        target_met=fit['selected_correct']>=target,prefix_SHA256=base.identities(m,paths),strict_original_CPU=True,full_physical=stage=='3',TEST_development=True)


def run(a):
    _,torch,m,_=base.numerical_modules();p=m.read(a.profile);validate_policy(p)
    assert m.sha(base.HERE/p['base_profile'])==p['base_profile_SHA256']
    assert m.sha(base.__file__)==p['suffix_engine_SHA256'] and m.sha(gap.__file__)==p['continuation_engine_SHA256']
    c=m.read(base.HERE/p['base_profile']);base.validate_config(c);c['intermediate_target']=305
    assert not a.out.exists() and not m.LOCK.exists();a.out.mkdir();torch.set_num_threads(2);torch.manual_seed(c['seed'])
    m.save(a.out/'protocol.json',dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),profile_SHA256=m.sha(a.profile),policy=p,
        scheduler=False,automatic_retry=False,original_model=True,new_inference_layers=0,TEST_gradient=False))
    model,data,dm,fields,paths,parent,reference,endpoints=admit(p,c,a.out)
    stage=p['parent_stage'];folder=a.out/('s_'+stage);folder.mkdir()
    m.save(a.out/'progress.json',dict(state='observed_underfit_suffix_fit',stage=stage,parent_PT_SHA256=m.sha(parent)))
    model,parent,fit=base.fit_suffix(model,data,dm,fields,paths,parent,reference,{**c,**p['learning_rates']},p['recipe'],folder/'fit',305)
    e=endpoint(m,stage,parent,fit,paths);endpoints[-1]=e;m.save(folder/'endpoint_report.json',e)
    if not e['target_met']:
        m.save(a.out/'report.json',dict(state='G1_underfit_budget_complete_gate_unmet',endpoints=endpoints,next_layer_not_started=True,SDK_released=not m.LOCK.exists()));return
    from lab_caltech20_tucam_capture_20261023 import acquire
    for stage in base.STAGES[len(fields):]:
        folder=a.out/('s_'+stage);folder.mkdir();target=340 if stage=='3' else 305
        m.save(a.out/'progress.json',dict(state='capturing',stage=stage,parent_PT_SHA256=m.sha(parent),completed_stages=len(fields)))
        fields=acquire(model,data,dm,fields,paths,base.ROOT/c['hardware_by_stage'][stage]['path'],stage,folder/'capture',parent,interstage_fit=True);paths[stage]=folder/'capture'
        for i,recipe in enumerate(c['recipes']):
            model,parent,fit=base.fit_suffix(model,data,dm,fields,paths,parent,reference,c,recipe,folder/('f'+str(i)),target)
            if fit['selected_correct']>=target:break
        if fit['selected_correct']<target:
            updated,recipe,observed_gap=gap.diagnostic_recipe(fit,c)
            m.save(folder/'error_diagnosis.json',dict(TRAIN_TEST_gap=observed_gap,recipe=recipe,TEST_gradient=False))
            model,parent,fit=base.fit_suffix(model,data,dm,fields,paths,parent,reference,updated,recipe,folder/'extra',target)
        e=endpoint(m,stage,parent,fit,paths);endpoints.append(e);m.save(folder/'endpoint_report.json',e)
        if not e['target_met']:
            m.save(a.out/'report.json',dict(state='stage_target_not_met_after_diagnostic_fit',endpoints=endpoints,next_layer_not_started=True,SDK_released=not m.LOCK.exists()));return
    m.save(a.out/'report.json',dict(state='complete',endpoints=endpoints,full_physical=True,strict_original_CPU=True,TEST_development=True,SDK_released=not m.LOCK.exists()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--profile',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:
        run(a);_,_,m,_=base.numerical_modules();m.save(a.out.with_suffix('.return.json'),dict(returncode=0))
    except BaseException as error:
        _,_,m,_=base.numerical_modules();m.save(a.out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False,all_data_preserved=True));m.save(a.out.with_suffix('.return.json'),dict(returncode=1,error=repr(error)));raise
