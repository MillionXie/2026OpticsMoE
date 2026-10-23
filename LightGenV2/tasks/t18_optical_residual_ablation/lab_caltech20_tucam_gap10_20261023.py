"""One-shot continuation at the existing L4 controller's audited boundary.

Never edits or stops the active capture/fit. Reuses its valid measured prefix,
then applies the human's revised10pp gate; every new optical suffix is fresh.
This is a finite experimental dependency workflow, not a periodic automation.
"""
import argparse,copy,json,os,time,traceback
from pathlib import Path
import lab_caltech20_tucam_layerwise_20261023 as base


def config(path):
    _,_,m,_=base.numerical_modules();policy=m.read(path)
    assert policy['schema']=='TUCam_L4_human_gap10_boundary_v1'
    parent=base.HERE/policy['base_profile'];assert m.sha(parent)==policy['base_profile_SHA256']
    c=m.read(parent);base.validate_config(c)
    assert policy['intermediate_target']==base.gate_target(346,10)==305
    assert policy['final_target']==340 and policy['maximum_parent_wait_s']<=28800
    c['intermediate_target']=305;c['policy_SHA256']=m.sha(path)
    return policy,c


def diagnostic_recipe(report,c):
    """One registered response to actual TRAIN/TEST gap, no random scanning."""
    r=report['results']['last'];gap=r['train']['accuracy']-r['test']['accuracy']
    updated=copy.deepcopy(c)
    if gap>.10:
        updated.update(weight_decay=.01,label_smoothing=.06)
        recipe=dict(name='overfit_lowerLR_regularized',epochs=12,seconds=600,lr_scale=.1,TRAIN_class_emphasis=.5)
    else:
        recipe=dict(name='TRAIN_fit_focused_suffix',epochs=24,seconds=900,lr_scale=.5,TRAIN_class_emphasis=.75)
    return updated,recipe,gap


def load_boundary(old,c):
    np,_,m,loop=base.numerical_modules()
    r=m.read(old/'report.json');assert r['state'] in ('complete','stage_target_not_met')
    assert m.read(old.with_suffix('.return.json'))['returncode']==0 and not m.LOCK.exists()
    data,dm=m.data_for(base.ROOT/c['sequence']);assert dm['sha256']==c['data_SHA256']
    fields={};paths={};endpoints=r['endpoints'];assert endpoints
    assert [e['stage'] for e in endpoints]==list(base.STAGES[:len(endpoints)])
    for e in endpoints:
        cap=old/('s_'+e['stage'])/'capture';q=m.read(cap/'report.json');audit=m.read(cap/'independent_audit.json')
        assert audit['passed'] and audit['matched_samples']==2011 and audit['report_SHA256']==m.sha(cap/'report.json')
        assert q['physical_ccd']==2011 and q['data_SHA256']==dm['sha256'] and q['PT_SHA256']==m.read(cap/'protocol.json')['PT_SHA256']
        fields[e['stage']]={}
        for role,n in (('train',1600),('test',411)):
            p=cap/role/'ccd.npy';assert m.sha(p)==q['arrays'][role]['SHA256'];a=np.load(p,mmap_mode='r');assert a.shape==(n,478,478) and a.dtype==np.uint8;fields[e['stage']][role]=a
        paths[e['stage']]=cap
    e=endpoints[-1];parent=Path(e['PT']);assert m.sha(parent)==e['PT_SHA256']
    assert base.identities(m,paths)==e['prefix_SHA256']
    reference=base.ROOT/c['reference'];assert m.sha(reference)==c['reference_SHA256']
    model,_=m.model_for(parent,e['PT_SHA256'],4,reference);score,_=base.evaluate(model,data,fields,'test');assert score['correct']==e['selected_correct']
    # The accepted PT must be backed by the original strict native bestlast audit.
    fit=m.read(parent.parent/'report.json');assert fit['strict_original_CPU'] and fit['results']['best']['PT_SHA256']==e['PT_SHA256']
    assert fit['selected_correct']==score['correct']
    return model,data,dm,fields,paths,parent,reference,endpoints,fit


def run(a):
    _,torch,m,_=base.numerical_modules();policy,c=config(a.profile);old=base.ROOT/policy['predecessor']
    assert not a.out.exists();a.out.mkdir(parents=True);torch.set_num_threads(2);torch.manual_seed(c['seed'])
    m.save(a.out/'protocol.json',dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),policy=policy,config=c,
        previous_controller_unchanged=True,automatic_restarts=False,scheduler=False,TEST_gradient=False))
    started=time.monotonic()
    while not old.with_suffix('.return.json').exists():
        assert time.monotonic()-started<policy['maximum_parent_wait_s'],'Existing finite workflow did not reach a boundary within the registered wait'
        m.save(a.out/'progress.json',dict(state='waiting_existing_controller_boundary',elapsed_s=time.monotonic()-started));time.sleep(15)
    ret=m.read(old.with_suffix('.return.json'))
    if ret['returncode']!=0:
        m.save(a.out/'report.json',dict(state='predecessor_failed_needs_diagnosis',predecessor_return=ret,new_CCD=0,automatic_retry=False));return
    model,data,dm,fields,paths,parent,reference,previous,fit=load_boundary(old,c)
    endpoints=list(previous);stage=previous[-1]['stage'];current=previous[-1]['selected_correct'];target=340 if stage=='3' else 305
    if current<target:
        folder=a.out/('s_'+stage);folder.mkdir()
        tuned,recipe,gap=diagnostic_recipe(fit,c);m.save(folder/'error_diagnosis.json',dict(TRAIN_TEST_gap=gap,source_report=str(parent.parent/'report.json'),recipe=recipe,TRAIN_only_gradient=True))
        model,parent,fit=base.fit_suffix(model,data,dm,fields,paths,parent,reference,tuned,recipe,folder/'extra',target);current=fit['selected_correct']
    e=dict(stage=stage,PT=str(parent),PT_SHA256=m.sha(parent),selected_correct=current,target=target,target_met=current>=target,
        prefix_SHA256=base.identities(m,paths),full_physical=stage=='3',strict_original_CPU=True,TEST_development=True,human_revised_gate=True)
    m.save(a.out/'boundary_admission.json',e);endpoints[-1]=e
    if current<target:
        m.save(a.out/'report.json',dict(state='stage_target_not_met_after_diagnostic_fit',endpoints=endpoints,next_layer_not_started=True,SDK_released=True));return
    from lab_caltech20_tucam_capture_20261023 import acquire
    for stage in base.STAGES[len(fields):]:
        folder=a.out/('s_'+stage);folder.mkdir();target=340 if stage=='3' else 305
        m.save(a.out/'progress.json',dict(state='capturing',stage=stage,completed_stages=len(fields),parent_PT_SHA256=m.sha(parent)))
        fields=acquire(model,data,dm,fields,paths,base.ROOT/c['hardware_by_stage'][stage]['path'],stage,folder/'capture',parent,interstage_fit=True);paths[stage]=folder/'capture'
        for i,recipe in enumerate(c['recipes']):
            m.save(a.out/'progress.json',dict(state='suffix_fitting',stage=stage,recipe=recipe['name'],target=target))
            model,parent,fit=base.fit_suffix(model,data,dm,fields,paths,parent,reference,c,recipe,folder/('f'+str(i)),target)
            if fit['selected_correct']>=target:break
        if fit['selected_correct']<target:
            tuned,recipe,gap=diagnostic_recipe(fit,c);m.save(folder/'error_diagnosis.json',dict(TRAIN_TEST_gap=gap,recipe=recipe,TRAIN_only_gradient=True))
            model,parent,fit=base.fit_suffix(model,data,dm,fields,paths,parent,reference,tuned,recipe,folder/'extra',target)
        e=dict(stage=stage,PT=str(parent),PT_SHA256=m.sha(parent),selected_correct=fit['selected_correct'],target=target,target_met=fit['selected_correct']>=target,
            prefix_SHA256=base.identities(m,paths),full_physical=stage=='3',strict_original_CPU=True,TEST_development=True)
        m.save(folder/'endpoint_report.json',e);endpoints.append(e)
        if not e['target_met']:
            m.save(a.out/'report.json',dict(state='stage_target_not_met_after_diagnostic_fit',endpoints=endpoints,next_layer_not_started=True,SDK_released=True));return
    m.save(a.out/'report.json',dict(state='complete',endpoints=endpoints,full_physical=True,strict_original_CPU=True,TEST_development=True,SDK_released=not m.LOCK.exists()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--profile',type=Path,default=base.HERE/'lab_caltech20_tucam_gap10_profile_20261023.json');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:
        run(a);_,_,m,_=base.numerical_modules();m.save(a.out.with_suffix('.return.json'),dict(returncode=0))
    except BaseException as e:
        _,_,m,_=base.numerical_modules();m.save(a.out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False,all_data_preserved=True));m.save(a.out.with_suffix('.return.json'),dict(returncode=1,error=repr(e)));raise
