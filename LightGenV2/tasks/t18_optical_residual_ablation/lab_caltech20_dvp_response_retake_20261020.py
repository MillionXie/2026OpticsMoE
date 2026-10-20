"""Prepare a fresh failed-layer retake from an independently passed parent.

Artifact namespaces only: no engineering copy, changed physical graph or lower
gate. Old failed-layer CCDs remain in their original namespace, never linked in.
"""
import argparse
from pathlib import Path

ATTEMPT='response_20261020'


def TRAIN_capacity(a):
    """One fixed published readout estimator, TRAIN only, no new checkpoint."""
    import torch
    import numpy as np
    import lab_caltech20_dvp_layerwise_20261019 as m
    from lab_caltech20_residual_terminal_cpu_20261019 import shrunk_readout
    torch.set_num_threads(2);session=a.sequence/f'L{a.depth}_strict8pp_20261020';prefix={4:3,6:1}[a.depth]
    mask=session/'fits'/(('1' if a.depth==4 else 'router')+'_spatial_response_retrain_20261020')
    parent=session/'fits'/({4:'1_binary_repaired_20261020',6:'router_binary_repaired_20261020'}[a.depth])
    target=mask/'TRAIN_capacity.json';assert not target.exists()
    r,p,proof=[m.read(mask/n) for n in ('report.json','protocol.json','independent_audit.json')]
    assert r['state']=='complete' and proof['passed'] and proof['report_SHA256']==m.sha(mask/'report.json')
    data,dm=m.data_for(a.sequence);fields,identities=m.fields_for(session,prefix,data,dm,m.session_hardware_digest(a.sequence))
    assert p['prefix_SHA256']==identities and p['data_SHA256']==dm['sha256'] and p['parent_PT_SHA256']==m.sha(parent/'best.pt')
    cfg=m.read(Path(__file__).with_name('lab_caltech20_DVP_shrinkage_head_profile_20261020.json'))
    assert cfg['original_TRAIN_cache_SHA256']==dm['source_cache_SHA256'];counts=torch.tensor(cfg['original_TRAIN_counts'],dtype=torch.float64)
    assert int(counts.sum())==3141;labels=torch.from_numpy(data['train'][1].astype(np.int64));rows={}
    for role,path in (('parent',parent/'best.pt'),('mask_last',mask/'last.pt')):
        model,_=m.model_for(path,m.sha(path),a.depth,session/'source_reference.ts');state=m.state_cpu(model);parts=[]
        hook=model.linear.register_forward_pre_hook(lambda module,args:parts.append(args[0].detach().numpy().copy()))
        try:actual,_=m.assess(model,data,fields,'train')
        finally:hook.remove()
        features=torch.tensor(np.concatenate(parts),dtype=torch.float64);assert features.shape==(1600,256)
        w,b,estimator=shrunk_readout(features,labels,counts);logits=features@w.T+b
        rows[role]=dict(PT_SHA256=m.sha(path),TRAIN_original_correct=actual['correct'],TRAIN_original_NLL=actual['nll'],
            TRAIN_capacity_correct=int((logits.argmax(1)==labels).sum()),TRAIN_capacity_NLL=float(torch.nn.functional.cross_entropy(logits,labels)),estimator=estimator)
        assert all(torch.equal(v,m.state_cpu(model)[k]) for k,v in state.items())
    old,new=rows['parent'],rows['mask_last']
    improved=(new['TRAIN_capacity_correct']>old['TRAIN_capacity_correct'] and new['TRAIN_capacity_NLL']<old['TRAIN_capacity_NLL'])
    q=dict(passed=True,capacity_improved=improved,rows=rows,TRAIN=1600,TEST_inference=False,VALID=0,new_PT_written=False,
        source_checkpoint_role='last',source_PT_SHA256=m.sha(mask/'last.pt'),parent_PT_SHA256=m.sha(parent/'best.pt'),
        source_report_SHA256=m.sha(mask/'report.json'),source_audit_SHA256=m.sha(mask/'independent_audit.json'),
        data_SHA256=dm['sha256'],prefix_SHA256=identities,source_git=m.git_head(),source_SHA256=m.sha(__file__))
    m.save(target,q);print(m.json.dumps({**q,'rows':{k:{n:v for n,v in row.items() if n!='estimator'} for k,row in rows.items()}}))


def assemble_head_continuation(a):
    """Audit a real sequential mask-LAST -> original-head workflow, not a merge.

    The original passed prefix is the compound workflow's start. Both actual
    intermediate PT roles remain explicit; no LAST is represented as a parent
    BEST, and no old failed physical CCD enters this pre-retake fit.
    """
    import torch
    import numpy as np
    import lab_caltech20_dvp_layerwise_20261019 as m
    from lab_caltech20_dvp_strict_retake_20261020 import required_correct
    torch.set_num_threads(2);old=a.sequence/f'L{a.depth}_strict8pp_20261020';prefix={4:3,6:1}[a.depth]
    rootname='1' if a.depth==4 else 'router'
    mask=old/'fits'/(rootname+'_spatial_response_retrain_20261020')
    head=old/'fits'/(rootname+'_spatial_last_readout_20261020')
    parent=old/'fits'/({4:'1_binary_repaired_20261020',6:'router_binary_repaired_20261020'}[a.depth])
    mr,mp,ma=[m.read(mask/n) for n in ('report.json','protocol.json','independent_audit.json')]
    hr,hp,ha=[m.read(head/n) for n in ('report.json','protocol.json','independent_audit.json')]
    assert ma['passed'] and ha['passed'] and mr['state']==hr['state']=='complete'
    assert ma['report_SHA256']==m.sha(mask/'report.json') and ha['report_SHA256']==m.sha(head/'report.json')
    assert mp['original_readout_frozen'] and hp['parent_PT_SHA256']==m.sha(mask/'last.pt')==mr['results']['last']['PT_SHA256']
    evidence=m.read(mask/'TRAIN_capacity.json')
    assert evidence['passed'] and evidence['capacity_improved'] and not evidence['TEST_inference']
    assert evidence['source_checkpoint_role']=='last' and evidence['source_PT_SHA256']==hp['parent_PT_SHA256']
    assert hp['config']['TRAIN_capacity_evidence_SHA256']==m.sha(mask/'TRAIN_capacity.json')
    assert mp['parent_PT_SHA256']==m.sha(parent/'best.pt') and hp['prefix']==mp['prefix']==prefix
    assert hr['selected_correct']>=required_correct(a.depth,prefix) and not hr['selected_e0']
    data,dm=m.data_for(a.sequence);fields,identities=m.fields_for(old,prefix,data,dm,m.session_hardware_digest(a.sequence))
    assert mp['prefix_SHA256']==hp['prefix_SHA256']==identities and mp['data_SHA256']==hp['data_SHA256']==dm['sha256']
    origin=torch.load(parent/'best.pt',map_location='cpu',weights_only=False)['model']
    intermediate=torch.load(mask/'last.pt',map_location='cpu',weights_only=False)['model']
    states={role:torch.load(head/(role+'.pt'),map_location='cpu',weights_only=False)['model'] for role in ('best','last')}
    model,_=m.model_for(head/'best.pt',hr['results']['best']['PT_SHA256'],a.depth,old/'source_reference.ts');allowed=m.allowed_keys(model,prefix)
    assert set(mp['allowed_keys'])==allowed-{'linear.weight','linear.bias'}
    assert set(hp['allowed_keys'])=={'linear.weight','linear.bias'}
    for state in states.values():
        assert all(torch.equal(v,state[k]) for k,v in intermediate.items() if not k.startswith('linear.'))
        assert all(torch.equal(v,state[k]) for k,v in origin.items() if k not in allowed)
    out=old/'fits'/(rootname+'_spatial_response_compound_20261020');assert not out.exists();out.mkdir()
    p=dict(parent_PT_SHA256=m.sha(parent/'best.pt'),prefix=prefix,depth=a.depth,prefix_SHA256=identities,data_SHA256=dm['sha256'],
        allowed_keys=sorted(allowed),TRAIN=1600,TEST=411,VALID=0,TEST_gradient=False,TEST_informed_development=True,
        source_git=m.git_head(),source_SHA256=m.sha(__file__),compound_sequential_workflow=True,assembly_optimizer_steps=0,
        mask_fit=dict(path=str(mask),initial_PT_SHA256=mp['parent_PT_SHA256'],actual_output_role='last',actual_output_PT_SHA256=m.sha(mask/'last.pt'),
                      report_SHA256=m.sha(mask/'report.json'),audit_SHA256=m.sha(mask/'independent_audit.json')),
        head_fit=dict(path=str(head),actual_initialization_role='mask_last',actual_initialization_PT_SHA256=hp['parent_PT_SHA256'],
                      selected_role='best',report_SHA256=m.sha(head/'report.json'),audit_SHA256=m.sha(head/'independent_audit.json')),
        no_new_inference_layer=True,measured_upstream_frozen=True,old_failed_downstream_CCD_used=False)
    m.save(out/'protocol.json',p);results={}
    for role,state in states.items():
        path=out/(role+'.pt');torch.save(dict(model=state,protocol=p,actual_sequence='original passed prefix -> mask full-budget LAST -> head '+role),path)
        native,_=m.model_for(path,m.sha(path),a.depth,old/'source_reference.ts');score,prob=m.assess(native,data,fields,'test')
        assert score['correct']==hr['results'][role]['TEST_development']['correct']
        np.savez_compressed(out/(role+'_TEST_predictions.npz'),ids=data['test'][2],labels=data['test'][1],probabilities=prob)
        results[role]=dict(TEST_development=score,strict_CPU=True,PT_SHA256=m.sha(path),predictions_SHA256=m.sha(out/(role+'_TEST_predictions.npz')))
    r=dict(state='complete',results=results,selected_correct=results['best']['TEST_development']['correct'],selected_e0=False,
        within8pp=True,frozen_parameters_unchanged=True,intermediate=True,compound_sequential_workflow=True,
        no_new_optimizer_steps=True,TEST_gradients=False,SDK_used=False,GPU_used=False)
    m.save(out/'report.json',r)
    m.save(out/'independent_audit.json',dict(passed=True,selected_correct=r['selected_correct'],report_SHA256=m.sha(out/'report.json'),
        protocol_SHA256=m.sha(out/'protocol.json'),original_structure_CPU=True,all_measured_upstream_frozen=True,
        true_mask_LAST_initialization_disclosed=True,head_nonreadout_exactly_mask_LAST=True,TRAIN=1600,TEST=411,VALID=0,
        source_git=m.git_head(),source_SHA256=m.sha(__file__)))
    print(m.json.dumps(dict(assembled=True,correct=r['selected_correct'],PT_SHA256=results['best']['PT_SHA256'],actual_mask_role='last',new_optimizer_steps=0)))


def prepare(a):
    import torch
    import lab_caltech20_dvp_layerwise_20261019 as m
    from lab_caltech20_dvp_strict_retake_20261020 import check_score,link_file
    torch.set_num_threads(2)
    assert not m.LOCK.exists()
    old=a.sequence/f'L{a.depth}_strict8pp_20261020'
    prefix={4:3,6:1}[a.depth];stage={4:'2',6:'0'}[a.depth]
    name=('1' if a.depth==4 else 'router')+('_spatial_response_compound_20261020' if a.parent_mode=='compound' else '_spatial_response_retrain_20261020')
    prior=old/'fits'/name;r=m.read(prior/'report.json');p=m.read(prior/'protocol.json')
    audit=m.read(prior/'independent_audit.json')
    digest=check_score(r,a.depth,prefix)
    assert not r['selected_e0'],'No changed response-trained candidate: no identical-PT reshoot'
    assert audit['passed'] and audit['report_SHA256']==m.sha(prior/'report.json')
    assert audit['protocol_SHA256']==m.sha(prior/'protocol.json') and audit['selected_correct']==r['selected_correct']
    data,dm=m.data_for(a.sequence);hardware=m.session_hardware_digest(a.sequence)
    fields,identities=m.fields_for(old,prefix,data,dm,hardware)
    assert p['prefix']==prefix and p['prefix_SHA256']==identities and p['data_SHA256']==dm['sha256']
    previous=(['router']+[str(i) for i in range(a.depth)])[prefix-1]
    chain=m.validate_fit_ancestry(old,name,digest,prefix,m.read(old/'capture'/previous/'report.json')['PT_SHA256'],identities,dm['sha256'])
    session=m.session_for(a);assert not session.exists();session.mkdir();links=[]
    for physical in ['router']+[str(i) for i in range(prefix-1)]:
        source=old/'capture'/physical
        assert m.read(source/'independent_capture_audit.json')['passed']
        for file in source.rglob('*'):
            if file.is_file():link_file(file,session/'capture'/physical/file.relative_to(source),links)
    for node in chain:
        source=old/'fits'/node['fit']
        assert m.read(source/'independent_audit.json')['passed']
        for file in source.rglob('*'):
            if file.is_file():link_file(file,session/'fits'/node['fit']/file.relative_to(source),links)
    for name in ('initial.pt','source_reference.ts','lab_gate.json'):
        link_file(old/name,session/name,links)
    _,newidentities=m.fields_for(session,prefix,data,dm,hardware)
    assert identities==newidentities and not (session/'capture'/stage).exists()
    m.save(session/'reuse_manifest.json',dict(source_session=str(old),links=links,passed_prefix=prefix,
        preserved_old_failed_stage=str(old/'capture'/stage),failed_stage_CCD_reused=False,
        data_SHA256=dm['sha256'],hardware_SHA256=hardware,prefix_SHA256=identities,
        parent_fit=prior.name,parent_PT_SHA256=digest,parent_audit_SHA256=m.sha(prior/'independent_audit.json'),
        source_git=m.git_head(),source_SHA256=m.sha(__file__),engineering_copy=False,immutable_artifact_links=True))
    m.save(session/'prepared.json',dict(passed=True,reuse_manifest_SHA256=m.sha(session/'reuse_manifest.json'),
        first_retake=stage,source_git=m.git_head(),no_automatic_next_layer=True))
    print(m.json.dumps(dict(prepared=True,depth=a.depth,first_retake=stage,parent_fit=prior.name,
        parent_PT_SHA256=digest,native_correct=r['selected_correct'],failed_layer_CCD_reused=False,SDK_used=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sequence',type=Path,required=True)
    p.add_argument('--depth',type=int,choices=(4,6),required=True)
    p.add_argument('--parent-mode',choices=('trained_best','compound'),default='trained_best')
    p.add_argument('--assemble-head',action='store_true')
    p.add_argument('--check-TRAIN-capacity',action='store_true')
    a=p.parse_args();a.attempt=ATTEMPT
    TRAIN_capacity(a) if a.check_TRAIN_capacity else assemble_head_continuation(a) if a.assemble_head else prepare(a)
