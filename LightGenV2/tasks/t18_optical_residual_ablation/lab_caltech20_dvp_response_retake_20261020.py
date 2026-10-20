"""Prepare a fresh failed-layer retake from an independently passed parent.

Artifact namespaces only: no engineering copy, changed physical graph or lower
gate. Old failed-layer CCDs remain in their original namespace, never linked in.
"""
import argparse
from pathlib import Path

ATTEMPT='response_20261020'


def response_drift_admitted(old,new_before,new_after):
    """Paired TRAIN diagnostic, not a new TEST score or a relaxed layer gate."""
    return (old['correct']-max(new_before['correct'],new_after['correct'])>=5
            and min(new_before['nll'],new_after['nll'])>old['nll']+1.)


def check_TRAIN_drift(a):
    """Independently replay unchanged old/new nominal TRAIN observations."""
    import torch
    import numpy as np
    from PIL import Image
    import lab_caltech20_dvp_layerwise_20261019 as m
    assert a.depth==4,'Only the physically evidenced L4 drift is admitted here'
    torch.set_num_threads(2);session=a.sequence/'L4_strict8pp_20261020'
    probes=a.sequence/'TRAIN_response_verification_desktop_20261020'
    target=probes/'TRAIN_response_revalidation_admission.json';assert not target.exists()
    report,proof=[m.read(probes/n) for n in ('report.json','independent_audit.json')]
    assert report['state']=='complete' and proof['passed'] and proof['report_SHA256']==m.sha(probes/'report.json')
    assert not report['TEST_used'] and report['SDK_released'] and report['hardware_unchanged']
    assert all(m.sha(path)==digest for path,digest in report['protected'].items())
    stability=m.read(a.sequence/'TRAIN_response_fit_20261020/stability.json')
    assert stability['passed'] and next(row for row in stability['depths'] if row['depth']==4)['within_connection_repeat_passed']
    data,dm=m.data_for(a.sequence);labels=data['train'][1][:20];assert labels.tolist()==list(range(20))
    fields,identities=m.fields_for(session,4,data,dm,m.session_hardware_digest(a.sequence))
    incumbent=session/'fits/2_Huber_continue_repaired_20261020';ir,ia=[m.read(incumbent/n) for n in ('report.json','independent_audit.json')]
    assert ia['passed'] and ia['report_SHA256']==m.sha(incumbent/'report.json')
    model,_=m.model_for(incumbent/'best.pt',ir['results']['best']['PT_SHA256'],4,session/'source_reference.ts')
    original=m.state_cpu(model);capture=session/'capture/2';cr=m.read(capture/'report.json');scores={};crop_proof=[]
    for role in ('old','nominal_before','nominal_after'):
        crops=[]
        for i in range(20):
            old_receipt=m.read(capture/'train'/f'{i:06d}.json')
            assert old_receipt['sample_id']==str(data['train'][2][i])
            if role=='old':crop=np.asarray(fields['2']['train'][i]).copy()
            else:
                receipt=m.read(probes/'L4'/f'{role}_{i:02d}.json');path=probes/'L4'/f'{role}_{i:02d}.png'
                assert receipt['sample_id']==old_receipt['sample_id'] and receipt['canonical_SHA256']==old_receipt['canonical_SHA256']
                assert receipt['source_capture_phase_SHA256']==cr['phase_SHA256']==receipt['phase_SHA256']
                assert receipt['crop_SHA256']==m.sha(path)
                crop=np.asarray(Image.open(path)).copy();crop_proof.append(dict(condition=role,index=i,crop_SHA256=m.sha(path),receipt_SHA256=m.sha(probes/'L4'/f'{role}_{i:02d}.json')))
            crops.append(crop)
        crops=np.stack(crops);parts=[]
        with torch.no_grad():
            for begin in range(0,20,4):
                ix=np.arange(begin,begin+4);model.measured={k:torch.from_numpy(np.asarray(v['train'][ix]).copy()) for k,v in fields.items()}
                model.measured['2']=torch.from_numpy(crops[ix].copy())
                parts.append(model(m.encode_images(data['train'][0][ix]))[0].numpy())
        p=np.concatenate(parts);scores[role]=dict(correct=int((p.argmax(1)==labels).sum()),nll=float(-np.log(p[np.arange(20),labels].clip(1e-12)).mean()))
    assert all(torch.equal(value,m.state_cpu(model)[key]) for key,value in original.items())
    admitted=response_drift_admitted(scores['old'],scores['nominal_before'],scores['nominal_after'])
    q=dict(passed=True,depth=4,failed_stage='2',TRAIN=20,TEST_inference=False,VALID=0,SDK_used=False,new_PT_written=False,
        response_revalidation_admitted=admitted,scores=scores,identical_canonical_inputs_and_nominal_phase=True,
        current_connection_repeat_passed=True,all_hardware_and_LUT_SHA_unchanged=True,crop_proof=crop_proof,
        data_SHA256=dm['sha256'],prefix_SHA256=identities,incumbent_PT_SHA256=m.sha(incumbent/'best.pt'),
        source_capture_report_SHA256=m.sha(capture/'report.json'),probe_report_SHA256=m.sha(probes/'report.json'),
        probe_audit_SHA256=m.sha(probes/'independent_audit.json'),source_git=m.git_head(),source_SHA256=m.sha(__file__),
        scope='Evidence-based one fresh L4 failed-stage capture under current stable response, using the unchanged already-native-passed input parent. Not a new trained-mask claim, blind same-PT retry, hardware correction, or next-layer gate bypass. All old data preserved.')
    m.save(target,q);print(m.json.dumps({k:v for k,v in q.items() if k not in ('prefix_SHA256','crop_proof')}))


def TRAIN_capacity(a):
    """One fixed published readout estimator, TRAIN only, no new checkpoint."""
    import torch
    import numpy as np
    import lab_caltech20_dvp_layerwise_20261019 as m
    from lab_caltech20_residual_terminal_cpu_20261019 import shrunk_readout
    torch.set_num_threads(2);session=a.sequence/f'L{a.depth}_strict8pp_20261020';prefix={4:3,6:1}[a.depth]
    mask=session/'fits'/(('1' if a.depth==4 else 'router')+'_spatial_response_retrain_20261020')
    parent=session/'fits'/({4:'1_binary_repaired_20261020',6:'router_binary_repaired_20261020'}[a.depth])
    target=mask/'TRAIN_capacity.json';progress=mask/'TRAIN_capacity_progress.json';assert not target.exists() and not progress.exists()
    r,p,proof=[m.read(mask/n) for n in ('report.json','protocol.json','independent_audit.json')]
    assert r['state']=='complete' and proof['passed'] and proof['report_SHA256']==m.sha(mask/'report.json')
    data,dm=m.data_for(a.sequence);fields,identities=m.fields_for(session,prefix,data,dm,m.session_hardware_digest(a.sequence))
    assert p['prefix_SHA256']==identities and p['data_SHA256']==dm['sha256'] and p['parent_PT_SHA256']==m.sha(parent/'best.pt')
    cfg=m.read(Path(__file__).with_name('lab_caltech20_DVP_shrinkage_head_profile_20261020.json'))
    assert cfg['original_TRAIN_cache_SHA256']==dm['source_cache_SHA256'];counts=torch.tensor(cfg['original_TRAIN_counts'],dtype=torch.float64)
    assert int(counts.sum())==3141;labels=torch.from_numpy(data['train'][1].astype(np.int64));rows={}
    for role,path in (('parent',parent/'best.pt'),('mask_last',mask/'last.pt')):
        model,_=m.model_for(path,m.sha(path),a.depth,session/'source_reference.ts');state=m.state_cpu(model);parts=[]
        m.save(progress,dict(body=role,state='TRAIN_features',completed=0,total=1600,TEST_inference=False))
        print(m.json.dumps(dict(body=role,state='TRAIN_features',completed=0,total=1600)),flush=True)
        def collect(module,args):
            parts.append(args[0].detach().numpy().copy());completed=sum(len(v) for v in parts)
            if completed%128==0 or completed==1600:
                row=dict(body=role,state='TRAIN_features',completed=completed,total=1600,TEST_inference=False)
                m.save(progress,row);print(m.json.dumps(row),flush=True)
        hook=model.linear.register_forward_pre_hook(collect)
        try:actual,_=m.assess(model,data,fields,'train')
        finally:hook.remove()
        features=torch.tensor(np.concatenate(parts),dtype=torch.float64);assert features.shape==(1600,256)
        w,b,estimator=shrunk_readout(features,labels,counts);logits=features@w.T+b
        rows[role]=dict(PT_SHA256=m.sha(path),TRAIN_original_correct=actual['correct'],TRAIN_original_NLL=actual['nll'],
            TRAIN_capacity_correct=int((logits.argmax(1)==labels).sum()),TRAIN_capacity_NLL=float(torch.nn.functional.cross_entropy(logits,labels)),estimator=estimator)
        assert all(torch.equal(v,m.state_cpu(model)[k]) for k,v in state.items())
    old,new=rows['parent'],rows['mask_last']
    improved=(new['TRAIN_capacity_correct']>=old['TRAIN_capacity_correct'] and new['TRAIN_capacity_NLL']<old['TRAIN_capacity_NLL']-1e-4)
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
    endings={'readout':'_spatial_last_readout_20261020','binary':'_spatial_last_binary_20261020','binary_continue':'_spatial_last_binary_continue_20261020'}
    head=old/'fits'/(rootname+endings[a.head_mode])
    parent=old/'fits'/({4:'1_binary_repaired_20261020',6:'router_binary_repaired_20261020'}[a.depth])
    mr,mp,ma=[m.read(mask/n) for n in ('report.json','protocol.json','independent_audit.json')]
    hr,hp,ha=[m.read(head/n) for n in ('report.json','protocol.json','independent_audit.json')]
    assert ma['passed'] and ha['passed'] and mr['state']==hr['state']=='complete'
    assert ma['report_SHA256']==m.sha(mask/'report.json') and ha['report_SHA256']==m.sha(head/'report.json')
    assert mp['original_readout_frozen'] and m.sha(mask/'last.pt')==mr['results']['last']['PT_SHA256']
    nodes=[];current=head
    for _ in range(4):
        nr,node_protocol,na=[m.read(current/n) for n in ('report.json','protocol.json','independent_audit.json')]
        assert nr['state']=='complete' and nr['all_nonhead_frozen'] and na['passed']
        assert na['report_SHA256']==m.sha(current/'report.json') and na['protocol_SHA256']==m.sha(current/'protocol.json')
        assert node_protocol['depth']==a.depth and node_protocol['prefix']==prefix and set(node_protocol['allowed_keys'])=={'linear.weight','linear.bias'}
        assert node_protocol['prefix_SHA256']==mp['prefix_SHA256'] and node_protocol['data_SHA256']==mp['data_SHA256']
        nodes.append(dict(path=str(current),selected_best_PT_SHA256=m.sha(current/'best.pt'),actual_parent_PT_SHA256=node_protocol['parent_PT_SHA256'],
            report_SHA256=m.sha(current/'report.json'),audit_SHA256=m.sha(current/'independent_audit.json')))
        if node_protocol['parent_PT_SHA256']==m.sha(mask/'last.pt'):
            initial_head_protocol=node_protocol
            break
        matches=[]
        for directory in (old/'fits').iterdir():
            if (directory/'report.json').exists():
                info=m.read(directory/'report.json')
                if info.get('results',{}).get('best',{}).get('PT_SHA256')==node_protocol['parent_PT_SHA256']:matches.append(directory)
        assert len(matches)==1;current=matches[0]
    else:raise AssertionError('Bounded head continuation ancestry exceeded')
    evidence=m.read(mask/'TRAIN_capacity.json')
    assert evidence['passed'] and evidence['capacity_improved'] and not evidence['TEST_inference']
    assert evidence['source_checkpoint_role']=='last' and evidence['source_PT_SHA256']==initial_head_protocol['parent_PT_SHA256']
    assert initial_head_protocol['config']['TRAIN_capacity_evidence_SHA256']==m.sha(mask/'TRAIN_capacity.json')
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
        assert set(state)==set(origin) and all(state[k].dtype==v.dtype for k,v in origin.items())
        assert all(torch.equal(v,state[k]) for k,v in intermediate.items() if not k.startswith('linear.'))
        assert all(torch.equal(v,state[k]) for k,v in origin.items() if k not in allowed)
    out=old/'fits'/(rootname+'_spatial_response_compound_20261020');assert not out.exists();out.mkdir()
    p=dict(parent_PT_SHA256=m.sha(parent/'best.pt'),prefix=prefix,depth=a.depth,prefix_SHA256=identities,data_SHA256=dm['sha256'],
        allowed_keys=sorted(allowed),TRAIN=1600,TEST=411,VALID=0,TEST_gradient=False,TEST_informed_development=True,
        source_git=m.git_head(),source_SHA256=m.sha(__file__),compound_sequential_workflow=True,assembly_optimizer_steps=0,
        mask_fit=dict(path=str(mask),initial_PT_SHA256=mp['parent_PT_SHA256'],actual_output_role='last',actual_output_PT_SHA256=m.sha(mask/'last.pt'),
                      report_SHA256=m.sha(mask/'report.json'),audit_SHA256=m.sha(mask/'independent_audit.json')),
        head_fit=dict(path=str(head),actual_initialization_role='mask_last',actual_initialization_PT_SHA256=initial_head_protocol['parent_PT_SHA256'],
                      selected_role='best',report_SHA256=m.sha(head/'report.json'),audit_SHA256=m.sha(head/'independent_audit.json')),
        head_continuation_chain=nodes,
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
    ending={'compound':'_spatial_response_compound_20261020','trained_best':'_spatial_response_retrain_20261020','native_anchor':'_spatial_native_anchor_20261021'}
    if a.parent_mode=='response_revalidation':
        assert a.depth==4
        driftpath=a.sequence/'TRAIN_response_verification_desktop_20261020/TRAIN_response_revalidation_admission.json';drift=m.read(driftpath)
        assert drift['passed'] and drift['response_revalidation_admitted'] and not drift['TEST_inference']
        assert drift['source_SHA256']==m.sha(__file__) and drift['depth']==4 and drift['failed_stage']=='2'
        assert drift['source_capture_report_SHA256']==m.sha(old/'capture/2/report.json')
        assert drift['probe_report_SHA256']==m.sha(driftpath.parent/'report.json') and drift['probe_audit_SHA256']==m.sha(driftpath.parent/'independent_audit.json')
        name='1_binary_repaired_20261020'
    else:name=('1' if a.depth==4 else 'router')+ending[a.parent_mode]
    prior=old/'fits'/name;r=m.read(prior/'report.json');p=m.read(prior/'protocol.json')
    audit=m.read(prior/'independent_audit.json')
    digest=check_score(r,a.depth,prefix)
    assert not r['selected_e0'],'No changed response-trained candidate: no identical-PT reshoot'
    assert audit['passed'] and audit['report_SHA256']==m.sha(prior/'report.json')
    assert audit['protocol_SHA256']==m.sha(prior/'protocol.json') and audit['selected_correct']==r['selected_correct']
    data,dm=m.data_for(a.sequence);hardware=m.session_hardware_digest(a.sequence)
    fields,identities=m.fields_for(old,prefix,data,dm,hardware)
    if a.parent_mode=='response_revalidation':
        assert drift['data_SHA256']==dm['sha256'] and all(drift['prefix_SHA256'][k]==v for k,v in identities.items())
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
        response_revalidation=a.parent_mode=='response_revalidation',
        TRAIN_response_revalidation_admission_SHA256=m.sha(driftpath) if a.parent_mode=='response_revalidation' else None,
        source_git=m.git_head(),source_SHA256=m.sha(__file__),engineering_copy=False,immutable_artifact_links=True))
    m.save(session/'prepared.json',dict(passed=True,reuse_manifest_SHA256=m.sha(session/'reuse_manifest.json'),
        first_retake=stage,source_git=m.git_head(),no_automatic_next_layer=True))
    print(m.json.dumps(dict(prepared=True,depth=a.depth,first_retake=stage,parent_fit=prior.name,
        parent_PT_SHA256=digest,native_correct=r['selected_correct'],failed_layer_CCD_reused=False,SDK_used=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sequence',type=Path,required=True)
    p.add_argument('--depth',type=int,choices=(4,6),required=True)
    p.add_argument('--parent-mode',choices=('trained_best','compound','native_anchor','response_revalidation'),default='trained_best')
    p.add_argument('--assemble-head',action='store_true')
    p.add_argument('--head-mode',choices=('readout','binary','binary_continue'),default='readout')
    p.add_argument('--check-TRAIN-capacity',action='store_true')
    p.add_argument('--check-TRAIN-drift',action='store_true')
    a=p.parse_args();a.attempt=ATTEMPT
    check_TRAIN_drift(a) if a.check_TRAIN_drift else TRAIN_capacity(a) if a.check_TRAIN_capacity else assemble_head_continuation(a) if a.assemble_head else prepare(a)
