"""Separate CPU process: audit one complete physical-forward closed-loop round.

No camera, optimizer, modified state, inference transform, or permission for a
following layer. Verify saved predictions against the ORIGINAL full graph.
"""
import argparse
from pathlib import Path
import os


def audit(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import numpy as np
    import torch
    import lab_caltech20_dvp_layerwise_20261019 as m
    from lab_caltech20_dvp_closedloop_20261021 import SPECS,HARDWARE,validate_state,original_replay,audit_capture
    torch.set_num_threads(2);torch.manual_seed(73)
    assert a.depth in SPECS and a.round in (1,2)
    out=a.sequence/'TRAIN_fullbatch_closedloop_20261021';folder=out/f'L{a.depth}_round{a.round}';proof=folder/'independent_round_audit.json'
    assert not proof.exists()
    root_protocol=m.read(out/'protocol.json');spec=SPECS[a.depth]
    assert m.session_hardware_digest(a.sequence)==HARDWARE==root_protocol['hardware_SHA256']
    hardware=m.read(a.sequence/'calibration/admission/admitted_hardware.json')
    for path_key,hash_key in (('coordinates','coordinates_SHA256'),('phase_lut','phase_LUT_SHA256'),('response_lut','response_lut_SHA256')):
        assert m.sha(hardware[path_key])==hardware[hash_key],('Fixed physical file changed',path_key)
    data,dm=m.data_for(a.sequence);assert dm['sha256']==root_protocol['data_SHA256']
    session=a.sequence/f'L{a.depth}_response_20261020';reference=session/'source_reference.ts'
    previous,previous_sha=m.fields_for(session,spec['prefix']-1,data,dm,HARDWARE)
    cap=folder/'capture';cr=m.read(cap/'report.json');ca=m.read(cap/'independent_audit.json');cp=m.read(cap/'protocol.json')
    assert cr['state']=='complete' and cr['SDK_released'] and cr['physical_ccd']==2011 and cr['TRAIN']==1600 and cr['TEST']==411
    assert ca['passed'] and ca['report_SHA256']==m.sha(cap/'report.json') and ca['protocol_SHA256']==m.sha(cap/'protocol.json')
    assert cr['protocol_SHA256']==m.sha(cap/'protocol.json') and cr['data_SHA256']==dm['sha256']
    assert cp['source_SHA256']==m.sha(m.HERE/'lab_caltech20_dvp_closedloop_20261021.py')
    assert cp['protected']=={p:m.sha(p) for p in cp['protected']}
    proposal=folder/'proposal';pr=m.read(proposal/'report.json');pp=m.read(proposal/'protocol.json');candidate=proposal/'candidate.pt'
    digest=m.sha(candidate);assert digest==pr['candidate_PT_SHA256']==cr['PT_SHA256']==cp['PT_SHA256']
    assert pr['TRAIN_consumed']==1600 and pr['mask_updates']==1 and pp['effective_gradient_batch']==1600 and pp['TEST_gradient'] is False
    assert pp['forward_replay_max_error']<=3e-6
    assert all(v['max_phase_delta']<=.040001 for v in pp['phase_steps'])
    fields=dict(previous);fields[spec['stage']]={};identities=dict(previous_sha)
    for role,size in (('train',1600),('test',411)):
        p=cap/role/'ccd.npy';assert m.sha(p)==cr['arrays'][role]['SHA256'] and cr['arrays'][role]['samples']==size
        array=np.load(p,mmap_mode='r');assert array.shape==(size,478,478) and array.dtype==np.uint8
        fields[spec['stage']][role]=array;identities[spec['stage']+'/'+role]=m.sha(p)
    model,_=m.model_for(candidate,digest,a.depth,reference);before=m.state_cpu(model)
    if a.round==1:
        proposal_parent=session/'fits'/spec['fit']/'best.pt';old_capture=session/'capture'/spec['stage']
        assert m.sha(proposal_parent)==spec['PT']
    else:
        proposal_parent=out/f'L{a.depth}_round1/suffix/best.pt';old_capture=out/f'L{a.depth}_round1/capture'
    parent,_=m.model_for(proposal_parent,m.sha(proposal_parent),a.depth,reference);parent_state=m.state_cpu(parent)
    lo,hi=parent.stage_slices[int(spec['stage'])];phase_keys={f'phase_sets.{k}' for k in range(lo,hi)}
    validate_state(m,model,parent_state,phase_keys)
    phase_delta={k:float((2*np.pi*(before[k].sigmoid()-parent_state[k].sigmoid())).abs().max()) for k in phase_keys}
    assert all(0<v<=.040001 for v in phase_delta.values())
    old_report=m.read(old_capture/'report.json');old_array=old_capture/'train/ccd.npy';assert m.sha(old_array)==old_report['arrays']['train']['SHA256']
    old_fields=dict(previous);old_fields[spec['stage']]={'train':np.load(old_array,mmap_mode='r')}
    parent_train,_=original_replay(parent,{'train':data['train']},old_fields,'train')
    fit=folder/'suffix';report=m.read(fit/'report.json');protocol=m.read(fit/'protocol.json');fa=m.read(fit/'independent_audit.json')
    assert report['state']=='complete' and report['protocol_SHA256']==m.sha(fit/'protocol.json')
    assert fa['passed'] and fa['report_SHA256']==m.sha(fit/'report.json') and fa['protocol_SHA256']==m.sha(fit/'protocol.json')
    assert protocol['depth']==a.depth and protocol['prefix']==spec['prefix'] and protocol['TEST_gradient'] is False
    allowed=m.allowed_keys(model,spec['prefix']);assert set(protocol['allowed_keys'])==allowed
    results={}
    for name in ('best','last'):
        pt=fit/(name+'.pt');r=report['results'][name];assert m.sha(pt)==r['PT_SHA256']
        current,_=m.model_for(pt,m.sha(pt),a.depth,reference);validate_state(m,current,before,allowed)
        metrics={}
        for role,size in (('test',411),('train',1600)):
            actual,prob=original_replay(current,data,fields,role);expected=r['TEST' if role=='test' else 'TRAIN']
            assert actual['samples']==size and actual['correct']==expected['correct'] and abs(actual['nll']-expected['nll'])<=3e-6
            metrics[role]=dict(correct=actual['correct'],NLL=actual['nll'])
            if role=='test':
                with np.load(fit/(name+'_TEST_predictions.npz')) as bank:
                    assert np.array_equal(bank['ids'],data[role][2]) and np.array_equal(bank['labels'],data[role][1])
                    error=float(np.abs(prob-bank['probabilities']).max());assert error<=3e-6
                    assert np.array_equal(prob.argmax(1),bank['probabilities'].argmax(1))
                    metrics[role]['prediction_maxabs']=error
        results[name]=dict(PT_SHA256=m.sha(pt),metrics=metrics)
    assert report['selected_correct']==results['best']['metrics']['test']['correct']
    # The post-mask pre-fine-tuning TRAIN result isolates the local adjoint
    # proposal from later suffix/head fitting; this is not an accuracy claim.
    raw_train,_=original_replay(model,data,fields,'train')
    result=dict(passed=True,source_git=m.git_head(),source_SHA256=m.sha(__file__),training_source_git=root_protocol['source_git'],
        depth=a.depth,round=a.round,TRAIN=1600,TEST=411,VALID=0,TEST_gradient=False,
        original_full_CPU_replay=True,all_upstream_and_current_capture_input_frozen=True,
        exact_candidate_capture_PT=digest,source_capture_report_SHA256=m.sha(cap/'report.json'),
        source_capture_audit_SHA256=m.sha(cap/'independent_audit.json'),
        source_fit_report_SHA256=m.sha(fit/'report.json'),source_fit_protocol_SHA256=m.sha(fit/'protocol.json'),
        data_SHA256=dm['sha256'],prefix_SHA256=identities,hardware_SHA256=HARDWARE,
        results=results,post_mask_before_suffix_TRAIN=raw_train,
        proposal_parent_PT_SHA256=m.sha(proposal_parent),parent_on_own_previous_TRAIN=parent_train,
        verified_underlying_phase_delta_max=phase_delta,proposal_all_other_tensors_frozen=True,
        actual_gate_passed=report['selected_correct']>=spec['target'],actual_target=spec['target'],
        final_target=spec['final'],full_physical=False,SDK_opened=False,GPU_used=False,
        TEST_informed_development=True,next_layer_permission=False)
    m.save(proof,result);print(m.json.dumps({k:v for k,v in result.items() if k not in ('prefix_SHA256','post_mask_before_suffix_TRAIN')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sequence',type=Path,required=True);p.add_argument('--depth',type=int,choices=(4,6),required=True);p.add_argument('--round',type=int,choices=(1,2),required=True);audit(p.parse_args())
