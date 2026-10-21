"""One physical TRAIN-loss-guided reverse-direction trial, not a mask grid.

Only if the completed, independently audited first direction worsened full
TRAIN NLL AND full TRAIN correct count. Mirror the same original-mask step
about its exact original parent, retain every higher model, and measure NEW
full TRAIN1600 -> TEST411. No claim of a calibrated physical Jacobian.
"""
import argparse
import os
from pathlib import Path


def mirrored_state(parent,positive,keys):
    import torch
    import math
    assert set(parent)==set(positive)
    assert all(torch.equal(v,positive[k]) for k,v in parent.items() if k not in keys)
    state={k:v.clone() for k,v in parent.items()};records={}
    for key in keys:
        old=2*math.pi*parent[key].sigmoid();plus=2*math.pi*positive[key].sigmoid()
        target=(2*old-plus).clamp(1e-4,2*math.pi-1e-4)
        state[key]=torch.logit(target/(2*math.pi));actual=2*math.pi*state[key].sigmoid()
        assert float((plus-old).abs().max())<=.040001 and float((actual-old).abs().max())<=.040003
        assert float((actual-target).abs().max())<=2e-6
        records[key]=dict(positive_delta_max=float((plus-old).abs().max()),reverse_delta_max=float((actual-old).abs().max()))
    return state,records


def selftest():
    import torch
    torch.manual_seed(73);parent={'phase':torch.randn(224,224)*.1,'frozen':torch.ones(3)}
    phi=2*torch.pi*parent['phase'].sigmoid();delta=torch.randn_like(phi).clamp(-2,2)*.015
    positive=dict(parent,phase=torch.logit((phi+delta)/(2*torch.pi)))
    state,records=mirrored_state(parent,positive,{'phase'});assert torch.equal(state['frozen'],parent['frozen'])
    error=float((state['phase'].sigmoid()+positive['phase'].sigmoid()-2*parent['phase'].sigmoid()).abs().max());assert error<5e-7
    print(dict(passed=True,underlying_phase_mirror_error=error,all_other_original_state_frozen=True,SDK_opened=False))


def run(a):
    import numpy as np
    import torch
    import psutil
    import lab_caltech20_dvp_layerwise_20261019 as m
    import lab_caltech20_dvp_closedloop_20261021 as loop
    torch.set_num_threads(2);torch.manual_seed(73)
    first=a.sequence/'TRAIN_fullbatch_closedloop_20261021';old_report=m.read(first/'report.json');returned=m.read(first.with_suffix('.return.json'));launch=m.read(first.with_suffix('.launch.json'))
    assert old_report['state']=='complete' and returned['returncode']==0 and not m.LOCK.exists()
    assert not (psutil.pid_exists(launch['pid']) and psutil.Process(launch['pid']).create_time()==launch['created']),'Prior producer not exited'
    spec_profile=m.read(a.profile);assert spec_profile['schema']=='full_TRAIN_reverse_direction_v1' and spec_profile['maximum_trials_each_depth']==1
    base_profile=m.HERE/'lab_caltech20_DVP_closedloop_profile_20261021.json';cfg=m.read(base_profile)
    cfg.update(suffix_epochs=spec_profile['suffix_epochs'],suffix_fit_seconds=spec_profile['suffix_fit_seconds'])
    m,data,dm,hardware,roots,parents=loop.base_admission(a)
    for path_key,hash_key in (('coordinates','coordinates_SHA256'),('phase_lut','phase_LUT_SHA256'),('response_lut','response_lut_SHA256')):assert m.sha(hardware[path_key])==hardware[hash_key]
    loop.gpu_admission(m,cfg);assert not a.out.exists();a.out.mkdir()
    protocol=dict(method='physical_full_TRAIN_reverse_direction_v1',source_git=m.git_head(),source_SHA256=m.sha(__file__),config=spec_profile,base_profile_SHA256=m.sha(base_profile),hardware_SHA256=loop.HARDWARE,data_SHA256=dm['sha256'],parents=parents,TRAIN=1600,TEST=411,VALID=0,TEST_gradient=False,TEST_informed_development=True,one_reverse_direction_not_grid=True,approximate_adjoint_not_calibrated=True,actual_targets={'4':314,'6':322},final_targets={'4':342,'6':350},recurring=False,automatic_retry=False)
    m.save(a.out/'protocol.json',protocol);history=[]
    for depth in (4,6):
        s=loop.SPECS[depth];source=first/f'L{depth}_round1';audit=m.read(source/'independent_round_audit.json')
        assert audit['passed'] and audit['source_fit_report_SHA256']==m.sha(source/'suffix/report.json')
        parent_train=audit['parent_on_own_previous_TRAIN'];new_train=audit['post_mask_before_suffix_TRAIN']
        incumbent=next(v for v in old_report['rows'] if v['depth']==depth);global_correct=incumbent['correct']
        extra=source/'fullTRAIN_row_curvature/report.json'
        if extra.exists():global_correct=max(global_correct,m.read(extra)['selected_correct'])
        lda=source/'fullTRAIN_OAS_head/report.json'
        if lda.exists():global_correct=max(global_correct,m.read(lda)['selected_correct'])
        eligible=(global_correct<s['target'] and new_train['nll']>=parent_train['nll']+spec_profile['minimum_TRAIN_NLL_worsening'] and parent_train['correct']-new_train['correct']>=spec_profile['minimum_TRAIN_correct_loss'])
        if not eligible:
            row=dict(depth=depth,state='not_admitted',reason='No full-TRAIN deterioration evidence or actual stage already passed',global_correct=global_correct,new_physical_frames=0);history.append(row);m.save(a.out/'history.json',history);continue
        folder=a.out/f'L{depth}_round1';folder.mkdir();proposal=folder/'proposal';proposal.mkdir()
        session,parent_pt,fields,identities,model=roots[depth]
        positive=source/'proposal/candidate.pt';pr=m.read(source/'proposal/report.json');assert m.sha(positive)==pr['candidate_PT_SHA256'] and pr['TRAIN_consumed']==1600
        pos=torch.load(positive,map_location='cpu',weights_only=False)['model'];before=m.state_cpu(model);keys=loop.mask_keys(model,s['stage']);state,steps=mirrored_state(before,pos,keys)
        model.load_state_dict(state);loop.validate_state(m,model,before,keys)
        pp=dict(method='mirror_of_audited_full_TRAIN_step',source_git=m.git_head(),source_SHA256=m.sha(__file__),original_parent_PT_SHA256=m.sha(parent_pt),positive_PT_SHA256=m.sha(positive),source_first_direction_audit_SHA256=m.sha(source/'independent_round_audit.json'),source_full_TRAIN_gradient_report_SHA256=m.sha(source/'proposal/report.json'),TRAIN_basis_records=1600,new_TRAIN_gradient_records=0,mask_updates=1,phase_steps=steps,TEST_gradient=False,VALID=0,full_TRAIN_feedback=dict(parent_correct=parent_train['correct'],parent_NLL=parent_train['nll'],positive_correct=new_train['correct'],positive_NLL=new_train['nll']),global_incumbent_correct=global_correct,old_parent_is_control_not_artificial_new_baseline=True)
        m.save(proposal/'protocol.json',pp);digest=loop.saved_model(m,model,proposal/'candidate.pt',pp)
        m.save(proposal/'report.json',dict(state='complete',candidate_PT_SHA256=digest,TRAIN_basis_records=1600,new_TRAIN_gradient_records=0,mask_updates=1,original_upstream_frozen=True,global_incumbent_correct=global_correct,new_mask_needs_new_CCD=True))
        previous={k:v for k,v in fields.items() if k!=s['stage']};m.save(a.out/'progress.json',dict(state='new_reverse_mask_capture',depth=depth,candidate_PT_SHA256=digest,global_incumbent_correct=global_correct))
        newfields=loop.acquire(m,a,data,dm,hardware,previous,model,digest,s['stage'],folder/'capture')
        loop.gpu_admission(m,cfg);fitcfg=dict(cfg,_reference=str(session/'source_reference.ts'));model,correct=loop.refine(m,data,newfields,model,s,fitcfg,folder/'suffix')
        row=dict(depth=depth,state='complete',candidate_correct=correct,global_incumbent_correct=global_correct,improved_global=correct>global_correct,target=s['target'],actual_gate_passed=correct>=s['target'],new_physical_frames=2011,full_physical=False,fit_report_SHA256=m.sha(folder/'suffix/report.json'),next_layer_started=False)
        history.append(row);m.save(a.out/'history.json',history);m.save(a.out/f'L{depth}_current.json',row);print(m.json.dumps(row),flush=True)
    m.save(a.out/'report.json',dict(state='complete',history=history,SDK_released=not m.LOCK.exists(),actual_gate_not_relaxed=True,all_old_PT_and_CCD_preserved=True,source_first_workflow_report_SHA256=m.sha(first/'report.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selftest',action='store_true');p.add_argument('--sequence',type=Path);p.add_argument('--profile',type=Path);p.add_argument('--out',type=Path);a=p.parse_args();selftest() if a.selftest else run(a)
