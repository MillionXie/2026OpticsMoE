"""One original-PT L6 temporal control after TRAIN same-nominal drift evidence.

Full fresh TRAIN1600 -> TEST411, same camera/settings/mask/input lineage. No
proposal, gradient, optimizer, new mask, TEST-based mask choice or CUDA model.
Can share the desktop display with a separate original L4 suffix GPU fit.
This controls observations across time, not a simultaneous isolated light test.
"""
import argparse
import os
from pathlib import Path

CORE_SHA='977eb00b88d85f1deb31eda557aa9e753fc6b15dfb8cf30d0aa529268c9c10d5'


def run(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import numpy as np
    import torch
    import psutil
    import lab_caltech20_dvp_layerwise_20261019 as m
    import lab_caltech20_dvp_closedloop_20261021 as loop
    torch.set_num_threads(2);torch.manual_seed(73)
    assert m.sha(loop.__file__)==CORE_SHA
    campaign=a.sequence/'TRAIN_fullbatch_adjoint_sign_20261021'
    recovery=campaign/'L6_USB_recovery_20261021';returned=m.read(recovery.with_suffix('.return.json'))
    launch=m.read(recovery.with_suffix('.launch.json'))
    assert returned['returncode']==0
    assert not (psutil.pid_exists(launch['pid']) and psutil.Process(launch['pid']).create_time()==launch['created'])
    preflight=recovery/'TRAIN_light_preflight';evidence=m.read(preflight/'nominal_TRAIN20_replay.json')
    assert evidence['same_exact_PT_and_nominal_phase_and_inputs'] and evidence['TRAIN']==20
    assert not evidence['TEST_used'] and evidence['old_nominal']['correct']==20
    assert evidence['current_nominal']['correct']<=18 and evidence['current_nominal']['nll']>evidence['old_nominal']['nll']+.1
    assert not a.out.exists() and not m.LOCK.exists();a.out.mkdir()
    m,data,dm,hardware,roots,parents=loop.base_admission(a)
    session,pt,fields,identities,model=roots[6];digest=m.sha(pt)
    assert digest==loop.SPECS[6]['PT']==evidence['PT_SHA256']
    original=m.state_cpu(model)
    protocol=dict(method='single_original_nominal_L6_temporal_control',source_git=m.git_head(),
        source_SHA256=m.sha(__file__),core_SHA256=CORE_SHA,PT_SHA256=digest,
        source_TRAIN20_evidence_SHA256=m.sha(preflight/'nominal_TRAIN20_replay.json'),
        hardware_SHA256=loop.HARDWARE,data_SHA256=dm['sha256'],prefix_SHA256=identities,
        source_old_parent=parents['6'],TRAIN=1600,TEST=411,VALID=0,new_model_parameters=0,
        optimizer_updates=0,new_mask=False,old_nominal_phase=True,settings_LUT_geometry_unchanged=True,
        CUDA_model_used=False,desktop_SLM_display_uses_GPU=True,single_SDK=True,
        automatic_retry=False,recurring=False,old_CCD_and_PT_preserved=True,
        comparison='Different acquisition times; cannot uniquely attribute differences to the nearby light',
        purpose='Separate retained original-mask observation drift from a newly proposed optical-mask step',
        next_layer_started=False,TEST_gradient=False,TEST_informed_development=True)
    m.save(a.out/'protocol.json',protocol)
    previous={k:v for k,v in fields.items() if k!='0'}
    current=loop.acquire(m,a,data,dm,hardware,previous,model,digest,'0',a.out/'capture')
    assert m.sha(a.out/'capture/phase.bmp')==m.sha(session/'capture/0/phase.bmp')
    loop.validate_state(m,model,original,set())
    test,prob=loop.original_replay(model,data,current,'test')
    np.savez_compressed(a.out/'TEST_predictions.npz',ids=data['test'][2],labels=data['test'][1],probabilities=prob)
    m.save(a.out/'report.json',dict(state='complete',new_capture_TRAIN=1600,new_capture_TEST=411,
        exact_original_PT_SHA256=digest,exact_original_phase_SHA256=m.sha(a.out/'capture/phase.bmp'),
        capture_report_SHA256=m.sha(a.out/'capture/report.json'),capture_audit_SHA256=m.sha(a.out/'capture/independent_audit.json'),
        protocol_SHA256=m.sha(a.out/'protocol.json'),current_same_PT_TEST=test,old_same_PT_TEST_correct=291,
        same_checkpoint_and_nominal_mask=True,all_state_frozen=True,optimizer_updates=0,
        no_new_training_improvement_claim=True,not_unique_ambient_light_causality=True,
        TEST_predictions_SHA256=m.sha(a.out/'TEST_predictions.npz'),SDK_released=not m.LOCK.exists(),
        CUDA_model_used=False,next_layer_started=False,full_physical=False))
    print(m.json.dumps(dict(complete=True,current_nominal_TEST_correct=test['correct'],old_nominal_TEST_correct=291,
        optimizer_updates=0,SDK_released=not m.LOCK.exists(),no_mask_change=True)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sequence',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);run(p.parse_args())
