"""Independent CPU: byte-identical optical stimuli, changed observation epoch.

Compare ALL2011 old/new receipt identities and fixed device readbacks; replay
the exact retained original model on both old and new TRAIN/TEST. No source
checkpoint selection, optimization, camera, exposure/LUT/geometry mutation.
"""
import argparse
import os
from pathlib import Path


def same_stimulus(old,new):
    for key in ('sample_id','role','index','data_SHA256','phase_SHA256','canonical_SHA256'):
        assert old[key]==new[key],('Optical stimulus changed',key)
    assert old.get('input_SHA256',old.get('BMP_SHA256'))==new['BMP_SHA256']
    for key in ('exposure_us','gain','format','bits','shape','flip_h','flip_v','rotate','opposite','wait_ms'):
        assert old['frame_metadata'][key]==new['frame_metadata'][key],('Device readback changed',key)


def run(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import numpy as np
    import torch
    import lab_caltech20_dvp_layerwise_20261019 as m
    import lab_caltech20_dvp_closedloop_20261021 as loop
    torch.set_num_threads(2)
    out=a.sequence/'TRAIN_nominal_control_20261021/L6';proof=out/'independent_temporal_control_audit.json'
    assert not proof.exists()
    returned=m.read(out.with_suffix('.return.json'));launch=m.read(out.with_suffix('.launch.json'))
    r=m.read(out/'report.json');protocol=m.read(out/'protocol.json');cap=out/'capture'
    assert returned['returncode']==0 and r['state']=='complete' and r['SDK_released']
    assert r['protocol_SHA256']==m.sha(out/'protocol.json')
    assert protocol['source_SHA256']==launch['source_SHA256']==m.sha(m.HERE/'lab_caltech20_dvp_nominal_control_20261021.py')
    assert protocol['optimizer_updates']==0 and not protocol['new_mask']
    assert m.session_hardware_digest(a.sequence)==loop.HARDWARE
    data,dm=m.data_for(a.sequence);assert dm['sha256']==protocol['data_SHA256']
    session=a.sequence/'L6_response_20261020';oldcap=session/'capture/0'
    fields,identities=m.fields_for(session,2,data,dm,loop.HARDWARE)
    cr=m.read(cap/'report.json');ca=m.read(cap/'independent_audit.json')
    assert ca['passed'] and ca['report_SHA256']==m.sha(cap/'report.json')
    assert r['capture_report_SHA256']==m.sha(cap/'report.json') and r['capture_audit_SHA256']==m.sha(cap/'independent_audit.json')
    assert cr['physical_ccd']==2011 and cr['TRAIN']==1600 and cr['TEST']==411 and cr['VALID']==0
    assert m.sha(cap/'phase.bmp')==m.sha(oldcap/'phase.bmp')==r['exact_original_phase_SHA256']
    current=dict(fields);current['0']={};checked=0
    for role,size in (('train',1600),('test',411)):
        path=cap/role/'ccd.npy';assert m.sha(path)==cr['arrays'][role]['SHA256']
        array=np.load(path,mmap_mode='r');assert array.shape==(size,478,478)
        current['0'][role]=array
        for i in range(size):
            old=m.read(oldcap/role/f'{i:06d}.json');new=m.read(cap/role/f'{i:06d}.json')
            assert new['sample_id']==str(data[role][2][i]) and new['PT_SHA256']==loop.SPECS[6]['PT']
            same_stimulus(old,new);checked+=1
        m.save(out/'temporal_audit_progress.json',dict(state='matching_exact_stimuli',role=role,checked=checked,expected=2011))
    assert checked==2011
    pt=session/'fits'/loop.SPECS[6]['fit']/'best.pt';digest=m.sha(pt)
    assert digest==loop.SPECS[6]['PT']==r['exact_original_PT_SHA256']
    model,_=m.model_for(pt,digest,6,session/'source_reference.ts');before=m.state_cpu(model)
    results={};probabilities={}
    for epoch,observations in (('old',fields),('current',current)):
        results[epoch]={}
        for role in ('test','train'):
            m.save(out/'temporal_audit_progress.json',dict(state='original_CPU_replay',observation_epoch=epoch,role=role))
            metric,p=loop.original_replay(model,data,observations,role)
            results[epoch][role]=dict(samples=metric['samples'],correct=metric['correct'],NLL=metric['nll'])
            probabilities[epoch+'_'+role]=p
    assert results['old']['test']['correct']==291
    assert results['current']['test']['correct']==r['current_same_PT_TEST']['correct']
    with np.load(out/'TEST_predictions.npz') as z:
        assert np.array_equal(z['ids'],data['test'][2]) and np.array_equal(z['labels'],data['test'][1])
        error=float(np.max(np.abs(z['probabilities']-probabilities['current_test'])))
        assert error<=3e-6 and np.array_equal(z['probabilities'].argmax(1),probabilities['current_test'].argmax(1))
    loop.validate_state(m,model,before,set())
    for role in ('train','test'):
        np.savez_compressed(out/(role+'_temporal_comparison.npz'),ids=data[role][2],labels=data[role][1],old=probabilities['old_'+role],current=probabilities['current_'+role])
    mean_ratios={role:float(np.median(np.asarray(current['0'][role]).mean((1,2))/np.asarray(fields['0'][role]).mean((1,2)).clip(1e-12))) for role in ('train','test')}
    m.save(proof,dict(passed=True,source_git=m.git_head(),source_SHA256=m.sha(__file__),
        original_PT_SHA256=digest,source_control_report_SHA256=m.sha(out/'report.json'),
        all2011_canonical_input_BMP_phase_ID_and_device_readbacks_identical=True,
        source_capture_audit_SHA256=m.sha(cap/'independent_audit.json'),
        old_prefix_SHA256=identities,current_array_SHA256=cr['arrays'],results=results,
        current_TEST_prediction_maxabs=error,all_model_state_frozen=True,matched_frame_mean_ratio_median=mean_ratios,
        same_device_readbacks_not_all_physical_conditions=True,
        interpretation='A real observation-epoch effect with identical recorded stimuli/readbacks. Does NOT uniquely identify ambient light, sensor ISP, mechanical drift or another physical cause.',
        hardware_files_modified=False,SDK_used=False,GPU_used=False,optimizer_updates=0,
        TEST_informed_development=True,full_physical=False,next_layer_started=False))
    print(m.json.dumps(dict(passed=True,identical_stimuli=checked,results=results,SDK_used=False)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sequence',type=Path,required=True);run(p.parse_args())
