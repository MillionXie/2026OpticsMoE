"""One evidence-admitted L4 suffix continuation, not another CCD/mask search.

E12 is still the best suffix epoch and full TRAIN remains below85%. The
subsequent three head-only passes plateaued at275. Continue only the ORIGINAL
uncaptured suffix/electronics with all1600 TRAIN per epoch. The measured E2
input and current mask are immutable. No next physical layer below314/411.
"""
import argparse
from pathlib import Path
import os

CORE_SHA='977eb00b88d85f1deb31eda557aa9e753fc6b15dfb8cf30d0aa529268c9c10d5'


def admission(a):
    import numpy as np
    import torch
    import lab_caltech20_dvp_layerwise_20261019 as m
    import lab_caltech20_dvp_closedloop_20261021 as loop
    torch.set_num_threads(2);torch.manual_seed(73)
    assert m.sha(loop.__file__)==CORE_SHA
    session=a.sequence/'L4_response_20261020'
    folder=a.sequence/'TRAIN_fullbatch_adjoint_sign_20261021/L4_round1'
    proof=m.read(folder/'independent_round_audit.json');raw=m.read(folder/'suffix/report.json')
    assert proof['passed'] and proof['source_fit_report_SHA256']==m.sha(folder/'suffix/report.json')
    assert raw['selected_epoch']==raw['actual_epochs']==12 and not raw['passed']
    parent=folder/'fullTRAIN_row_curvature';report=m.read(parent/'report.json')
    assert report['state']=='complete' and report['selected_correct']==275 and report['actual_passes']==3
    assert report['results']['best']['TRAIN']['accuracy']<.85
    history=m.read(parent/'history.json')
    assert max(v['selected_correct'] for v in history if v['pass_index']==2)==max(v['selected_correct'] for v in history if v['pass_index']==3)==275
    digest=report['results']['best']['PT_SHA256'];assert m.sha(parent/'best.pt')==digest
    model,_=m.model_for(parent/'best.pt',digest,4,session/'source_reference.ts')
    raw_model,_=m.model_for(folder/'suffix/best.pt',proof['results']['best']['PT_SHA256'],4,session/'source_reference.ts')
    loop.validate_state(m,model,m.state_cpu(raw_model),{'linear.weight','linear.bias'})
    assert m.session_hardware_digest(a.sequence)==loop.HARDWARE
    data,dm=m.data_for(a.sequence)
    fields,identities=m.fields_for(session,3,data,dm,loop.HARDWARE)
    cap=folder/'capture';cr=m.read(cap/'report.json');ca=m.read(cap/'independent_audit.json')
    assert ca['passed'] and ca['report_SHA256']==m.sha(cap/'report.json')
    assert cr['physical_ccd']==2011 and cr['data_SHA256']==dm['sha256']
    fields['2']={}
    for role,n in (('train',1600),('test',411)):
        path=cap/role/'ccd.npy';assert m.sha(path)==cr['arrays'][role]['SHA256']
        array=np.load(path,mmap_mode='r');assert array.shape==(n,478,478) and array.dtype==np.uint8
        fields['2'][role]=array;identities['2/'+role]=m.sha(path)
    actual,_=loop.original_replay(model,data,fields,'test');assert actual['correct']==275
    return m,loop,data,dm,fields,identities,model,parent,folder


def run(a):
    import torch
    m,loop,data,dm,fields,identities,model,parent,folder=admission(a)
    assert not a.out.exists() and not m.LOCK.exists()
    cfg=m.read(m.HERE/'lab_caltech20_DVP_closedloop_profile_20261021.json')
    cfg.update(suffix_epochs=24,suffix_fit_seconds=600,
        _reference=str(a.sequence/'L4_response_20261020/source_reference.ts'))
    loop.gpu_admission(m,cfg)
    protocol=dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),core_SHA256=CORE_SHA,
        parent_PT_SHA256=m.sha(parent/'best.pt'),parent_report_SHA256=m.sha(parent/'report.json'),
        source_round_audit_SHA256=m.sha(folder/'independent_round_audit.json'),
        source_capture_report_SHA256=m.sha(folder/'capture/report.json'),
        prefix_SHA256=identities,data_SHA256=dm['sha256'],TRAIN=1600,TEST=411,VALID=0,
        actual_target=314,maximum_new_epochs=24,maximum_fit_seconds=600,
        optimizer='Fresh original AdamW; prior checkpoint did not retain optimizer state',
        learning_rates_and_loss_unchanged=True,current_E2_mask_and_all_upstream_frozen=True,
        only_original_uncaptured_G2_mask_and_permitted_electronics=True,new_CCD=0,
        TEST_gradient=False,TEST_informed_development=True,SDK_used=False,
        original_higher_PT_preserved=True,next_layer_started=False,automatic_retry=False)
    model,correct=loop.refine(m,data,fields,model,loop.SPECS[4],cfg,a.out)
    m.save(a.out/'continuation_protocol.json',protocol)
    print(m.json.dumps(dict(complete=True,correct=correct,target=314,
        actual_gate_passed=correct>=314,SDK_used=False,new_CCD=0,next_layer_started=False)),flush=True)
    del model;torch.cuda.empty_cache()


def audit(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import numpy as np
    m,loop,data,dm,fields,identities,parent_model,parent,folder=admission(a)
    report=m.read(a.out/'report.json');protocol=m.read(a.out/'continuation_protocol.json')
    assert report['state']=='complete' and report['protocol_SHA256']==m.sha(a.out/'protocol.json')
    assert protocol['source_SHA256']==m.sha(__file__) and protocol['parent_PT_SHA256']==m.sha(parent/'best.pt')
    assert protocol['prefix_SHA256']==identities and protocol['data_SHA256']==dm['sha256']
    before=m.state_cpu(parent_model);allowed=m.allowed_keys(parent_model,4);results={}
    for name in ('best','last'):
        digest=m.sha(a.out/(name+'.pt'));expected=report['results'][name]
        assert digest==expected['PT_SHA256']
        model,_=m.model_for(a.out/(name+'.pt'),digest,4,a.sequence/'L4_response_20261020/source_reference.ts')
        loop.validate_state(m,model,before,allowed);rows={}
        for role in ('train','test'):
            score,p=loop.original_replay(model,data,fields,role)
            ref=expected['TRAIN' if role=='train' else 'TEST']
            assert score['correct']==ref['correct'] and abs(score['nll']-ref['nll'])<=3e-6
            rows[role]=dict(correct=score['correct'],nll=score['nll'])
            if role=='test':
                with np.load(a.out/(name+'_TEST_predictions.npz')) as z:
                    assert np.array_equal(z['ids'],data['test'][2]) and np.array_equal(z['labels'],data['test'][1])
                    assert np.array_equal(z['probabilities'].argmax(1),p.argmax(1)) and np.max(np.abs(z['probabilities']-p))<=3e-6
        results[name]=dict(PT_SHA256=digest,metrics=rows)
    m.save(a.out/'independent_continuation_audit.json',dict(passed=True,
        source_git=m.git_head(),source_SHA256=m.sha(__file__),results=results,
        report_SHA256=m.sha(a.out/'report.json'),protocol_SHA256=m.sha(a.out/'continuation_protocol.json'),
        all1600TRAIN_and411TEST_original_CPU_verified=True,all_measured_masks_and_upstream_frozen=True,
        SDK_used=False,GPU_used=False,actual_gate_passed=report['selected_correct']>=314,
        full_physical=False,TEST_informed_development=True,next_layer_started=False))
    print(m.json.dumps(dict(passed=True,correct=report['selected_correct'],SDK_used=False)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('fit','audit'),required=True)
    p.add_argument('--sequence',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a) if a.mode=='fit' else audit(a)
