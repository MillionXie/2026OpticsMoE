"""Deploy the explicitly frozen346/411 L4 residual model, TRAIN60/class.

Same existing inference, layer-major capture and original5140 readout. No optical
mask update in this first complete deployment; never reuse another model's CCD.
The first24 TRAIN frames are retained and signal-gated as part of full capture.
"""
import argparse,json,os,sys,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];sys.path[:0]=[str(HERE),str(ROOT)]
import numpy as np
import torch
from lab_caltech20_rich_model_20261018 import load_checkpoint,encode_images,sha,DIRECT
from lab_caltech20_rich_layerwise_20261018 import load_data
from lab_caltech20_rich_capture_20261018 import layer_forward,metrics
from lab_caltech20_controls46_20261018 import capture_group,adapt_head,LOCK
from lab_caltech10_20261015 import save
PT_SHA='7b0d7f055a16f7f6bbf9858c9ecaea572ecd0b0d6e96c618630f4a4e077c09a5'


@torch.inference_mode()
def gate(a):
    torch.set_num_threads(2);datasets,dm=load_data(a.session)
    model,_=load_checkpoint(a.session/'residual_L4.pt',PT_SHA,4,.3,alpha_schema=DIRECT)
    from caltech20_rich_prototype_consistency import Alpha40MoE
    source_model=Alpha40MoE(4).eval();source_model.load_state_dict(model.state_dict(),strict=True)
    proof=json.loads((a.session/'server_bridge_gate.json').read_text());assert proof['passed'] and proof['correct']==346 and proof['PT_SHA256']==PT_SHA
    reference=a.session/'simulation_TEST_reference.npz';assert sha(reference)==proof['reference_SHA256']
    images,y,ids=datasets['test'];parts=[];source_error=0.
    for b in range(0,411,8):
        x=encode_images(images[b:b+8]);value=model(x)[0];original=source_model(x)[0];source_error=max(source_error,float((value-original).abs().max()));parts.append(value.numpy())
    assert source_error<=3e-6,'Full TEST same-host original-model/bridge parity is required'
    p=np.concatenate(parts)
    with np.load(reference,allow_pickle=False) as z:
        assert np.array_equal(ids,z['ids']) and np.array_equal(y,z['labels']);err=float(np.abs(p-z['probabilities']).max());cross_host_disagreements=int((p.argmax(1)!=z['probabilities'].argmax(1)).sum())
    assert int((p.argmax(1)==y).sum())==346
    # Exercise the exact layer-major replay for the first TRAIN sample, no SDK.
    images,_,ids=datasets['train'];images=images[:1];ids=ids[:1];expected=model(encode_images(images))[0].numpy();previous={}
    for stage in ['router','0','1','2','3']:
        def simulated(current,amplitude,phase,sample_ids):
            field=torch.from_numpy(amplitude.copy())*torch.exp(2j*np.pi*torch.from_numpy(phase.astype(np.float32)/256))[None]
            value=model.propagator(torch.nn.functional.pad(field,(20,20,20,20))).abs().square()[:,20:498,20:498].numpy();previous[current]=value;return value
        actual=layer_forward(model,images,ids,stage,dict(previous),simulated)
    replay_error=float(np.abs(actual['probabilities']-expected).max());assert replay_error<=3e-6
    result=dict(passed=True,PT_SHA256=PT_SHA,TEST=411,correct=346,accuracy=346/411,full_TEST_same_host_source_model_error=source_error,cross_host_probability_error=err,cross_host_prediction_disagreements=cross_host_disagreements,cross_host_bitwise_equivalence_claimed=False,layer_major_replay_error=replay_error,data_SHA256=dm['sha256'],TRAIN=1200,VALID=0,planned_CCD=8055,source_SHA256=sha(__file__),capture_source_SHA256=sha(HERE/'lab_caltech20_controls46_20261018.py'),SDK_opened=False,GPU_used=False)
    save(a.session/'lab_bridge_gate.json',result);print(json.dumps(result),flush=True)


def run(a):
    torch.set_num_threads(4);gateproof=json.loads((a.session/'lab_bridge_gate.json').read_text());assert gateproof['passed'] and gateproof['source_SHA256']==sha(__file__) and gateproof['capture_source_SHA256']==sha(HERE/'lab_caltech20_controls46_20261018.py')
    assert not LOCK.exists() and not (a.session/'controller.json').exists()
    datasets,dm=load_data(a.session);assert dm['sha256']==gateproof['data_SHA256']
    a.residual_datasets=datasets;a.residual_data_manifest=dm;a.dependency_kind='latest human: locked L4 residual346/411, L2 paused, no scheduled monitoring'
    save(a.session/'controller.json',dict(pid=os.getpid(),source_SHA256=sha(__file__),PT_SHA256=PT_SHA,planned_CCD=8055,TRAIN=1200,TEST=411,VALID=0,mode='fixed_optics_full_capture_then_existing_head_CPU',recurring_monitor=False))
    spec=dict(depth=4,rho=.3,alpha_schema=DIRECT,file='residual_L4.pt',sha256=PT_SHA,simulation_correct=346,simulation_total=411)
    model,data=capture_group(a,spec)
    fit=adapt_head(a.session/'L4',model,data)
    save(a.session/'report.json',dict(state='complete',capture=json.loads((a.session/'L4/capture_report.json').read_text()),readout=fit,SDK_released=not LOCK.exists(),TEST_is_development=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('gate','run'),required=True)
    for n in ('session','parent','coordinates','machine-config','phase-lut','response-lut','phase-sdk','amplitude-sdk'):p.add_argument('--'+n,type=Path,required=n=='session')
    a=p.parse_args()
    if sys.stdout is None:sys.stdout=(a.session/'controller.log').open('x',encoding='utf-8',buffering=1);sys.stderr=sys.stdout
    try:gate(a) if a.mode=='gate' else run(a)
    except BaseException as e:save(a.session/('gate_failure.json' if a.mode=='gate' else 'failure.json'),dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False));raise
