"""Actual frozen Qwen interface check on two fixed TRAIN inputs; never an SDK."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .scene_lab_protocol import Ledger,STAGES,array_sha,identity,put,sha
from .scene_lab_entry import train_requests
from .scene_lab_native import NativeRunner,load_head,F9,F6


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--qwen',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    import torch
    from torch.nn import functional as F
    from PIL import Image
    from .scene_paired_render import install_optical_frontend
    from .qwen3_scene_paired_train import new_qwen,raw_scene_features
    from .scene_optical_boundary import attach
    task=Path(__file__).resolve().parent;base=task/'runs/simulation';torch.set_num_threads(2)
    if a.out.exists():raise FileExistsError(a.out)
    a.out.mkdir(parents=True)
    requests=train_requests(base/'scene_natural_TRAIN608_condition_cache_20261030/TRAIN_input_manifest.json',base/'scene_two_sentence_pairs_20261029',2)
    qwen,processor=new_qwen(SimpleNamespace(qwen=a.qwen,qwen_side=256))
    info=install_optical_frontend(qwen,base/'qwen3_scene_LG_frontend_20261029',base/'qwen3_scene_LG_teacher_features_20261029',base/'scene_natural_TRAIN608_condition_cache_20261030/LG')
    if info['checkpoint_sha256']!=F6:raise ValueError('Wrong frontend')
    head,_=load_head(base/'scene_natural_single_EG_joint12_training_20261030/best_checkpoint.pt',expected_sha256=F9);head=head.cuda()
    runner=NativeRunner(qwen,processor,head,base/'scene_two_sentence_pairs_20261029',frontend_sha256=F6,head_sha256=F9)
    def simulated(stage,amp,phase,ideal=None):
        if ideal is not None:raise ValueError('No ideal fallback supplied')
        modality=stage.split('_',1)[0];prop=qwen.model.hybrid[modality].paths[0].core.propagator
        value=torch.from_numpy(amp).cuda()*torch.exp(1j*torch.from_numpy(phase).cuda())
        with torch.inference_mode():return prop(F.pad(value,(20,20,20,20))).abs().square()[:,20:498,20:498].cpu().float().numpy()
    parity=[];reference_frames={};reconstruction_sensitivity=[]
    for request in requests:
        with Image.open(base/'scene_two_sentence_pairs_20261029'/request['reference'])as im:image=im.convert('RGB').copy()
        frames={};order=[]
        def collect(stage,amplitude,phase,ideal):
            if ideal is None or stage!=STAGES[len(order)]:raise ValueError('Wrong diagnostic order')
            order.append(stage);frames[stage]=ideal.detach().cpu().float().numpy().copy();return ideal
        restore=attach(qwen.model.hybrid,collect,simulation_diagnostic=True)
        try:h,v,_=raw_scene_features(qwen,processor,image,request['prompt'],256,keep_on_device=True)
        finally:restore()
        if tuple(order)!=STAGES:raise ValueError('Incomplete diagnostic frame set')
        reference_frames[request['sample_id']]=frames
        with torch.inference_mode():expected=head(h[None].float(),v[None].float())
        exact=lambda stage,amplitude,phase,ideal=None:frames[stage].copy()
        actual=runner(request,exact)['conditions']
        errors={k:float((value.cpu()-actual[k]).abs().max())for k,value in expected.items()}
        if max(errors.values())>3e-6:raise ValueError(('Exact-frame native interface diverged',errors))
        parity.append(dict(sample_id=request['sample_id'],maxabs=errors))
        rebuilt=runner(request,simulated)['conditions']
        # Separate sensitivity diagnostic, never relax the exact interface gate.
        reconstruction_sensitivity.append(dict(sample_id=request['sample_id'],conditions={k:dict(maxabs=float((value.cpu()-rebuilt[k]).abs().max()),relative_RMSE=float(((value.cpu()-rebuilt[k]).square().mean()/value.cpu().square().mean().clamp_min(1e-20)).sqrt()))for k,value in expected.items()}))
    dark=np.zeros((1,478,478),np.float32);gain=np.ones_like(dark)
    session=json.loads((task/'scene_lab_session_template.json').read_text());session.update(calibration_approved=True,session_id='CPU_GPU_SYNTHETIC_ONLY',scope='offline_synthetic',camera='none_simulation',wavelength_nm=532,distance_m=.1,
        amplitude_pitch_um=17,phase_pitch_um=8,amplitude_native_shape=[1080,1920],phase_native_shape=[1200,1920],orientation='synthetic_canonical',exposure_us=1,gain='synthetic',wait_ms=0,intensity_units='simulation')
    for key in session:
        if key.endswith('_sha256'):session[key]='a'*64
    session.update(dark_sha256=array_sha(dark),gain_map_sha256=array_sha(gain))
    import subprocess
    pin=subprocess.check_output(['git','rev-parse','HEAD'],cwd=task,text=True).strip()
    ledger=Ledger(a.out/'synthetic_ledger',session,dict(F9=F9,F6=F6,source_commit=pin),requests,create=True)
    for stage in STAGES:
        pending=ledger.prepare(stage,runner)
        for request in requests:
            folder=ledger.folder(stage,request);ticket=json.loads((folder/'ticket.json').read_text())
            # These are labelled synthetic fixtures from the same original graph.
            # Once each prefix fixture is replayed exactly, the next input must
            # reproduce that graph's input; Ledger checks its complete identity.
            raw=reference_frames[request['sample_id']][stage].copy()
            meta=dict(ticket_sha256=identity(ticket),session_sha256=identity(session),stage=stage,request_sha256=identity(request),canonical_orientation=session['orientation'],exposure_us=1,gain='synthetic',wait_ms=0,
                frame_provenance='synthetic_fixture',native_frame_sha256=array_sha(raw),canonical_raw_sha256=array_sha(raw),capture_UTC='2026-10-30T00:00:00Z')
            ledger.accept(stage,request,raw,meta,dark,gain)
        put(a.out/(stage+'_complete.json'),dict(requests=len(pending),synthetic_only=True,physical_CCD=False))
    replays=[]
    for request in requests:
        full=ledger.infer(request,runner)['conditions']
        exact=lambda stage,amplitude,phase,ideal=None:reference_frames[request['sample_id']][stage].copy()
        original=runner(request,exact)['conditions']
        errors={k:float((full[k]-original[k]).abs().max())for k in full}
        if max(errors.values())>3e-6:raise ValueError(('Measured-prefix orchestration diverged',errors))
        replays.append(dict(sample_id=request['sample_id'],maxabs=errors))
    put(a.out/'report.json',dict(complete=True,source_commit=pin,F9=F9,F6=F6,n=2,TRAIN_only=True,TEST_used=False,SDK_used=False,physical_CCD=False,native_parity=parity,full_synthetic_prefix_replay=replays,reconstructed_continuous_field_sensitivity=reconstruction_sensitivity,
        stage_order=STAGES,partial_and_complete_Qwen_attempts=runner.native_forward_attempts,completed_adapter_Qwen_forwards=runner.completed_forwards,
        note='Real frozen Qwen/F9 head, simulated continuous CCD only. No SDXL generation or hardware5% quality claim. Synthetic nominal geometry is not site calibration.'))
    print(json.dumps(dict(passed=True,TRAIN=2,native_parity=parity,prefix_replay=replays,SDK=False,physical_CCD=False)))


if __name__=='__main__':main()
