"""One TRAIN-only finite-extinction hypothesis for the observation surrogate.

The old approximation predicts identically zero for zero amplitude, whereas
the unchanged bench records a substantial zero-input signal. A coherent input
floor is ONE possible missing effect, not an identified cause or a new LUT.
Fit216 old TRAIN responses only. Admit only on BOTH old other-mask TRAIN sets
AND40 new original-PT TRAIN responses after the reported ambient change.
No classifier, mask, camera settings or deployment graph is modified here.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

OLD_DATA='c806c13cec4f3641952e71c18d17e010410690e80cf749a21d777537301a979d'
OLD_PARAMS='d7463e748133e570fd953fa43f430212f7b7e8ab6cf8d8671a9b380df84bc2f5'
UUID='GPU-6dcca91a-8e08-1a50-9aa6-81defeaed50b'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,r):Path(p).write_text(json.dumps(r,indent=2),encoding='utf-8')


def finite_extinction_response(response,amplitude,turns,transfer,raw,floor):
    """Training-only effective field, zero..1 floor ratio; white remains one."""
    return response((amplitude+floor)/(1+floor),turns,transfer,raw)


def pack(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    import lab_caltech20_dvp_layerwise_20261019 as m
    from lab_caltech20_dvp_closedloop_20261021 import SPECS,HARDWARE
    torch.set_num_threads(2);assert not a.out.exists();a.out.mkdir()
    old=a.sequence/'TRAIN_coherent_response_surrogate_20261021'
    assert sha(old/'observations.npz')==OLD_DATA
    old_manifest=m.read(old/'manifest.json');assert old_manifest['SHA256']==OLD_DATA
    fit_ids={v['sample_id'] for v in old_manifest['rows'] if v['role']=='fit_TRAIN'}
    old_fit=old/'fit_GPU2_ownership_recovery_20261021'
    assert m.read(old_fit/'report.json')['state']=='complete' and sha(old_fit/'best_response.npz')==OLD_PARAMS
    assert m.session_hardware_digest(a.sequence)==HARDWARE
    data,dm=m.data_for(a.sequence);cap=a.sequence/'TRAIN_nominal_control_20261021/L6/capture'
    cp=m.read(cap/'protocol.json');assert cp['PT_SHA256']==SPECS[6]['PT'] and cp['data_SHA256']==dm['sha256']
    session=a.sequence/'L6_response_20261020';pt=session/'fits'/SPECS[6]['fit']/'best.pt'
    model,_=m.model_for(pt,SPECS[6]['PT'],6,session/'source_reference.ts')
    turns=torch.remainder(torch.angle(model.plane('0')),2*torch.pi)/(2*torch.pi)
    phase=torch.floor(turns*256).clamp(0,255)[20:498,20:498].detach().numpy().astype(np.uint8)
    assert np.array_equal(m.phase_native(phase),np.asarray(Image.open(cap/'phase.bmp')))
    cfg=m.read(a.sequence/'calibration/admission/admitted_hardware.json')
    warp=cv2.getPerspectiveTransform(np.float32(m.corners(Path(cfg['coordinates']))),np.float32([[0,0],[477,0],[477,477],[0,477]]))
    amplitudes=[];observations=[];records=[]
    for i in range(20,60):
        stem=cap/'train'/f'{i:06d}';row=m.read(stem.with_suffix('.json'))
        canonical=stem.with_name(stem.name+'_input.png');raw=stem.with_name(stem.name+'_raw.png');crop=stem.with_suffix('.png')
        assert row['role']=='train' and row['index']==i and row['sample_id']==str(data['train'][2][i])
        assert row['PT_SHA256']==SPECS[6]['PT'] and row['phase_SHA256']==sha(cap/'phase.bmp')
        assert sha(canonical)==row['canonical_SHA256'] and sha(raw)==row['raw_SHA256']
        meta=row['frame_metadata'];assert abs(meta['exposure_us']-60006)<2 and meta['gain']==1
        assert meta['shape']==[3648,5480] and not any(meta[k] for k in ('flip_h','flip_v','rotate','opposite'))
        assert row['sample_id'] not in fit_ids
        reconstructed=m.orient(cv2.warpPerspective(np.asarray(Image.open(raw)),warp,(478,478),flags=cv2.INTER_AREA,borderMode=cv2.BORDER_CONSTANT,borderValue=0),cfg['orientation'])
        observed=np.asarray(Image.open(crop));assert np.array_equal(reconstructed,observed)
        amplitudes.append(np.asarray(Image.open(canonical)).copy());observations.append(observed.copy())
        records.append(dict(index=i,sample_id=row['sample_id'],receipt_SHA256=sha(stem.with_suffix('.json')),raw_SHA256=sha(raw),crop_SHA256=sha(crop),canonical_SHA256=sha(canonical)))
    light=a.sequence/'TRAIN_fullbatch_adjoint_sign_20261021/L6_USB_recovery_20261021/TRAIN_light_preflight'
    zero=[]
    for name in ('background_0','background_1'):
        record=m.read(light/(name+'.json'));path=light/(name+'.png')
        assert sha(path)==record['crop_SHA256'] and record['phase_SHA256']==sha(cap/'phase.bmp')
        zero.append(np.asarray(Image.open(path)).copy())
    path=a.out/'current_nominal_TRAIN40.npz'
    np.savez_compressed(path,amplitude=np.stack(amplitudes),phase=np.broadcast_to(phase,(40,478,478)).copy(),observed=np.stack(observations),zero_observed=np.stack(zero))
    manifest=dict(schema='TRAIN_finite_extinction_observation_v1',source_git=m.git_head(),source_SHA256=sha(__file__),file=path.name,SHA256=sha(path),bytes=path.stat().st_size,
        old_observations_SHA256=OLD_DATA,old_parameters_SHA256=OLD_PARAMS,old_fit_records=216,old_fit_distinct_TRAIN_ids=len(fit_ids),old_other_records=80,new_other_TRAIN_records=40,new_TRAIN_ids=[v['sample_id'] for v in records],records=records,
        all_other_response_records=120,all_other_distinct_TRAIN_ids=len({v['sample_id'] for v in old_manifest['rows'] if v['role']!='fit_TRAIN'}|{v['sample_id'] for v in records}),
        data_SHA256=dm['sha256'],hardware_SHA256=HARDWARE,nominal_PT_SHA256=SPECS[6]['PT'],nominal_phase_SHA256=sha(cap/'phase.bmp'),capture_protocol_SHA256=sha(cap/'protocol.json'),
        individual_finished_frames_verified=True,whole_capture_not_yet_required=True,TEST_used=False,VALID=0,labels_used=False,SDK_used=False,new_device_call=False,
        zero_frames_not_fit=True,light_change_not_unique_cause=True)
    save(a.out/'manifest.json',manifest);print(json.dumps({k:v for k,v in manifest.items() if k not in ('records','new_TRAIN_ids')}),flush=True)


def selftest():
    import torch
    import lab_caltech20_dvp_coherent_response_20261021 as old
    torch.set_num_threads(2);torch.manual_seed(73)
    assert sha(old.__file__)=='c08da807054fb5716bd6c36923218ee192870cb8e1959f6d7732c8f6d9155cc6'
    amp=torch.rand(2,478,478,dtype=torch.float64);turns=torch.rand_like(amp)
    f=torch.fft.fftfreq(518,dtype=torch.float64);h=torch.exp(60j*(f[:,None].square()+f[None,:].square()))
    raw=old.initial('cpu').double()
    with torch.no_grad():raw['sigma'].copy_(torch.logit(torch.tensor(.2/12,dtype=torch.float64)))
    actual=finite_extinction_response(old.response,amp,turns,h,raw,0.)
    assert torch.equal(actual,old.response(amp,turns,h,raw))
    zero=torch.zeros_like(amp)
    assert float(old.response(zero,turns,h,raw).max())==0
    assert float(finite_extinction_response(old.response,zero,turns,h,raw,.2).max())>0
    u=torch.tensor(-1.38629436112,dtype=torch.float64,requires_grad=True)
    target=old.response(amp,turns,h,raw).detach()
    loss=old.normalized_loss(finite_extinction_response(old.response,amp,turns,h,raw,u.sigmoid()),target)
    gradient=float(torch.autograd.grad(loss,u)[0]);step=.001
    with torch.no_grad():finite=float((old.normalized_loss(finite_extinction_response(old.response,amp,turns,h,raw,(u+step).sigmoid()),target)-old.normalized_loss(finite_extinction_response(old.response,amp,turns,h,raw,(u-step).sigmoid()),target))/(2*step))
    assert abs(gradient)>1e-6 and abs(finite-gradient)<max(1e-7,.005*abs(gradient))
    print(json.dumps(dict(passed=True,zero_floor_exact_old_function=True,old_zero_input_maximum=0.,nonzero_floor_nonzero_field=True,floor_derivative=gradient,finite_difference=finite,SDK_used=False)),flush=True)


def fit(a):
    import subprocess
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==UUID
    import numpy as np
    import torch
    from torch.nn import functional as F
    import lab_caltech20_dvp_coherent_response_20261021 as old
    assert sha(old.__file__)=='c08da807054fb5716bd6c36923218ee192870cb8e1959f6d7732c8f6d9155cc6'
    torch.set_num_threads(2);torch.manual_seed(73)
    assert not a.out.exists() and torch.cuda.is_available() and 'A100' not in torch.cuda.get_device_name(0)
    owners=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    assert all(v.split(',')[0].strip()!=UUID or int(v.split(',')[1])==os.getpid() for v in owners.splitlines())
    torch.cuda.set_per_process_memory_fraction(.6,0)
    manifest=json.loads(a.manifest.read_text());path=a.manifest.parent/manifest['file'];assert sha(path)==manifest['SHA256'] and not manifest['TEST_used'] and not manifest['labels_used']
    oldroot=a.manifest.parent.parent/'TRAIN_coherent_response_surrogate_20261021'
    legacy=oldroot/'observations.npz';params=oldroot/'fit_GPU2_ownership_recovery_20261021/best_response.npz'
    assert sha(legacy)==OLD_DATA and sha(params)==OLD_PARAMS
    with np.load(legacy) as z:
        aa=z['amplitude'].copy();pp=z['phase'].copy();oo=z['observed'].copy();mask=z['fit'].copy();depth=z['depth'].copy();h=torch.from_numpy(z['transfer'].copy()).cuda()
    with np.load(path) as z:
        aa=np.concatenate([aa,z['amplitude']]);pp=np.concatenate([pp,z['phase']]);oo=np.concatenate([oo,z['observed']]);zero_observed=z['zero_observed'].copy()
    amp=torch.from_numpy(aa.astype(np.float32)/255).cuda();phase=torch.from_numpy(pp.astype(np.float32)/256).cuda();obs=torch.from_numpy(oo.astype(np.float32)).cuda()
    train=np.flatnonzero(mask);checks={'old_L4_TRAIN':np.flatnonzero((~mask)&(depth==4)),'old_L6_TRAIN':np.flatnonzero((~mask)&(depth==6)),'current_L6_TRAIN':np.arange(296,336)}
    assert len(train)==216 and all(len(v)==40 for v in checks.values())
    raw=old.initial('cuda')
    with np.load(params) as z:raw.load_state_dict({k:torch.from_numpy(z[k].copy()).cuda() for k in z.files})
    u=torch.nn.Parameter(torch.tensor(-1.38629436112,device='cuda'))
    def evaluate(indices,floor):
        losses=[];pixel=[];coarse=[]
        with torch.no_grad():
            for begin in range(0,len(indices),4):
                j=indices[begin:begin+4];pred=finite_extinction_response(old.response,amp[j],phase[j],h,raw,floor);observed=obs[j]
                losses.append(float(old.normalized_loss(pred,observed))*len(j))
                for dest,x,y in ((pixel,pred,observed),(coarse,F.adaptive_avg_pool2d(pred[:,None],(16,16))[:,0],F.adaptive_avg_pool2d(observed[:,None],(16,16))[:,0])):
                    x=x.flatten(1);y=y.flatten(1);x=x-x.mean(1,keepdim=True);y=y-y.mean(1,keepdim=True)
                    dest.extend(((x*y).sum(1)/(x.norm(dim=1)*y.norm(dim=1)).clamp_min(1e-12)).cpu().tolist())
        return dict(loss=sum(losses)/len(indices),pixel_PCC=float(np.median(pixel)),coarse_PCC=float(np.median(coarse)))
    baseline={k:evaluate(v,0.) for k,v in checks.items()};best=evaluate(train,0.)['loss'];best_state={k:v.detach().cpu().clone() for k,v in raw.items()};best_floor=0.;selected=0
    a.out.mkdir();protocol=dict(source_git=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_SHA256=sha(__file__),base_response_source_SHA256=sha(old.__file__),manifest_SHA256=sha(a.manifest),old_parameters_SHA256=OLD_PARAMS,TRAIN_fit_records=216,fit_distinct_TRAIN_ids=manifest['old_fit_distinct_TRAIN_ids'],other_TRAIN_records=120,other_distinct_TRAIN_ids=manifest['all_other_distinct_TRAIN_ids'],labels_used=False,TEST_used=False,VALID=0,maximum_full_updates=30,maximum_fit_seconds=240,microbatch=4,original_response_LR=.02,floor_LR=.1,floor_initial=.2,floor_bound=[0.,1.],initialization_is_hypothesis_not_measured_calibration=True,
        environment=dict(Torch=str(torch.__version__),NumPy=np.__version__,CPU_threads=torch.get_num_threads(),GPU_UUID=UUID,GPU_name=torch.cuda.get_device_name(0),GPU_fraction=.6),
        deployed_amplitude_BMP_unchanged=True,device_LUT_exposure_coordinates_unchanged=True,new_classifier_PT=False,new_mask=False,one_physics_hypothesis_not_parameter_grid=True,all_checks_require_loss_ratio=.8,minimum_pixel_PCC=.3,minimum_coarse_PCC=.45,minimum_each_PCC_gain=.05)
    save(a.out/'protocol.json',protocol)
    optimizer=torch.optim.Adam([dict(params=list(raw.parameters()),lr=.02),dict(params=[u],lr=.1)])
    history=[];start=time.monotonic()
    for step in range(1,31):
        optimizer.zero_grad(set_to_none=True);value=0.
        for begin in range(0,216,4):
            ix=train[begin:begin+4];loss=old.normalized_loss(finite_extinction_response(old.response,amp[ix],phase[ix],h,raw,u.sigmoid()),obs[ix]);(loss*len(ix)/216).backward();value+=float(loss.detach())*len(ix)/216
        if value<best:best=value;best_state={k:v.detach().cpu().clone() for k,v in raw.items()};best_floor=float(u.sigmoid().detach());selected=step
        penalty=.002*(raw['mag'].square().mean()+raw['amplitude_phase'].square().mean());penalty.backward()
        assert torch.isfinite(u.grad) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in raw.values())
        torch.nn.utils.clip_grad_norm_(list(raw.parameters())+[u],2.);optimizer.step()
        row=dict(step=step,TRAIN_response_records=216,TRAIN_loss=value,floor=float(u.sigmoid().detach()),elapsed_s=time.monotonic()-start)
        history.append(row);save(a.out/'history.json',history);save(a.out/'progress.json',dict(state='TRAIN_only_finite_extinction_fit',latest=row));print(json.dumps(row),flush=True)
        if time.monotonic()-start>=240:break
    last_floor=float(u.sigmoid().detach());final=evaluate(train,last_floor)['loss']
    if final<best:best=final;best_state={k:v.detach().cpu().clone() for k,v in raw.items()};best_floor=last_floor;selected=step
    np.savez_compressed(a.out/'last_response.npz',**{k:v.detach().cpu().numpy() for k,v in raw.items()},floor_effective=np.array(last_floor))
    raw.load_state_dict(best_state);results={k:evaluate(v,best_floor) for k,v in checks.items()}
    admitted=all(results[k]['loss']<=.8*baseline[k]['loss'] and results[k]['pixel_PCC']>=max(.3,baseline[k]['pixel_PCC']+.05) and results[k]['coarse_PCC']>=max(.45,baseline[k]['coarse_PCC']+.05) for k in checks)
    np.savez_compressed(a.out/'best_response.npz',**{k:v.detach().cpu().numpy() for k,v in raw.items()},floor_effective=np.array(best_floor))
    save(a.out/'report.json',dict(state='complete',actual_updates=step,selected_step=selected,selected_e0=selected==0,TRAIN_best_loss=best,baseline=baseline,fitted=results,selected_floor=best_floor,response_admitted=admitted,zero_observed_mean=float(zero_observed.mean()),zero_frames_fit=False,zero_floor_old_response_identically_zero=True,finite_floor_is_not_an_identified_stray_light_source=True,
        new_classifier_or_mask_generated=False,device_or_deployment_graph_modified=False,TEST_used=False,VALID=0,labels_used=False,SDK_used=False,protocol_SHA256=sha(a.out/'protocol.json'),response_SHA256=sha(a.out/'best_response.npz')))
    print(json.dumps(dict(complete=True,response_admitted=admitted,baseline=baseline,fitted=results,selected_floor=best_floor,new_mask=False)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('pack','fit','selftest'),required=True)
    p.add_argument('--sequence',type=Path);p.add_argument('--manifest',type=Path);p.add_argument('--out',type=Path)
    a=p.parse_args();selftest() if a.mode=='selftest' else pack(a) if a.mode=='pack' else fit(a)
