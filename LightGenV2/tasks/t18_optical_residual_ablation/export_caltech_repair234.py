"""Matched offline export for RGB/dense4/preserve478; no hardware or test access."""
import argparse,json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import numpy as np
import torch
from PIL import Image
from caltech_repair234_model import CorrectedMoE,deployment_contract
from export_caltech_deploy234 import sha,brightness
from check_caltech_deploy234 import native
import caltech_repair234 as r

def raw_ccd(value):
    if not torch.is_tensor(value) or value.device.type!='cpu' or value.ndim!=3 or tuple(value.shape[-2:])!=(478,478):raise ValueError('Expected raw linear CPU CCD[B,478,478]')
    if not torch.isfinite(value).all() or (value<0).any():raise ValueError('Raw CCD must be finite and nonnegative')
    return value

def routing_from_ccd(model,value):
    value=raw_ccd(value);energy=torch.einsum('bhw,khw->bk',value,model.router_masks)
    centered=energy-energy.mean(1,keepdim=True);dense=torch.softmax(centered/(centered.square().mean(1,keepdim=True)+1e-8).sqrt()/2,1)
    return dense/dense.square().sum(1,keepdim=True).sqrt().clamp_min(1e-8)

def amplitude_from_ccd(model,value,next_stage):
    if next_stage not in ('expert','global'):raise ValueError('Unknown next optical stage')
    return model.nonlinear(raw_ccd(value),next_stage)

def encode_stage(model,amplitude,stage,weights=None):
    if not isinstance(model,CorrectedMoE) or model.raw_router.device.type!='cpu':raise ValueError('Corrected CPU graph required')
    if stage=='router':expected=224
    else:
        if stage not in [str(k) for k in range(model.depth)]:raise ValueError('Unknown stage')
        expected=478 if model.stages[int(stage)]=='global' else 224
    if amplitude.device.type!='cpu' or amplitude.ndim!=3 or tuple(amplitude.shape[-2:])!=(expected,expected):raise ValueError(f'Expected CPU amplitude[B,{expected},{expected}] for {stage}')
    if not torch.isfinite(amplitude).all() or (amplitude<0).any():raise ValueError('Amplitude must be finite and nonnegative')
    if stage=='router':
        if weights is not None:raise ValueError('Router does not receive routing weights')
    elif model.stages[int(stage)]=='expert':
        if weights is None or weights.device.type!='cpu' or tuple(weights.shape)!=(len(amplitude),4):raise ValueError('Expert requires CPU dense4 weights[B,4]')
        if not torch.isfinite(weights).all() or (weights<=0).any() or not torch.allclose(weights.square().sum(1),torch.ones(len(amplitude)),atol=1e-5,rtol=0):raise ValueError('Dense4 finite positive L2-normalized amplitude weights required')
    with torch.no_grad():_,amp,phase=model.physical_fields(amplitude,stage,weights)
    return np.rint(amp.numpy().clip(0,1)*255).astype(np.uint8),np.rint(phase.numpy()*256).astype(np.uint8)

def save_sample_bmps(model,amplitude,stage,out,weights=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);amps,phase=encode_stage(model,amplitude,stage,weights)
    Image.fromarray(phase).save(out/'phase_canonical.bmp');native(phase,'phase').save(out/'phase_native.bmp');stats=[]
    for k,gray in enumerate(amps):
        Image.fromarray(gray).save(out/f'amplitude_{k:06d}_canonical.bmp');native(gray,'amplitude').save(out/f'amplitude_{k:06d}_native.bmp');stats.append(dict(sample_index=k,**brightness(gray)))
    receipt=dict(stage=stage,samples=len(amps),input_shape=list(amplitude.shape[-2:]),canonical_shape=[478,478],amplitude_native_shape=[1080,1920],phase_native_shape=[1200,1920],brightness=stats,files={p.name:sha(p) for p in out.iterdir() if p.is_file()},export_source_sha256=sha(__file__),devices_opened=0,model_forward_calls=0,test_evaluations=0)
    r.save(out/'encoding_receipt.json',receipt,exclusive=True);return receipt

def export_selected(model,config,out,checkpoint,checkpoint_sha256,classes,sources):
    if not isinstance(model,CorrectedMoE) or model.raw_router.device.type!='cpu':raise ValueError('Corrected CPU graph required')
    assert len(classes)==20 and len(set(classes))==20 and config['depth']==model.depth and config['residual_rho']==model.rho
    assert sha(checkpoint)==checkpoint_sha256 and all(torch.isfinite(v).all() for v in model.state_dict().values())
    out=Path(out);out.mkdir(parents=True,exist_ok=False);maps={};stages=[]
    for stage in ['router']+[str(k) for k in range(model.depth)]:
        with torch.no_grad():plane=model.plane(stage).cpu()[20:498,20:498]
        gray=torch.floor(torch.remainder(torch.angle(plane),2*math.pi)/(2*math.pi)*256).clamp(0,255).to(torch.uint8).numpy()
        Image.fromarray(gray).save(out/f'phase_{stage}_canonical.bmp');native(gray,'phase').save(out/f'phase_{stage}_native.bmp');maps[stage]=plane.abs().numpy()
        kind='router' if stage=='router' else model.stages[int(stage)];stages.append(dict(stage=stage,kind=kind,input_size=478 if kind=='global' else 224))
    np.savez_compressed(out/'amplitude_multipliers.npz',**maps);np.savez_compressed(out/'fixed_masks.npz',detectors=model.detectors.numpy(),router=model.router_masks.numpy())
    r.save(out/'classes.json',dict(classes=list(classes),class_label=list(range(20))),exclusive=True)
    value=dict(contract=deployment_contract(model.depth,model.rho),checkpoint=str(Path(checkpoint).resolve()),checkpoint_sha256=checkpoint_sha256,selected_parameter_sha256=r.digest(model.state_dict()),training_config=config,training_sources=sources,export_source_sha256=sha(__file__),frozen_encoder_source_sha256=sha(Path(__file__).with_name('caltech_repair234_model.py')),stages=stages,files={p.name:sha(p) for p in out.iterdir() if p.is_file()},canonical_shape=[478,478],phase_native_shape=[1200,1920],amplitude_native_shape=[1080,1920],phase_only_is_sufficient=False,amplitude_export_api='export_caltech_repair234.save_sample_bmps; actual frozen physical_fields, no peak stretch',hardware_sequence='Whole dataset in optical dependency order: fixed RGB input ->router CCD ->continuous4 routing retained; expert CCD ->global478 or expert224 via exact amplitude_from_ccd; joint amplitude+phase each stage; final20 detector energies normalized',calibration_status='Offline only; new LUT/ROI/orientation/exposure/gain and raw-linear CCD scale/session needed before bench use',devices_opened=0,model_forward_calls=0,test_arrays_read=0,test_evaluations=0)
    r.save(out/'manifest.json',value,exclusive=True);return value

def selected_study(base,out):
    proof=r.read(base/'cpu_report/audit.json');assert proof['passed'] and proof['chosen_profile_case'] is not None
    meta=r.read(base/'metadata.json');assert not (Path('/proc')/str(meta['pid'])).exists()
    assert all(not (Path('/proc')/str(j['pid'])).exists() for j in r.read(base/'status.json')['completed'])
    key=proof['chosen_profile_case'];fit=proof['fits'][key];ck=base/key/'best_checkpoint.pt';assert sha(ck)==fit['checkpoint_sha256'];r.engine.verify_sources(fit['sources'],meta['git_commit']);state=torch.load(ck,map_location='cpu',weights_only=False);assert state['config']==fit['config'] and state['epoch']==fit['selected_epoch']
    model=CorrectedMoE(fit['depth'],fit['rho']).eval();model.load_state_dict(state['model'],strict=True)
    return export_selected(model,fit['config'],out,ck,fit['checkpoint_sha256'],fit['config']['classes'],fit['sources'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    assert not a.out.exists()
    try:selected_study(a.base,a.out)
    except BaseException as e:
        if a.out.exists():r.save(a.out/'failure.json',dict(error=repr(e),automatic_retry=False))
        raise
