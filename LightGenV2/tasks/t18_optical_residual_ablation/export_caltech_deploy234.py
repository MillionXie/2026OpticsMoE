"""Offline joint-BMP export for frozen T18 weights; never opens devices or datasets."""
import hashlib,json,math
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from check_caltech_deploy234 import native
from caltech_deploy234_model import deployment_contract

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def brightness(gray):
    values=np.asarray(gray,dtype=np.uint8).astype(np.float64)/255
    illuminated=values[values>0]
    def describe(x):
        return dict(n=int(x.size),mean=float(x.mean()) if x.size else 0.,p99=float(np.quantile(x,.99)) if x.size else 0.,maximum=float(x.max()) if x.size else 0.)
    return dict(active=describe(values),illuminated=describe(illuminated),zero_fraction=float((values==0).mean()),saturation_fraction=float((values==1).mean()),extra_peak_normalization=False)

def encode_stage(model,amplitude,stage,weights=None):
    """Use the actual frozen simulation encoder, without propagation/model forward.

    amplitude is the 224x224 RMS-encoded input (or previous raw-CCD OEO output).
    Router weights are absent. Main stages use the frozen Top2 routing weights.
    """
    if model.raw_router.device.type!='cpu':raise ValueError('Offline BMP export is CPU only')
    if amplitude.ndim!=3 or tuple(amplitude.shape[-2:])!=(224,224):raise ValueError('Expected RMS-encoded [B,224,224] amplitude')
    if not torch.isfinite(amplitude).all() or (amplitude<0).any():raise ValueError('Amplitude must be finite and nonnegative')
    if stage!='router' and (weights is None or tuple(weights.shape)!=(len(amplitude),4)):raise ValueError('Main stage needs [B,4] frozen routing weights')
    with torch.no_grad():
        _,amp,phase=model.physical_fields(amplitude,stage,weights)
    gray=np.rint(np.clip(amp.numpy(),0,1)*255).astype(np.uint8)
    phase_gray=np.rint(phase.numpy()*256).astype(np.uint8)
    return gray,phase_gray

def save_sample_bmps(model,amplitude,stage,out,weights=None):
    """Write canonical/native joint pairs and per-sample brightness; no acquisition.

    Call in optical dependency order. Main input follows ccd_nonlinearity(raw CCD)
    between stages; NEVER normalize the BMP to its own peak.
    """
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    amps,phase=encode_stage(model,amplitude,stage,weights)
    Image.fromarray(phase,'L').save(out/'phase_canonical.bmp');native(phase,'phase').save(out/'phase_native.bmp')
    stats=[]
    for k,gray in enumerate(amps):
        Image.fromarray(gray,'L').save(out/f'amplitude_{k:06d}_canonical.bmp')
        native(gray,'amplitude').save(out/f'amplitude_{k:06d}_native.bmp')
        stats.append(dict(sample_index=k,**brightness(gray)))
    receipt=dict(stage=stage,samples=len(amps),canonical_shape=[478,478],amplitude_native_shape=[1080,1920],phase_native_shape=[1200,1920],brightness=stats,files={p.name:sha(p) for p in out.iterdir() if p.is_file()},devices_opened=0,model_forward_calls=0)
    (out/'encoding_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    return receipt

def export_selected(model,config,out,checkpoint,checkpoint_sha256,classes,sources):
    """Static selected phase/multiplier/mask package; no sample or test inference."""
    if model.raw_router.device.type!='cpu':raise ValueError('Static export is CPU only')
    assert len(classes)==20 and config['depth']==model.depth and config['residual_rho']==model.rho
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    maps={}
    for stage in ['router']+[str(k) for k in range(model.depth)]:
        with torch.no_grad():plane=model.plane(stage).cpu()[20:498,20:498]
        gray=torch.floor(torch.remainder(torch.angle(plane),2*math.pi)/(2*math.pi)*256).clamp(0,255).to(torch.uint8).numpy()
        Image.fromarray(gray,'L').save(out/f'phase_{stage}_canonical.bmp')
        native(gray,'phase').save(out/f'phase_{stage}_native.bmp')
        maps[stage]=plane.abs().numpy()
    np.savez_compressed(out/'amplitude_multipliers.npz',**maps)
    np.savez_compressed(out/'fixed_masks.npz',detectors=model.detectors.cpu().numpy(),router=model.router_masks.cpu().numpy())
    (out/'classes.json').write_text(json.dumps(dict(classes=list(classes),class_label=list(range(20))),indent=2)+'\n',encoding='utf-8')
    value=dict(contract=deployment_contract(model.depth,model.rho),checkpoint=str(Path(checkpoint).resolve()),checkpoint_sha256=checkpoint_sha256,training_config=config,training_sources=sources,export_source_sha256=sha(__file__),frozen_encoder_source_sha256=sha(Path(__file__).with_name('caltech_deploy234_model.py')),files={p.name:sha(p) for p in out.iterdir() if p.is_file()},canonical_shape=[478,478],phase_native_shape=[1200,1920],amplitude_native_shape=[1080,1920],phase_only_is_sufficient=False,amplitude_export_api='export_caltech_deploy234.save_sample_bmps: exact frozen physical_fields, round255; no peak stretch; records active and illuminated mean/p99/max/zero/saturation',hardware_sequence='router CCD -> frozen Top2 weights; main stages receive original amplitude then ccd_nonlinearity(raw linear CCD), with the same frozen routing weights',calibration_status='Offline only; fresh phase/amplitude LUT, camera ROI/orientation/exposure/gain and hardware session required before bench use',devices_opened=0,model_forward_calls=0)
    (out/'manifest.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    return value
