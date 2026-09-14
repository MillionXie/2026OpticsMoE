"""Pinned LGVQ models, with measured CCD substitution in the ORIGINAL forward.

No copied electronic forward, no automatic phase switching, no hardware imports.
The propagation boundary is intercepted before abs-square; all existing routing,
normalization, scalar gates and the exact trained readout remain in the model.
"""
from __future__ import annotations
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import types
import numpy as np

STAGES = ('vision_router','vision_expert','vision_global','language_router','language_expert','language_global')
PINS = {
 'temporal': {'run':'multivideo16x4_rank_s163','sha256':'5303b574b200e14bf943af21c60a246720be453b9cf8847cd93c0eaaa243a77c','srcc':0.8044,'videos_per_field':16},
 'spatial': {'run':'spatial_readout_1m_srcc067','sha256':'95e12397ccf8c960fa30ba9dfb400b69d2c2ebd02288ac6a4879870ab592828b','srcc':0.6710079009479295,'videos_per_field':1},
}

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
 tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str,allow_nan=False),encoding='utf-8');tmp.replace(p)

def restore_settings(target, values):
 if target=='temporal':
  from .multivideo_settings import MultiVideoSettings as Settings, MultiVideoGeometry as Geometry
 else:
  from experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.settings import ExperimentSettings as Settings, Geometry
 known={f.name:f for f in dataclasses.fields(Settings)}
 kw={k:v for k,v in values.items() if k in known}
 kw['geometry']=Geometry(**values['geometry'])
 for k,v in list(kw.items()):
  if k=='geometry':continue
  if 'Path' in str(known[k].type) and v is not None:
   kw[k]=tuple(Path(x) for x in v) if isinstance(v,list) else Path(v)
 return Settings(**kw)

def load_model(target,checkpoint,device='cpu'):
 import torch
 if sha(checkpoint)!=PINS[target]['sha256']:raise ValueError('Wrong checkpoint: target SHA256 does not match the requested formal version')
 saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
 settings=restore_settings(target,saved['settings'])
 if settings.target_name!=target or saved['architecture']!=settings.architecture_label:raise ValueError('Model/target/architecture mismatch')
 if target=='temporal':
  from .models.multivideo9x4 import build_model
 else:
  from experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.modeling import build_model
 model=build_model(settings);model.load_state_dict(saved['state_dict'],strict=True)
 model.eval().to(device)
 if getattr(settings,'serial_router_flatfield_calibration',False):raise ValueError('Extra optical flat-field reference is not implemented by this six-pass contract')
 return model,settings

def phase_planes(model,target):
 """Unflipped physical learned phase, not arg(coherent leakage + modulation)."""
 import torch
 g=model.settings.geometry;n=g.active_size;state=model.state_dict();planes={};supports={}
 keys=['parallel_router.raw_router_phase','parallel_optics.raw_expert_phase','parallel_optics.raw_global_phase','serial_router.raw_router_phase','serial_optics.raw_expert_phase','serial_optics.raw_global_phase']
 for stage,key in zip(STAGES,keys):
  raw=state[key];ph=(2*math.pi*torch.sigmoid(raw.float())).detach().cpu().numpy()
  plane=np.zeros((n,n),np.float32);support=np.zeros((n,n),bool)
  def tile(value,y,x):
   h,w=value.shape
   if y<0 or x<0 or y+h>n or x+w>n or support[y:y+h,x:x+w].any():raise ValueError('Phase tiles outside aperture or overlapping')
   plane[y:y+h,x:x+w]=value;support[y:y+h,x:x+w]=True
  if target=='temporal':
   if stage in ('vision_router','vision_expert'):
    idx=0
    origins=(((g.frame_lane_size-g.frame_expert_size)//2,)*2,) if stage=='vision_router' else g.frame_expert_origins_local
    for vy,vx in g.video_origins:
     for fy,fx in g.frame_origins_local:
      for ey,ex in origins:tile(ph[idx],vy+fy+ey,vx+fx+ex);idx+=1
   elif stage in ('vision_global','language_global'):
    for i,(y,x) in enumerate(g.video_origins):tile(ph[i],y+g.video_phase_offset,x+g.video_phase_offset)
   elif stage=='language_router':
    for i,(y,x) in enumerate(g.video_origins):tile(ph[i],y+g.video_field_offset,x+g.video_field_offset)
   else:
    idx=0
    for y,x in g.video_origins:
     for ey,ex in g.video_expert_origins_local:tile(ph[idx],y+ey,x+ex);idx+=1
  elif stage=='vision_router':
   offset=(g.lane_size-g.parallel_expert_size)//2
   for i,(y,x) in enumerate(g.lane_origins):tile(ph[i],y+offset,x+offset)
  elif stage=='vision_expert':
   idx=0
   for y,x in g.lane_origins:
    for ey,ex in g.parallel_expert_origins:tile(ph[idx],y+ey,x+ex);idx+=1
  elif stage in ('vision_global','language_global'):tile(ph,0,0)
  elif stage=='language_router':
   off=(n-g.serial_expert_size)//2;tile(ph,off,off)
  else:
   for i,(y,x) in enumerate(g.serial_expert_origins):tile(ph[i],y,x)
  planes[stage]=plane;supports[stage]=support
 return planes,supports

class StopAtPlane(Exception):pass

class OpticalBoundary:
 """Scoped adapter; restoring methods on exit preserves the original model."""
 def __init__(self,model,target,measured=None,stop_before=None):
  self.model=model;self.target=target;self.measured=measured or {};self.stop_before=stop_before
  if tuple(self.measured)!=STAGES[:len(self.measured)]:raise ValueError('Measured CCDs must form a contiguous six-pass prefix')
  self.index=0;self.amplitudes={};self.detectors={};self.originals=[]
  self.planes,self.supports=phase_planes(model,target)
 def __enter__(self):
  import torch
  for owner in ('parallel_router','parallel_optics','serial_router','serial_optics'):
   prop=getattr(self.model,owner).propagation;original=prop.forward
   self.originals.append((prop,original))
   def forward(module,field,original=original,owner=owner):
    if self.index>=6:raise RuntimeError('Unexpected additional optical propagation')
    stage=STAGES[self.index]
    expected=('parallel_router','parallel_optics','parallel_optics','serial_router','serial_optics','serial_optics')[self.index]
    if owner!=expected:raise RuntimeError('Optical pass order changed')
    self.index+=1;g=self.model.settings.geometry;m=g.active_margin;n=g.active_size
    active=field[:,m:m+n,m:m+n]
    p=torch.as_tensor(self.planes[stage],device=field.device);support=torch.as_tensor(self.supports[stage],device=field.device)
    levels=getattr(self.model.settings,'phase_quantization_levels',0)
    if levels:p=torch.round(p/ (2*math.pi/(levels-1)))*(2*math.pi/(levels-1))
    dc=self.model.settings.unmodulated_power_fraction_eval
    modulation=math.sqrt(1-dc)*torch.exp(1j*p)+math.sqrt(dc)
    modulation=torch.where(support,modulation,torch.ones_like(modulation))
    amplitude=active/modulation
    if amplitude.imag.abs().max()>1e-4 or amplitude.real.min() < -1e-5:raise RuntimeError('Cannot recover real nonnegative SLM amplitude from exact model; phase layout mismatch')
    self.amplitudes[stage]=amplitude.real.clamp_min(0).detach()
    if stage in self.measured:
     detector=self.measured[stage].to(field.device).float()
     if detector.shape==active.shape:detector=torch.nn.functional.pad(detector,(m,m,m,m))
     if detector.shape!=field.shape or not torch.isfinite(detector).all() or detector.min()<0:raise ValueError('Invalid measured intensity shape/range')
     self.detectors[stage]=detector.detach();return detector.sqrt().to(torch.complex64)
    output=original(field);self.detectors[stage]=output.abs().square().detach()
    if stage==self.stop_before:raise StopAtPlane(stage)
    return output
   prop.forward=types.MethodType(forward,prop)
  return self
 def __exit__(self,*args):
  for prop,original in self.originals:prop.forward=original

def forward(model,batch):
 names=('vision_tokens','quality_tokens','language_tokens','language_mask','raw_frames','vgg_tokens','resnet_tokens','mobilenet_tokens')
 device=next(model.parameters()).device
 return model(**{k:v.to(device) for k,v in batch.items() if k in names},optical_enabled=True)

def replay(model,target,batch,measured=None,stop_before=None):
 import torch
 result=None
 with torch.inference_mode(),OpticalBoundary(model,target,measured,stop_before) as tap:
  try:result=forward(model,batch)
  except StopAtPlane:pass
 return result,tap
