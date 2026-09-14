"""Per-layer SHS/Holoeye CLI. No automatic phase changes or background capture."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
from .lab_runtime import PINS,STAGES,read,write,sha

def identity(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def raster(active,device,kind):
 """Resample physical pitch once, then apply explicit device orientation once."""
 a=np.asarray(active,np.float32)
 if a.shape!=(478,478):raise ValueError('Expected canonical 478x478 field')
 if kind=='phase':a=np.rint(np.mod(a,2*math.pi)/(2*math.pi)*255).astype(np.uint8)
 else:
  if not np.isfinite(a).all() or a.min()<0:raise ValueError('Amplitude must be finite and nonnegative')
  positive=a[a>0];scale=float(np.percentile(positive,99.5)) if positive.size else 0
  if scale<=0:raise ValueError('Empty amplitude')
  a=np.rint(np.clip(a/scale,0,1)*255).astype(np.uint8)
 pixels=round(478*17/float(device['pixel_pitch_um']))
 im=Image.fromarray(a).resize((pixels,pixels),Image.Resampling.NEAREST if kind=='phase' else Image.Resampling.BILINEAR)
 a=np.asarray(im)
 if device.get('flip_vertical'):a=np.flipud(a)
 if device.get('flip_horizontal'):a=np.fliplr(a)
 if kind=='phase' and device.get('gray_encoding')=='inverted_255_minus_g':a=255-a
 w,h=device.get('size_wh',device.get('expected_resolution_wh'))
 cx,cy=device['center_xy'];x=int(round(cx-pixels/2));y=int(round(cy-pixels/2))
 if x<0 or y<0 or x+pixels>w or y+pixels>h:raise ValueError('Active aperture outside physical panel')
 background=255 if kind=='phase' and device.get('gray_encoding')=='inverted_255_minus_g' else 0
 result=np.full((h,w),background,np.uint8);result[y:y+pixels,x:x+pixels]=a
 return result

def config(path):
 c=read(path)
 if c['model_active_pixels']!=478 or c['model_pitch_um']!=17 or c['distance_m']!=.1 or c['wavelength_nm']!=532:raise ValueError('Physical model contract mismatch')
 if c['camera']['exposure_us']<=0 or c['settle_delay_ms']<0:raise ValueError('Invalid timing')
 return c

def session(root,name):
 import re
 if not re.fullmatch('[A-Za-z0-9_-]{1,80}',name):raise ValueError('Invalid session name')
 return root/'sessions'/name

def open_session(a):
 root=Path(a.project).resolve();s=session(root,a.session);state=read(s/'session.json');c=config(a.config)
 release=read(root/'release.json')
 if state['hardware_sha256']!=identity(c) or state['release_sha256']!=sha(root/'release.json'):raise ValueError('Config/release changed: create a NEW session')
 if sha(root/'weights/best_checkpoint.pt')!=PINS[release['target']]['sha256']:raise ValueError('Wrong weights')
 return root,s,state,c,release

def initialize(a):
 root=Path(a.project).resolve();release=read(root/'release.json');c=config(a.config);s=session(root,a.session)
 if s.exists():raise FileExistsError('Session exists; do not overwrite')
 if release['checkpoint_sha256']!=PINS[release['target']]['sha256']:raise ValueError('Wrong release')
 if sha(root/'weights/best_checkpoint.pt')!=release['checkpoint_sha256']:raise ValueError('Weights changed')
 selected=release['fields'] if a.fields==0 else release['fields'][:a.fields]
 if not selected:raise ValueError('No fields')
 s.mkdir(parents=True)
 write(s/'session.json',dict(target=release['target'],hardware_sha256=identity(c),hardware_config=c,release_sha256=sha(root/'release.json'),fields=selected,measured_stages=[],automatic_phase_switching=False,created=time.strftime('%Y-%m-%dT%H:%M:%S')))
 for stage in STAGES:
  phase=np.load(root/'phases'/f'{stage}.npy',allow_pickle=False)
  out=s/'phase';out.mkdir(exist_ok=True);Image.fromarray(raster(phase,c['phase_slm'],'phase')).save(out/f'{STAGES.index(stage)+1:02d}_{stage}.bmp')
 print('INITIALIZED',s,flush=True)

def verified_ccd(s,stage,key):
 p=s/'ccd'/stage/(key+'.png');rec=read(p.with_suffix('.record.json'))
 if rec['sha256']!=sha(p):raise ValueError('CCD changed after capture: '+str(p))
 return p,rec

def prepare(a):
 import torch
 from .lab_runtime import load_model,replay
 root,s,state,c,release=open_session(a);idx=STAGES.index(a.stage)
 if state['measured_stages']!=list(STAGES[:idx]):raise ValueError('Previous measured stages incomplete, or requested stage already passed')
 model,settings=load_model(release['target'],root/'weights/best_checkpoint.pt',a.device)
 dest=s/'play'/a.stage;dest.mkdir(parents=True,exist_ok=True);entries=[]
 for item in state['fields']:
  source=root/item['file']
  if sha(source)!=item['sha256']:raise ValueError('Cached input modified')
  batch=torch.load(source,map_location='cpu',weights_only=False);measured={};upstream={}
  for stage in STAGES[:idx]:
   p,rec=verified_ccd(s,stage,item['key']);upstream[stage]=rec['sha256']
   intensity=np.asarray(Image.open(p),dtype=np.float32)
   gain=float(c.get('detector_intensity_scale',{}).get(stage,1/255))
   if gain<=0:raise ValueError('Detector scale must be positive')
   measured[stage]=torch.from_numpy(intensity.copy())[None]*gain
  _,tap=replay(model,release['target'],batch,measured,a.stage)
  amplitude=tap.amplitudes[a.stage][0].cpu().numpy();bmp=dest/(item['key']+'.bmp')
  Image.fromarray(raster(amplitude,c['amplitude_slm'],'amplitude')).save(bmp)
  theory=tap.detectors[a.stage][0].cpu().numpy();m=settings.geometry.active_margin;active=theory[m:m+478,m:m+478]
  theory_dir=s/'theoretical_ccd'/a.stage;theory_dir.mkdir(parents=True,exist_ok=True)
  np.save(theory_dir/(item['key']+'.npy'),active)
  peak=float(np.percentile(active,99.5));Image.fromarray(np.rint(np.clip(active/max(peak,1e-12),0,1)*255).astype(np.uint8)).save(theory_dir/(item['key']+'.png'))
  entries.append(dict(key=item['key'],bmp=bmp.name,sha256=sha(bmp),upstream_ccd_sha256=upstream,positive_amplitude_p995=float(np.percentile(amplitude[amplitude>0],99.5)),theory_preview_only_scale_p995=peak))
 phase=s/'phase'/f'{idx+1:02d}_{a.stage}.bmp'
 write(dest/'manifest.json',dict(stage=a.stage,hardware_sha256=state['hardware_sha256'],release_sha256=state['release_sha256'],phase_file=str(phase.relative_to(s)),phase_sha256=sha(phase),entries=entries))
 print('READY',a.stage,len(entries),'Load and KEEP phase:',phase,flush=True)

def capture(a):
 import cv2
 root,s,state,c,release=open_session(a);idx=STAGES.index(a.stage);mf=read(s/'play'/a.stage/'manifest.json')
 if state['measured_stages']==list(STAGES[:idx+1]):print('Stage already complete');return
 if state['measured_stages']!=list(STAGES[:idx]):raise ValueError('Non-contiguous capture')
 if mf['hardware_sha256']!=state['hardware_sha256'] or mf['release_sha256']!=state['release_sha256']:raise ValueError('Stale manifest')
 phase=s/mf['phase_file']
 if sha(phase)!=mf['phase_sha256']:raise ValueError('Phase BMP changed')
 if not a.phase_ready:raise ValueError('Load this stage phase first; then explicitly pass --phase-ready. This is NOT optical verification.')
 if not c.get('geometry_confirmed'):raise ValueError('ROI is not confirmed')
 bench=Path(a.bench_root).resolve();sys.path.insert(0,str(bench))
 from slm_camera import Controller
 # Share the existing desktop-job ownership lock; never steal another capture.
 lock=bench/'results/dual_jobs/ACTIVE.lock';lock.parent.mkdir(parents=True,exist_ok=True)
 fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
 try:
  pts=np.float32([c['logical_corners_full_sensor_xy'][k] for k in ('top_left','top_right','bottom_right','bottom_left')]);H=cv2.getPerspectiveTransform(pts,np.float32([[-.5,-.5],[477.5,-.5],[477.5,477.5],[-.5,477.5]]))
  with Controller(c) as hw:
   for i,e in enumerate(mf['entries'],1):
    p=s/'ccd'/a.stage/(e['key']+'.png');record=p.with_suffix('.record.json')
    if record.exists():
     _,old=verified_ccd(s,a.stage,e['key'])
     if old['amplitude_sha256']!=e['sha256'] or old['phase_sha256']!=mf['phase_sha256'] or old['upstream_ccd_sha256']!=e['upstream_ccd_sha256']:raise ValueError('Capture identity mismatch')
     continue
    for stage,digest in e['upstream_ccd_sha256'].items():
     _,rec=verified_ccd(s,stage,e['key'])
     if rec['sha256']!=digest:raise ValueError('Upstream CCD changed; regenerate this stage')
    bmp=s/'play'/a.stage/e['bmp']
    if sha(bmp)!=e['sha256']:raise ValueError('Amplitude BMP changed')
    raw,meta=hw.capture(bmp);im=np.rint(np.clip(cv2.warpPerspective(raw.astype(np.float32),H,(478,478)),0,255)).astype(np.uint8)
    q=dict(p99=float(np.percentile(im,99)),std=float(im.std()),saturation=float((im==255).mean()))
    if (q['p99']<=8 and q['std']<1.5) or q['saturation']>.01:
     failure=s/'rejected'/a.stage;failure.mkdir(parents=True,exist_ok=True);Image.fromarray(im).save(failure/(e['key']+'.png'));write(failure/(e['key']+'.json'),dict(quality=q,camera=meta))
     raise RuntimeError('Near-dark/saturated CCD rejected; previous valid samples retained. '+str(q))
    p.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(im).save(p,compress_level=1)
    write(record,dict(sha256=sha(p),amplitude_sha256=e['sha256'],phase_sha256=mf['phase_sha256'],upstream_ccd_sha256=e['upstream_ccd_sha256'],hardware_sha256=state['hardware_sha256'],quality=q,camera=meta,phase_confirmation='explicit_flag_not_optical_verification',raw_saved=False))
    write(s/'status.json',dict(status='capturing',stage=a.stage,completed=i,total=len(mf['entries']),quality=q));print('Captured',a.stage,i,'/',len(mf['entries']),q,flush=True)
  state['measured_stages']=list(STAGES[:idx+1]);write(s/'session.json',state)
  write(s/'status.json',dict(status='stage_complete_wait_for_user_phase_change',stage=a.stage,completed=len(mf['entries'])))
 finally:lock.unlink(missing_ok=True)

def evaluate(a):
 import torch
 from .lab_runtime import load_model,replay
 from experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.metrics import regression_metrics
 root,s,state,c,release=open_session(a)
 if state['measured_stages']!=list(STAGES):raise ValueError('All SIX measured CCD stages required; no simulated fallback')
 model,_=load_model(release['target'],root/'weights/best_checkpoint.pt',a.device);pred=[];labels=[];rows=[]
 for item in state['fields']:
  batch=torch.load(root/item['file'],map_location='cpu',weights_only=False);measure={}
  for stage in STAGES:
   p,_=verified_ccd(s,stage,item['key']);measure[stage]=torch.from_numpy(np.array(Image.open(p),dtype=np.float32))[None]*float(c.get('detector_intensity_scale',{}).get(stage,1/255))
  result,_=replay(model,release['target'],batch,measure);scores=result['prediction'].detach().cpu().reshape(-1).tolist()
  for slot,(score,valid) in enumerate(zip(scores,item['valid'])):
   if valid:rows.append(dict(field=item['key'],slot=slot,video=item['sample_ids'][slot],prediction=score,target=item['targets'][slot]));pred.append(score);labels.append(item['targets'][slot])
 metrics=regression_metrics(torch.tensor(pred),torch.tensor(labels),release['target']) if len(labels)>2 else None
 write(s/'results.json',dict(status='real_six_pass_evaluation',target=release['target'],metrics=metrics,count=len(labels),full_test=len(labels)==558,rows=rows,simulation_reference=PINS[release['target']]['srcc']))
 print('RESULT',s/'results.json',metrics,flush=True)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['init','prepare','capture','evaluate'])
 p.add_argument('--project',default='.');p.add_argument('--config',default='LAB.local.json');p.add_argument('--session',required=True)
 p.add_argument('--stage',choices=STAGES);p.add_argument('--fields',type=int,default=4,help='0=all packaged fields; temporal field=16 videos, spatial field=1 video')
 p.add_argument('--device',default='cuda');p.add_argument('--bench-root',default='../ABO_Lab_SHS_8um');p.add_argument('--phase-ready',action='store_true')
 a=p.parse_args()
 if a.action in ('prepare','capture') and not a.stage:p.error('--stage is required')
 if a.fields<0:p.error('--fields must be nonnegative')
 globals()[{'init':'initialize'}.get(a.action,a.action)](a)

if __name__=='__main__':main()
