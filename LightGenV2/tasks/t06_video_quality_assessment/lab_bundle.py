"""Build pinned, offline, field-sharded LGVQ handoffs without touching devices."""
from __future__ import annotations
import argparse
import dataclasses
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
import numpy as np
import torch
from .lab_runtime import PINS,STAGES,load_model,phase_planes,forward,replay,read,write,sha

def build(a):
 root=Path(__file__).resolve().parents[3];out=Path(a.output).resolve();source=Path(a.source_root).resolve()
 checkpoint=source/'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation'/PINS[a.target]['run']/'best_checkpoint.pt'
 model,settings=load_model(a.target,checkpoint,a.device)
 # Evaluation uses only the canonical view, never training augmentations/teachers.
 updates={k:v for k,v in dict(vision_cache_view_paths=(),quality_feature_cache_view_paths=(),raw_frame_cache_view_paths=(),training_soft_targets_path=None).items() if k in {f.name for f in dataclasses.fields(settings)}}
 settings=dataclasses.replace(settings,**updates)
 from experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.data import load_single_metric_cache
 payload=load_single_metric_cache(settings)
 if out.exists():raise FileExistsError('Use a new release output, not overwrite')
 out.mkdir(parents=True);(out/'weights').mkdir();shutil.copy2(checkpoint,out/'weights/best_checkpoint.pt')
 count=PINS[a.target]['videos_per_field'];indices=[i for i,split in enumerate(payload['splits']) if split=='test'];all_indices=list(indices)
 valid=[True]*len(indices)
 if len(indices)%count:
  padding=count-len(indices)%count;indices+=indices[:padding];valid += [False]*padding
 groups=[(indices[i:i+count],valid[i:i+count]) for i in range(0,len(indices),count)]
 if a.max_fields:groups=groups[:a.max_fields]
 planes,supports=phase_planes(model,a.target);(out/'phases').mkdir();(out/'inputs').mkdir()
 for stage,plane in planes.items():np.save(out/'phases'/f'{stage}.npy',plane)
 entries=[];pred=[];labels=[];audit=[]
 for index,(group,mask) in enumerate(groups):
  batch={}
  for k in ('vision_tokens','quality_tokens','raw_frames'):
   if k in payload:
    values=payload[k][group].clone()
    batch[k]=values.unsqueeze(0) if a.target=='temporal' else values
  for k in ('language_tokens','language_mask'):batch[k]=payload[k][0:1].clone()
  key=f'field_{index:04d}';path=out/'inputs'/f'{key}.pt';torch.save(batch,path)
  with torch.inference_mode():original=forward(model,batch)
  prediction=original['prediction'].detach().cpu().reshape(-1)
  for pos,is_valid in enumerate(mask):
   if is_valid:pred.append(float(prediction[pos]));labels.append(float(payload['targets'][group[pos]]))
  if index<2:
   result,tap=replay(model,a.target,batch)
   replayed,restored=replay(model,a.target,batch,tap.detectors)
   delta=float((result['prediction']-original['prediction']).abs().max());replay_delta=float((replayed['prediction']-original['prediction']).abs().max())
   if delta>1e-5 or replay_delta>1e-3 or tuple(tap.amplitudes)!=STAGES:raise RuntimeError(f'Six-pass numerical replay failed: {delta}, {replay_delta}')
   audit.append(dict(field=key,original_vs_tap_max_abs=delta,original_vs_six_ccd_replay_max_abs=replay_delta,passes=list(tap.amplitudes)))
  entries.append(dict(key=key,file=path.relative_to(out).as_posix(),sha256=sha(path),source_indices=group,valid=mask,sample_ids=[payload['sample_ids'][i] for i in group],targets=[float(payload['targets'][i]) for i in group],simulation_prediction=prediction.tolist()))
  if index%20==0:print('PACKED',a.target,index+1,'/',len(groups),flush=True)
 from experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.metrics import regression_metrics
 metrics=regression_metrics(torch.tensor(pred),torch.tensor(labels),a.target) if len(pred)>2 else None
 full=len(pred)==558
 if full and abs(metrics['srcc']-PINS[a.target]['srcc'])>0.00015:raise RuntimeError('Pinned model simulation did not reproduce requested SRCC: '+str(metrics))
 # Copy committed runtime only, never a dirty server optimization worktree.
 paths=['LightGenV2/__init__.py','LightGenV2/tasks/__init__.py','LightGenV2/tasks/t06_video_quality_assessment/__init__.py','LightGenV2/tasks/t06_video_quality_assessment/project.py','LightGenV2/tasks/t06_video_quality_assessment/models/__init__.py','LightGenV2/tasks/t06_video_quality_assessment/models/multivideo9x4.py','LightGenV2/tasks/t06_video_quality_assessment/multivideo_settings.py','LightGenV2/tasks/t06_video_quality_assessment/lab_runtime.py','LightGenV2/tasks/t06_video_quality_assessment/lab_bench.py','LightGenV2/tasks/t06_video_quality_assessment/lab_phase.py','experiments/__init__.py']
 backend='experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54'
 paths += [backend+'/'+name for name in ('__init__.py','modeling.py','settings.py','metrics.py')]
 for rel in paths:
  src=root/rel
  if not src.exists() and src.name=='__init__.py':continue
  dst=out/'runtime'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
 for name in ('run_lab.py','COMMAND_SHS.md'):
  shutil.copy2(root/'LightGenV2/tasks/t06_video_quality_assessment/hardware'/name,out/('run.py' if name=='run_lab.py' else name))
 # Do not accidentally borrow dependencies from the source checkout/PYTHONPATH.
 import sys
 subprocess.run([sys.executable,'-I',str(out/'run.py'),'--help'],cwd=out,check=True,stdout=subprocess.DEVNULL)
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
 release=dict(schema_version=1,target=a.target,reference_srcc=PINS[a.target]['srcc'],checkpoint_sha256=sha(checkpoint),source_checkpoint=str(checkpoint),source_commit=commit,field_video_count=count,frame_count=4,test_videos_in_package=len(pred),full_test=full,simulation_metrics=metrics,six_pass_replay=audit,fields=entries,automatic_phase_switching=False,physical_geometry=dict(model_pitch_um=17,device_pitch_um=8,active_pixels=478,distance_m=.1),feature_source=dict(manifest_sha256=payload['manifest_sha256'],vision_cache_path=str(settings.vision_cache_path),language_cache_path=str(settings.language_cache_path)))
 write(out/'release.json',release)
 assets={str(p.relative_to(out)).replace('\\','/'):sha(p) for p in out.rglob('*') if p.is_file()};write(out/'SHA256.json',assets)
 zip_path=out.with_suffix('.zip')
 if zip_path.exists():raise FileExistsError(zip_path)
 with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
  for p in out.rglob('*'):
   if p.is_file():z.write(p,p.relative_to(out).as_posix())
 write(out.with_suffix('.delivery.json'),dict(zip=str(zip_path),sha256=sha(zip_path),bytes=zip_path.stat().st_size,simulation_metrics=metrics))
 print('DELIVERED',zip_path,metrics,flush=True)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--target',choices=PINS,required=True);p.add_argument('--source-root',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');p.add_argument('--max-fields',type=int,default=0)
 build(p.parse_args())
if __name__=='__main__':main()
