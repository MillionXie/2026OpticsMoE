"""Optical phase-switch evidence, distinct from SDK file/ACK verification.

Fixed uniform amplitude, fixed 150us exposure. A flat/target/flat/target
sequence must change the CCD and reproduce both states. This proves a
repeatable response, not exact per-pixel phase calibration or optical fidelity.
No production images are replaced. Run in the SAME SDK owner as capture.
"""
import threading,time,uuid
import copy
from pathlib import Path
import numpy as np
from .lab_runtime import read,write,sha

def probe_config(formal,approved,token):
 """Keep historical diagnostic geometry intact; use only its RAW sensor output.

 Current formal ROI is applied separately below, never forged into old evidence.
 """
 c=copy.deepcopy(approved)
 c['camera']=copy.deepcopy(formal['camera']);c['camera']['exposure_us']=150.
 c['amplitude_slm']=copy.deepcopy(formal['amplitude_slm'])
 c['settle_delay_ms']=formal['settle_delay_ms']
 c.update(diagnostic_only=True,diagnostic_session='smoke_phase_guard_'+token)
 c.pop('camera_exposure_us_by_stage',None)
 return c

def compare(a,b):
 valid=(a<250)&(b<250)
 if valid.mean()<.99 or min(a[valid].std(),b[valid].std())<2:
  raise ValueError('Phase probe clipped or lacks spatial signal')
 return float(np.corrcoef(a[valid],b[valid])[0,1])

def assess(frames):
 if len(frames)!=4:raise ValueError('Expected flat/target/flat/target')
 repeats=[compare(frames[0],frames[2]),compare(frames[1],frames[3])]
 cross=[compare(frames[i],frames[j]) for i in (0,2) for j in (1,3)]
 return dict(passed=min(repeats)>=.95 and max(cross)<=.98 and min(repeats)-max(cross)>=.02,
             repeat_pcc=repeats,cross_pcc=cross,minimum_repeat_pcc=.95,
             maximum_cross_pcc=.98,minimum_separation=.02)

def reference_match(current,reference):
 """Cross-time reference: reject wrong patterns without amplifying speckle drift.

 Keep raw evidence and a raw floor; a fixed sigma=1 ROI-pixel filter checks the
 stable structure. This filtering NEVER touches production CCD tensors.
 Same-sequence switching and repeatability still use unfiltered PCC.
 """
 import cv2
 raw=compare(current,reference)
 smooth=compare(cv2.GaussianBlur(current,(0,0),1),cv2.GaussianBlur(reference,(0,0),1))
 return dict(raw_pcc=raw,structure_pcc=smooth,minimum_raw_pcc=.90,
             minimum_structure_pcc=.98,diagnostic_gaussian_sigma_px=1.,passed=raw>=.90 and smooth>=.98)

def probe(a,link,sdk,pump,out,label):
 """Worker owns only remote camera I/O; caller pumps phase SDK main thread."""
 import cv2
 from PIL import Image
 from dual_run import Remote
 from .lab_bench import raster
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 token=uuid.uuid4().hex;case='results/phase_guard/'+token
 errors=[];data={}
 def capture():
  try:
   with Remote(link) as r:
    with r.sftp.open(a.project+'/'+a.config,'rb') as f:
     import json
     c=json.loads(f.read().decode('utf-8'))
    base=link.get('phase_probe_config','results/smoke_configs/smoke_abo_full400us240ms_20260913.json')
    diagnostic=probe_config(c,r.read(base),token)
    image=out/(label+'_amplitude.bmp')
    Image.fromarray(raster(np.ones((478,478),np.float32),c['amplitude_slm'],'amplitude')).save(image)
    r.ps(f"New-Item -ItemType Directory -Force -Path '{r.root}/{case}' | Out-Null")
    r.sftp.put(str(image),r.root+'/'+case+'/uniform.bmp')
    config='results/smoke_configs/phase_guard_'+token+'.json';r.putjson(config,diagnostic)
    r.job(dict(action='probe',config=config,bmp=case+'/uniform.bmp',out=case+'/capture'))
    path=out/(label+'_raw.png');r.sftp.get(r.root+'/'+case+'/capture/raw.png',str(path))
    H=cv2.getPerspectiveTransform(np.float32([c['logical_corners_full_sensor_xy'][k] for k in ['top_left','top_right','bottom_right','bottom_left']]),np.float32([[-.5,-.5],[477.5,-.5],[477.5,477.5],[-.5,477.5]]))
    im=cv2.warpPerspective(np.array(Image.open(path),np.float32),H,(478,478))
    Image.fromarray(np.rint(im).astype(np.uint8)).save(out/(label+'_roi.png'))
    data.update(image=im,raw_sha256=sha(path),amplitude_sha256=sha(image),remote=case)
  except BaseException as e:errors.append(e)
 worker=threading.Thread(target=capture);worker.start();last=time.monotonic()
 while worker.is_alive():
  pump()
  if time.monotonic()-last>=1:sdk.repeat();last=time.monotonic()
  time.sleep(.01)
 worker.join()
 if errors:raise errors[0]
 return data

def preflight(a,link,sdk,pump,out):
 from PIL import Image
 out=Path(out);out.mkdir(parents=True,exist_ok=False)
 flat=out/'flat.bmp';Image.fromarray(np.zeros((1200,1920),np.uint8)).save(flat)
 frames=[];evidence=[]
 for i,phase in enumerate([flat,a.phase,flat,a.phase]):
  if (a.out/'RELEASE').exists():raise RuntimeError('Release requested during phase preflight')
  receipt=sdk.show(phase,pump=pump)
  row=probe(a,link,sdk,pump,out,str(i));frames.append(row.pop('image'));evidence.append(dict(row,receipt=receipt))
 report=dict(assess(frames),exposure_us=150.,phase_sha256=sha(a.phase),rows=evidence,
             scope='Repeatable optical switching only; not exact phase fidelity certification')
 if getattr(a,'phase_reference_dir',None):
  bank=Path(a.phase_reference_dir);entry=read(bank/'manifest.json')['stages'][a.stage]
  if entry['phase_sha256']!=sha(a.phase) or entry['exposure_us']!=150.:raise ValueError('Phase reference identity mismatch')
  expected=bank/entry['roi_file']
  if sha(expected)!=entry['roi_sha256']:raise ValueError('Phase reference image changed')
  reference=np.array(Image.open(expected),np.float32)
  report['reference_checks']=[reference_match(frames[i],reference) for i in (1,3)]
  report['reference_manifest_sha256']=sha(bank/'manifest.json')
  report['passed']=report['passed'] and all(x['passed'] for x in report['reference_checks'])
 write(out/'report.json',report)
 if not report['passed']:raise ValueError('Optical phase transition preflight failed; production prohibited')
 return frames[-1]

def postflight(a,link,sdk,pump,out,reference):
 row=probe(a,link,sdk,pump,out,'post');im=row.pop('image');pcc=compare(reference,im)
 report=dict(row,pcc_to_preflight=pcc,passed=pcc>=.95,minimum_pcc=.95)
 write(Path(out)/'postflight.json',report)
 if not report['passed']:raise ValueError('Phase postflight changed; stage requires review/recapture')
