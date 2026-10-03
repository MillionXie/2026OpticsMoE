"""SALICON three-pass bench; actual upstream CCDs drive all later inputs."""
import argparse
from pathlib import Path
import time
import numpy as np
from PIL import Image
from LightGenV2.tasks.t06_video_quality_assessment.lab_bench import (
    identity, raster, config, session, verified_ccd, capture_staged)
from LightGenV2.tasks.t06_video_quality_assessment.lab_runtime import read,write,sha

STAGES=('vision_router','vision_expert','vision_global')
CHECKPOINT_SHA='036bc8caedcde4d6dabce276b1a2e4af15d960d88620098b19e1c198849b8bfe'


def open_session(a):
    root=Path(a.project).resolve();s=session(root,a.session)
    state=read(s/'session.json');c=config(a.config);release=read(root/'release.json')
    if state['hardware_sha256']!=identity(c) or state['release_sha256']!=sha(root/'release.json'):
        raise ValueError('Hardware/release changed; create a NEW session')
    if release['target']!='salicon' or release['stages']!=list(STAGES) or sha(root/'weights/best_checkpoint.pt')!=CHECKPOINT_SHA:
        raise ValueError('Wrong SALICON release/weights')
    return root,s,state,c,release


def initialize(a):
    root=Path(a.project).resolve();release=read(root/'release.json');c=config(a.config)
    s=session(root,a.session)
    if s.exists():raise FileExistsError('Session exists; do not overwrite')
    if release['checkpoint_sha256']!=CHECKPOINT_SHA or sha(root/'weights/best_checkpoint.pt')!=CHECKPOINT_SHA:
        raise ValueError('Wrong SALICON checkpoint')
    fields=release['fields'] if a.fields==0 else release['fields'][:a.fields]
    if not fields:raise ValueError('No samples')
    s.mkdir(parents=True)
    write(s/'session.json',dict(target='salicon',hardware_sha256=identity(c),hardware_config=c,
         release_sha256=sha(root/'release.json'),fields=fields,measured_stages=[],
         created=time.strftime('%Y-%m-%dT%H:%M:%S')))
    dest=s/'phase';dest.mkdir()
    for i,stage in enumerate(STAGES,1):
        plane=np.load(root/'phases'/(stage+'.npy'),allow_pickle=False)
        Image.fromarray(raster(plane,c['phase_slm'],'phase')).save(dest/f'{i:02d}_{stage}.bmp')
    print('INITIALIZED',s,len(fields),'images,',len(fields)*3,'captures',flush=True)


def measured_prefix(s,c,item,stages):
    import torch
    values,hashes={},{}
    for stage in stages:
        p,rec=verified_ccd(s,stage,item['key']);hashes[stage]=rec['sha256']
        scale=float(c.get('detector_intensity_scale',{}).get(stage,1/255))
        if scale<=0:raise ValueError('Invalid fixed detector scale')
        values[stage]=torch.from_numpy(np.array(Image.open(p),dtype=np.float32))[None]*scale
    return values,hashes


def prepare(a):
    import torch
    from .lab_runtime import load_model,replay
    root,s,state,c,release=open_session(a);idx=STAGES.index(a.stage)
    if state['measured_stages']!=list(STAGES[:idx]):raise ValueError('Upstream stages incomplete')
    model=load_model(root,a.device);dest=s/'play'/a.stage;dest.mkdir(parents=True,exist_ok=True)
    entries=[]
    for i,item in enumerate(state['fields']):
        source=root/item['file']
        if sha(source)!=item['sha256']:raise ValueError('Cached input modified')
        batch=torch.load(source,map_location='cpu',weights_only=False)
        measured,upstream=measured_prefix(s,c,item,STAGES[:idx])
        _,tap=replay(model,batch,measured,a.stage)
        amp=tap.amplitudes[a.stage][0].cpu().numpy()
        bmp=dest/(item['key']+'.bmp')
        Image.fromarray(raster(amp,c['amplitude_slm'],'amplitude')).save(bmp)
        if i<16:
            preview=s/'theoretical_ccd'/a.stage;preview.mkdir(parents=True,exist_ok=True)
            intensity=tap.detectors[a.stage][0].cpu().numpy()
            np.save(preview/(item['key']+'.npy'),intensity)
            peak=max(float(np.percentile(intensity,99.5)),1e-12)
            Image.fromarray(np.rint(np.clip(intensity/peak,0,1)*255).astype(np.uint8)).save(preview/(item['key']+'.png'))
        entries.append(dict(key=item['key'],sample_id=item['sample_id'],bmp=bmp.name,sha256=sha(bmp),
                            upstream_ccd_sha256=upstream,positive_amplitude_p995=float(np.percentile(amp[amp>0],99.5))))
        if i%100==0:print('Prepared',a.stage,i+1,'/',len(state['fields']),flush=True)
    phase=s/'phase'/f'{idx+1:02d}_{a.stage}.bmp'
    write(dest/'manifest.json',dict(stage=a.stage,hardware_sha256=state['hardware_sha256'],
         release_sha256=state['release_sha256'],phase_file=phase.relative_to(s).as_posix(),
         phase_sha256=sha(phase),entries=entries))
    print('READY',a.stage,len(entries),flush=True)


def capture(a):
    return capture_staged(a,open_session,STAGES)


def audit(a):
    """Read-only complete-stage audit, used before releasing the phase owner."""
    root,s,state,c,release=open_session(a)
    idx=STAGES.index(a.stage)
    if state['measured_stages'][:idx+1]!=list(STAGES[:idx+1]):raise ValueError('Stage incomplete')
    mf=read(s/'play'/a.stage/'manifest.json')
    expected={item['key'] for item in state['fields']}
    if len(mf['entries'])!=len(expected) or {e['key'] for e in mf['entries']}!=expected:raise ValueError('Manifest sample mismatch')
    if sha(s/mf['phase_file'])!=mf['phase_sha256']:raise ValueError('Phase changed')
    rows=[]
    for e in mf['entries']:
        p,r=verified_ccd(s,a.stage,e['key'])
        if r['hardware_sha256']!=state['hardware_sha256'] or r['phase_sha256']!=mf['phase_sha256'] or r['amplitude_sha256']!=e['sha256'] or r['upstream_ccd_sha256']!=e['upstream_ccd_sha256']:
            raise ValueError('Capture identity mismatch: '+e['key'])
        if r['camera'].get('incomplete',False):raise ValueError('Incomplete camera frame')
        for stage,digest in e['upstream_ccd_sha256'].items():
            _,previous=verified_ccd(s,stage,e['key'])
            if previous['sha256']!=digest:raise ValueError('Upstream image changed')
        rows.append(dict(key=e['key'],**r['quality']))
    report=dict(stage=a.stage,count=len(rows),status='passed',hardware_sha256=state['hardware_sha256'],
                phase_sha256=mf['phase_sha256'],minimum_p99=min(r['p99'] for r in rows),
                maximum_saturation=max(r['saturation'] for r in rows),rows=rows)
    write(s/'audits'/(a.stage+'.json'),report)
    print('AUDIT PASSED',a.stage,len(rows),'min_p99',report['minimum_p99'],flush=True)


def evaluate(a):
    import torch
    from .lab_runtime import load_model,replay,REFERENCE_CC
    from .reproduce_baseline import independent_cc
    from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency.objectives import SaliencyAccumulator,density_from_logits
    root,s,state,c,release=open_session(a)
    if state['measured_stages']!=list(STAGES):raise ValueError('Require all THREE actual CCD stages')
    model=load_model(root,a.device);acc=SaliencyAccumulator();rows=[]
    for i,item in enumerate(state['fields']):
        path=root/item['file']
        if sha(path)!=item['sha256']:raise ValueError('Cached sample changed')
        batch=torch.load(path,map_location='cpu',weights_only=False)
        measured,hashes=measured_prefix(s,c,item,STAGES)
        logits,_=replay(model,batch,measured)
        density=density_from_logits(logits)
        acc.update(logits,batch['density'].to(a.device),batch['fixation'].to(a.device))
        cc=float(independent_cc(density.cpu().numpy(),batch['density'].numpy())[0])
        rows.append(dict(sample_id=item['sample_id'],key=item['key'],cc_float64=cc,
                         simulation_cc=item['simulation_cc'],ccd_sha256=hashes))
        if i<16:
            out=s/'saliency_previews';out.mkdir(exist_ok=True)
            for label,value in [('measured_prediction',density),('target',batch['density'])]:
                image=value.detach().float().cpu().numpy().squeeze()
                np.save(out/(item['key']+'_'+label+'.npy'),image)
                Image.fromarray(np.rint(image/max(float(image.max()),1e-12)*255).astype(np.uint8)).save(out/(item['key']+'_'+label+'.png'))
        if i%100==0:print('Evaluated',i+1,'/',len(state['fields']),'CC',np.mean([r['cc_float64'] for r in rows]),flush=True)
    result=dict(status='real_three_pass_evaluation',metrics=acc.compute(),cc_float64=float(np.mean([r['cc_float64'] for r in rows])),
                samples=len(rows),full_test=len(rows)==5000,simulation_reference_cc=REFERENCE_CC,
                simulation_same_subset_cc=float(np.mean([r['simulation_cc'] for r in rows])),rows=rows,
                checkpoint_sha256=CHECKPOINT_SHA,hardware_sha256=state['hardware_sha256'],release_sha256=state['release_sha256'])
    write(s/'results.json',result);print('RESULT',s/'results.json',result['metrics'],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['init','prepare','capture','evaluate','audit']);p.add_argument('--project',default='.')
    p.add_argument('--config',default='LAB.20260914.json');p.add_argument('--session',required=True)
    p.add_argument('--stage',choices=STAGES);p.add_argument('--fields',type=int,default=4)
    p.add_argument('--device',default='cuda');p.add_argument('--bench-root',default='../ABO_Lab_SHS_8um')
    p.add_argument('--phase-ready',action='store_true');a=p.parse_args()
    if a.fields<0 or (a.action in ('prepare','capture','audit') and not a.stage):p.error('Invalid fields/stage')
    globals()[{'init':'initialize'}.get(a.action,a.action)](a)

if __name__=='__main__':main()
