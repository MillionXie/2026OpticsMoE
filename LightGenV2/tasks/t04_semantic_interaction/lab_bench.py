"""OpenMoji six-pass SHS deployment; no simulated fallback for completed planes."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from LightGenV2.tasks.t06_video_quality_assessment.lab_bench import (
    identity, raster, config, session, verified_ccd, capture_staged)
from LightGenV2.tasks.t06_video_quality_assessment.lab_runtime import read, write, sha
from .lab_runtime import STAGES, CHECKPOINT_SHA, REFERENCE


def open_session(a):
    root=Path(a.project).resolve(); s=session(root,a.session)
    state=read(s/'session.json'); c=config(a.config); release=read(root/'release.json')
    if state['hardware_sha256']!=identity(c) or state['release_sha256']!=sha(root/'release.json'):
        raise ValueError('Hardware/release changed: NEW session required')
    if release['target']!='openmoji' or release['stages']!=list(STAGES) or sha(root/'weights/best_checkpoint.pt')!=CHECKPOINT_SHA:
        raise ValueError('Wrong OpenMoji release')
    return root,s,state,c,release


def export(a):
    import torch
    from .lab_runtime import load_model,dataset,batch_for,phase_planes,replay
    from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.metrics import MetricAccumulator
    root=Path(a.project).resolve()
    if (root/'release.json').exists(): raise FileExistsError('Export already exists')
    model,cfg=load_model(root,a.device); data=dataset(cfg)
    acc=MetricAccumulator(); fields=[]; audits=[]
    for i in range(len(data)):
        batch=batch_for(data,i,a.device)
        output,tap=replay(model,batch)
        records,_,_=acc.update(output,batch)
        if i<4:
            restored,_=replay(model,batch,tap.detectors)
            error=max(float((output[k]-restored[k]).abs().max()) for k in ('category_logits','edit_logits','task_logits'))
            if error>2e-4 or tuple(tap.amplitudes)!=STAGES:
                raise RuntimeError('Six-plane replacement replay failed: '+str(error))
            audits.append(dict(index=i,max_logit_error=error,passed=True))
        record=data.records[i]
        source=root/'data'/record['relative_dir']/'source.png'
        fields.append(dict(key=f'field_{i:05d}',index=i,sample_id=record['sample_id'],
                           source_sha256=sha(source),simulation_metrics=records[0]))
        if i%100==0: print('SIMULATION',i+1,'/',len(data),flush=True)
    metrics=acc.compute()
    if len(data)!=1000 or abs(metrics['overall']['changed_cell_accuracy']-REFERENCE)>.005:
        raise RuntimeError('Pinned full-test reproduction mismatch: '+str(metrics))
    (root/'phases').mkdir(exist_ok=True)
    for stage,phase in phase_planes(model).items(): np.save(root/'phases'/(stage+'.npy'),phase)
    write(root/'release.json',dict(target='openmoji',checkpoint_sha256=CHECKPOINT_SHA,stages=list(STAGES),
        reference_changed_cell_accuracy=REFERENCE,simulation_metrics=metrics,full_test=True,fields=fields,
        six_plane_replay_audits=audits,dataset_sha256=sha(root/'data/test.jsonl'),
        token_cache_sha256=sha(root/'data/token_embeddings_v1.pt'),source=read(root/'SHS_SOURCE.json')))
    print('EXPORT COMPLETE',metrics,flush=True)


def initialize(a):
    root=Path(a.project).resolve(); release=read(root/'release.json'); c=config(a.config)
    s=session(root,a.session)
    if s.exists():raise FileExistsError('Session exists')
    if sha(root/'weights/best_checkpoint.pt')!=CHECKPOINT_SHA:raise ValueError('Wrong checkpoint')
    fields=release['fields'] if a.fields==0 else release['fields'][:a.fields]
    if not fields:raise ValueError('No samples')
    s.mkdir(parents=True); (s/'phase').mkdir()
    write(s/'session.json',dict(target='openmoji',hardware_sha256=identity(c),hardware_config=c,
        release_sha256=sha(root/'release.json'),fields=fields,measured_stages=[],created=time.strftime('%Y-%m-%dT%H:%M:%S')))
    for i,stage in enumerate(STAGES,1):
        plane=np.load(root/'phases'/(stage+'.npy'),allow_pickle=False)
        Image.fromarray(raster(plane,c['phase_slm'],'phase')).save(s/'phase'/f'{i:02d}_{stage}.bmp')
    print('INITIALIZED',s,len(fields),'samples',len(fields)*6,'captures',flush=True)


def measured_prefix(s,c,item,stages):
    import torch
    measured,hashes={},{}
    for stage in stages:
        p,record=verified_ccd(s,stage,item['key']); hashes[stage]=record['sha256']
        scale=float(c.get('detector_intensity_scale',{}).get(stage,1/255))
        if not np.isfinite(scale) or scale<=0:raise ValueError('Invalid detector scale')
        measured[stage]=torch.from_numpy(np.array(Image.open(p),dtype=np.float32))[None]*scale
    return measured,hashes


def checked_data(root,release,cfg):
    from .lab_runtime import dataset
    if sha(root/'data/test.jsonl')!=release['dataset_sha256'] or sha(root/'data/token_embeddings_v1.pt')!=release['token_cache_sha256']:
        raise ValueError('Test identities/token cache changed')
    return dataset(cfg)


def prepare(a):
    from .lab_runtime import load_model,batch_for,replay
    root,s,state,c,release=open_session(a); idx=STAGES.index(a.stage)
    if state['measured_stages']!=list(STAGES[:idx]):raise ValueError('Upstream stages incomplete')
    model,cfg=load_model(root,a.device); data=checked_data(root,release,cfg)
    dest=s/'play'/a.stage; dest.mkdir(parents=True,exist_ok=True); entries=[]
    for i,item in enumerate(state['fields']):
        record=data.records[item['index']]
        if record['sample_id']!=item['sample_id'] or sha(root/'data'/record['relative_dir']/'source.png')!=item['source_sha256']:
            raise ValueError('Input changed')
        batch=batch_for(data,item['index'],a.device)
        measured,upstream=measured_prefix(s,c,item,STAGES[:idx])
        _,tap=replay(model,batch,measured,a.stage)
        amp=tap.amplitudes[a.stage][0].cpu().numpy()
        bmp=dest/(item['key']+'.bmp')
        Image.fromarray(raster(amp,c['amplitude_slm'],'amplitude')).save(bmp)
        if i<16:
            preview=s/'theoretical_ccd'/a.stage; preview.mkdir(parents=True,exist_ok=True)
            intensity=tap.detectors[a.stage][0].cpu().numpy()
            np.save(preview/(item['key']+'.npy'),intensity)
            peak=max(float(np.percentile(intensity,99.5)),1e-12)
            Image.fromarray(np.rint(np.clip(intensity/peak,0,1)*255).astype(np.uint8)).save(preview/(item['key']+'.png'))
        entries.append(dict(key=item['key'],sample_id=item['sample_id'],bmp=bmp.name,sha256=sha(bmp),
                            upstream_ccd_sha256=upstream,positive_amplitude_p995=float(np.percentile(amp[amp>0],99.5))))
        if i%100==0:print('PREPARED',a.stage,i+1,'/',len(state['fields']),flush=True)
    phase=s/'phase'/f'{idx+1:02d}_{a.stage}.bmp'
    write(dest/'manifest.json',dict(stage=a.stage,hardware_sha256=state['hardware_sha256'],
        release_sha256=state['release_sha256'],phase_file=phase.relative_to(s).as_posix(),phase_sha256=sha(phase),entries=entries))
    print('READY',a.stage,len(entries),flush=True)


def capture(a):return capture_staged(a,open_session,STAGES)


def audit(a):
    root,s,state,c,release=open_session(a); idx=STAGES.index(a.stage)
    if state['measured_stages'][:idx+1]!=list(STAGES[:idx+1]):raise ValueError('Stage incomplete')
    mf=read(s/'play'/a.stage/'manifest.json')
    expected={i['key'] for i in state['fields']}
    if len(mf['entries'])!=len(expected) or {e['key'] for e in mf['entries']}!=expected:raise ValueError('Wrong sample identities')
    if sha(s/mf['phase_file'])!=mf['phase_sha256']:raise ValueError('Phase changed')
    for entry in mf['entries']:
        _,rec=verified_ccd(s,a.stage,entry['key'])
        for k,v in dict(hardware_sha256=state['hardware_sha256'],phase_sha256=mf['phase_sha256'],
                        amplitude_sha256=entry['sha256'],upstream_ccd_sha256=entry['upstream_ccd_sha256']).items():
            if rec[k]!=v:raise ValueError('Stale CCD record '+entry['key'])
        for previous,digest in entry['upstream_ccd_sha256'].items():
            _,old=verified_ccd(s,previous,entry['key'])
            if old['sha256']!=digest:raise ValueError('Upstream CCD changed')
    write(s/'audits'/(a.stage+'.json'),dict(status='passed',count=len(expected),phase_sha256=mf['phase_sha256']))


def evaluate(a):
    from .lab_runtime import load_model,batch_for,replay
    from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.metrics import MetricAccumulator
    from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.assets import load_icons,render_grid
    root,s,state,c,release=open_session(a)
    if state['measured_stages']!=list(STAGES):raise ValueError('All SIX actual CCD stages required')
    model,cfg=load_model(root,a.device); data=checked_data(root,release,cfg); acc=MetricAccumulator(); rows=[]
    icons=load_icons(cfg)
    for i,item in enumerate(state['fields']):
        batch=batch_for(data,item['index'],a.device)
        measured,hashes=measured_prefix(s,c,item,STAGES)
        output,_=replay(model,batch,measured)
        values,prediction,_=acc.update(output,batch)
        rows.append(dict(**values[0],ccd_sha256=hashes,predicted_grid=prediction[0].cpu().tolist()))
        if i<16:
            folder=s/'predictions'/item['sample_id']; folder.mkdir(parents=True,exist_ok=True)
            for name,grid in [('source',batch['source_grid'][0]),('target',batch['target_grid'][0]),('prediction',prediction[0])]:
                render_grid(grid.cpu().numpy(),cfg,icons).save(folder/(name+'.png'))
            write(folder/'instruction.json',dict(instruction=batch['instruction'][0],metrics=values[0]))
    write(s/'results.json',dict(status='real_six_pass_evaluation',metrics=acc.compute(),rows=rows,
        full_test=len(rows)==1000,reference_changed_cell_accuracy=REFERENCE,checkpoint_sha256=CHECKPOINT_SHA,
        hardware_sha256=state['hardware_sha256'],source_grid_used_only_for_final_preservation=True))
    print('RESULT',acc.compute(),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['export','init','prepare','capture','audit','evaluate','auto','probe'])
    p.add_argument('--project',default='.'); p.add_argument('--config',default='LAB.local.json')
    p.add_argument('--session',default='pilot01'); p.add_argument('--stage',choices=STAGES)
    p.add_argument('--fields',type=int,default=4); p.add_argument('--device',default='cuda')
    p.add_argument('--bench-root',default='../ABO_Lab_SHS_8um'); p.add_argument('--phase-ready',action='store_true')
    p.add_argument('--phase-config',default='PHASE.local.json'); a=p.parse_args()
    if a.fields<0 or (a.action in ('prepare','capture','audit') and not a.stage):p.error('Invalid fields/stage')
    if a.action in ('auto','probe'):
        from .lab_control import run
        return run(a)
    globals()[{'init':'initialize'}.get(a.action,a.action)](a)


if __name__=='__main__':main()
