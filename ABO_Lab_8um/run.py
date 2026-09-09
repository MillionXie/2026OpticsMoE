"""One-machine, six-stage measured-CCD replay. No simulated stage fallback."""
import argparse
import csv
import time
from pathlib import Path
import numpy as np
from PIL import Image
from common import ROOT,STAGES,config,setup_imports,session_path,sha,read,write,hardware_identity,CHECKPOINT_SHA

def samples(limit):
    setup_imports()
    # client.py is included in A100 source, but data definitions are kept local.
    root=ROOT/'assets/test_dataset'
    with (root/'titles.csv').open(encoding='utf-8-sig',newline='') as f: titles=list(csv.DictReader(f))
    with (root/'test.csv').open(encoding='utf-8-sig',newline='') as f: images=list(csv.DictReader(f))
    if len(images)!=2400 or [int(t['label']) for t in titles]!=list(range(100)): raise ValueError('Dataset contract mismatch')
    out=[{'id':f"title_{int(t['label']):03d}",'kind':'title','text':t['title'],'label':int(t['label'])} for t in titles]
    for r in images[:limit or None]:
        path=root/r['image_path']
        if not path.resolve().is_relative_to(root.resolve()): raise ValueError('Image path escape')
        out.append({'id':'image_'+r['sample_id'],'kind':'image','path':str(path),'label':int(r['label']),'source_sha256':sha(path)})
    return out

def load_session(name,c):
    root=session_path(name); state=read(root/'session.json')
    if state['hardware_identity']!=hardware_identity(c): raise ValueError('Hardware configuration changed. Use a NEW --session; no mixing old CCDs.')
    for s in state['samples']:
        if s['kind']=='image':
            # Rebase relative paths for portable sessions.
            s['path']=str(ROOT/s['path'])
            if sha(s['path'])!=s['source_sha256']: raise ValueError('Sample changed')
    return root,state

def measured(root,s):
    result={}; p=root/'ccd'/s['id']
    for stage in STAGES:
        record=p/(stage+'.record.json')
        if not record.exists(): continue
        meta=read(record)
        for key,value in meta['files'].items():
            if sha(p/key)!=value: raise ValueError(f'CCD file changed: {p/key}')
        result[stage]=read(p/(stage+'.route.json')) if stage.endswith('_router') else np.asarray(Image.open(p/(stage+'.png'))).copy()
    return result

def initialize(args,c):
    from patterns import calibration,export
    out=session_path(args.session)
    if (out/'session.json').exists(): raise FileExistsError('Session already exists; use prepare to resume.')
    ss=samples(args.limit)
    for s in ss:
        if s['kind']=='image': s['path']=str(Path(s['path']).relative_to(ROOT))
    calibration(c); export(c)
    write(out/'session.json',{'schema':1,'capture_mode':'real','checkpoint_sha256':CHECKPOINT_SHA,
        'hardware_identity':hardware_identity(c),'hardware_config':c,'samples':ss,
        'test_queries':sum(s['kind']=='image' for s in ss),'candidate_titles':100,
        'note':'Titles also require three measured Language stages; never silently use simulated titles.'})
    print('Session created:',out)

def prepare(args,c):
    from backend import create,forward,optical_contract
    from patterns import amplitude,save
    setup_imports(); from abo_dual.backend import NeedCapture
    root,state=load_session(args.session,c); b=create(args.device); b.guard_optics()
    stage=args.stage; out=root/'play'/stage; entries=[]
    contract=optical_contract(b); write(root/'optical_contract.json',contract)
    for s in state['samples']:
        if s['kind']=='title' and stage.startswith('vision'): continue
        m=measured(root,s)
        if stage in m: continue
        try: forward(b,s,m,release=True)
        except NeedCapture as e:
            if e.stage!=stage: raise RuntimeError(f"{s['id']}: needs {e.stage} first; requested {stage}. Finish previous capture.")
            field=e.amplitude
            if stage.endswith('_router'):
                if field.shape!=(224,224): raise ValueError('Router input shape mismatch')
                active=np.zeros((478,478),np.float32); active[127:351,127:351]=field; field=active
            bmp,encoding=amplitude(field,c)
            name=f"{len(entries):05d}.bmp"; save(out/name,bmp)
            entries.append({'id':s['id'],'bmp':name,'sha256':sha(out/name),'encoding':encoding})
        else: raise RuntimeError('Optical guard was bypassed; capture required')
        if len(entries)%20==0: print('Prepared',stage,len(entries),flush=True)
    phase=ROOT/'generated/P'/f'{STAGES.index(stage)+1:02d}_{stage}.bmp'
    write(out/'manifest.json',{'stage':stage,'phase_file':str(phase.relative_to(ROOT)),
        'phase_sha256':sha(phase),'hardware_identity':hardware_identity(c),'entries':entries})
    print('Ready:',len(entries),'BMPs. Load phase:',phase)
    from memory import memory_report
    report=memory_report(b); write(out/'compute_memory.json',report); print(report,flush=True)

def capture(args,c):
    from hardware import Bench
    setup_imports(); from abo_dual.common import routing_from_ccd
    from router_quality import assess,require_accepted,warn
    root,state=load_session(args.session,c); p=root/'play'/args.stage; mf=read(p/'manifest.json')
    if mf['hardware_identity']!=hardware_identity(c): raise ValueError('Prepared under another hardware configuration')
    phase=ROOT/mf['phase_file']
    if sha(phase)!=mf['phase_sha256']: raise ValueError('Phase changed since prepare')
    if not c['geometry_confirmed'] or c['capture_input_range'] is None: raise ValueError('Complete ROI/orientation/range calibration before formal capture')
    print('Load and KEEP this manual phase:',phase,'\nSHA256:',mf['phase_sha256'])
    if input('Type y only after checking the phase screen; otherwise Ctrl+C: ').strip().lower()!='y': raise RuntimeError('Phase not confirmed')
    contract=read(root/'optical_contract.json'); mapping={s['id']:s for s in state['samples']}
    with Bench(c) as bench:
        for i,e in enumerate(mf['entries'],1):
            if args.stage in measured(root,mapping[e['id']]): continue
            bmp=p/e['bmp']
            if sha(bmp)!=e['sha256']: raise ValueError('Amplitude BMP changed')
            out=root/'ccd'/e['id']/args.stage
            frame=bench.capture(bmp,out)
            files={out.name+'.png':sha(out.with_suffix('.png')),
                   out.name+'.tif':sha(out.with_suffix('.tif'))}
            quality=None
            if args.stage.endswith('_router'):
                files[out.name+'.json']=sha(out.with_suffix('.json'))
                # Capture-time identities survive rejection, enabling recovery
                # without any new exposure or changing original pixels.
                pending=out.with_suffix('.capture.json')
                write(pending,{'stage':args.stage,'sample':e['id'],'files':dict(files),
                    'phase_sha256':mf['phase_sha256'],'amplitude_sha256':e['sha256'],
                    'hardware_identity':hardware_identity(c),'capture_mode':'real'})
                files[pending.name]=sha(pending)
                route=routing_from_ccd(frame,contract)
                quality=assess(frame,route,contract)
                qp=out.with_suffix('.quality.json'); write(qp,quality)
                require_accepted(quality); warn(quality); files[qp.name]=sha(qp)
                rp=out.with_suffix('.route.json'); write(rp,route); files[rp.name]=sha(rp)
            write(out.with_suffix('.record.json'),{'stage':args.stage,'sample':e['id'],'files':files,
                'phase_sha256':mf['phase_sha256'],'amplitude_sha256':e['sha256'],
                'hardware_identity':hardware_identity(c),'capture_mode':'real','router_quality':quality})
            print(f'Captured {args.stage} {i}/{len(mf["entries"])} {e["id"]}',flush=True)

def evaluate(args,c):
    from backend import create,forward
    root,state=load_session(args.session,c); b=create(args.device); b.guard_optics()
    titles=[]; images=[]; labels=[]; ids=[]
    for s in state['samples']:
        m=measured(root,s); required=STAGES[3:] if s['kind']=='title' else STAGES
        if any(stage not in m for stage in required): raise ValueError(f'{s["id"]} has missing real CCD stages')
        v=forward(b,s,m,release=True)
        if s['kind']=='title': titles.append(v)
        else: images.append(v); labels.append(s['label']); ids.append(s['id'])
    import torch
    metrics,rows=b.m._metrics(torch.tensor(np.stack(images)),torch.tensor(np.stack(titles)),labels)
    write(root/'results/metrics.json',{'mode':'real_six_stage','metrics':metrics,'n_queries':len(images),
        'n_candidates':100,'historical_simulation_r1':1934/2400,'checkpoint_sha256':CHECKPOINT_SHA,
        'predictions':[dict(sample_id=s,**r) for s,r in zip(ids,rows)]})
    np.savez(root/'results/embeddings.npz',queries=np.stack(images),titles=np.stack(titles),labels=labels)
    print(metrics)
    from memory import memory_report
    write(root/'results/compute_memory.json',memory_report(b))

def probe(args,c):
    from hardware import Bench
    p=ROOT/'results/probe'; p.mkdir(parents=True,exist_ok=True)
    with Bench(c,slm=bool(args.bmp)) as bench:
        bench.capture(ROOT/args.bmp if args.bmp else None,p/'raw',rectify=False)
    print('Raw unwarped camera image:',p/'raw.tif')

def exposure(args,c):
    from hardware import Bench,canonical
    out=ROOT/'results/exposure'/time.strftime('%Y%m%d_%H%M%S'); out.mkdir(parents=True)
    print('Keep generated/cal/P_ZERO.bmp on phase SLM. 32 gray values x 3 frames.')
    if input('Ready? y: ').strip().lower()!='y': return
    rows=[]
    with Bench(c) as bench:
        for bmp in sorted((ROOT/'generated/cal/gray').glob('*.bmp')):
            for rep in range(3):
                raw=bench.capture(bmp,out/f'{bmp.stem}_{rep}',rectify=False)
                region=canonical(raw,c)[111:367,111:367] if c['geometry_confirmed'] else raw[raw.shape[0]//2-128:raw.shape[0]//2+128,raw.shape[1]//2-128:raw.shape[1]//2+128]
                maximum=c['capture_input_range'][1] if c['capture_input_range'] else np.iinfo(raw.dtype).max
                rows.append({'gray':int(bmp.stem[1:]),'frame':rep,'mean':float(region.mean()),
                    'max_raw':int(raw.max()),'full_frame_saturation_fraction':float((raw>=maximum).mean()),
                    'measurement_window':'canonical center256' if c['geometry_confirmed'] else 'uncalibrated raw center256'})
    write(out/'response.json',{'config':c,'rows':rows,'no_LUT_generated':True})
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    grays=sorted(set(r['gray'] for r in rows)); means=[np.mean([r['mean'] for r in rows if r['gray']==g]) for g in grays]
    fig,ax=plt.subplots(); ax.plot(grays,means,'o-'); ax.set(xlabel='Amplitude gray',ylabel='Mean CCD (fixed scale)'); fig.savefig(out/'response.png',dpi=160); plt.close(fig)
    print('Saved',out,'; this checks the new Holoeye response, not the old Meadowlark LUT.')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['init','prepare','capture','evaluate','probe','exposure'])
    p.add_argument('--config'); p.add_argument('--session',default='pilot01')
    p.add_argument('--stage',choices=STAGES); p.add_argument('--device',default='auto')
    p.add_argument('--limit',type=int,default=4,help='Test queries only; all 100 candidate titles are always retained. 0=2400.')
    p.add_argument('--bmp',help='Relative to this package; probe otherwise captures camera only')
    a=p.parse_args()
    if a.action in ('prepare','capture') and a.stage is None: p.error('--stage required')
    if not 0<=a.limit<=2400: p.error('--limit must be 0..2400')
    c,_=config(a.config); globals()[{'init':'initialize'}.get(a.action,a.action)](a,c)

if __name__=='__main__': main()
