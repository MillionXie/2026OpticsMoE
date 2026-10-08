"""LSP cached-stem export and sequential three-layer SHS acquisition.

Caches only frozen Qwen stem tokens, never the learned optical/electronic path.
Uses the existing calibrated bench read-only; no shared-source modification.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

from .build_lab_package import sha
from .lab_field_units import FieldUnits, STAGES
from .lab_preflight import EXPECTED_SHA


def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_suffix(p.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')
    temporary.replace(p)


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def export(a):
    import torch
    from . import run, training, modeling
    from .lab_runtime import CachedStudent, phase_planes
    root = a.project.resolve()
    if root.exists(): raise FileExistsError('Preserve existing export')
    if sha(a.checkpoint) != EXPECTED_SHA: raise ValueError('Wrong LSP PT')
    root.mkdir(parents=True); (root/'cache').mkdir()
    entries = []
    build = modeling.build_student
    cached = None
    def factory(loaded, settings):
        nonlocal cached
        if settings.tta_enabled: raise ValueError('This 3000-frame release requires original no-TTA protocol')
        model = build(loaded, settings)
        original = model.core.forward_groups
        def groups(values, shapes):
            model._lab_groups = values
            model._lab_shapes = shapes
            return original(values, shapes)
        model.core.forward_groups = groups
        def observed(m, inputs, result):
            nonlocal cached
            if cached is None:
                write(root/'settings.json', settings.__dict__)
                cached = CachedStudent(SimpleNamespace(**settings.__dict__)).eval().to(loaded.device)
                cached.core.load_state_dict(m.core.state_dict(), strict=True)
                cached.head.load_state_dict(m.head.state_dict(), strict=True)
                (root/'phases').mkdir()
                import numpy as np
                for stage, value in phase_planes(cached).items(): np.save(root/'phases'/(stage+'.npy'), value)
            joined = {'tokens':torch.cat(m._lab_groups).detach(), 'grid':torch.tensor(m._lab_shapes, device=loaded.device)}
            with torch.no_grad(): batch_output = cached(joined)
            if not torch.allclose(batch_output[0], result[0], atol=2e-5, rtol=2e-5):
                raise RuntimeError('Cached model changes the original same-batch graph')
            for i, tokens in enumerate(m._lab_groups):
                key = f'test_{len(entries):05d}'
                batch = {'tokens':tokens.detach().cpu(), 'grid':torch.tensor([m._lab_shapes[i]])}
                with torch.no_grad():
                    output = cached(batch)
                error = float((output[0]-result[0][i:i+1]).abs().max())
                # Batch-one GEMM/FFT kernels can differ from batch24 in float32.
                # Same-batch identity is checked above; retain measured delta.
                if not torch.allclose(output[0], result[0][i:i+1], atol=2e-3, rtol=2e-5):
                    raise RuntimeError(f'Cached stem changes heatmap: {error}')
                batch['simulation_heatmap'] = result[0][i:i+1].detach().cpu()
                p = root/'cache'/(key+'.pt'); torch.save(batch,p)
                entries.append({'key':key,'file':p.relative_to(root).as_posix(),'sha256':sha(p),'cached_heatmap_error':error})
                if len(entries)%100==0: print('EXPORTED',len(entries),flush=True)
        model.register_forward_hook(observed)
        return model
    training.build_student = factory; run.build_student = factory
    result = run.run(SimpleNamespace(profile='main_dc20_no_shift_warmstart',run_dir=str(root/'reference'),seed=42,phase='evaluate',checkpoint=str(a.checkpoint)))
    if len(entries)!=1000 or result['test_samples']!=1000: raise ValueError('Require original full TEST1000')
    import shutil
    (root/'weights').mkdir(); shutil.copy2(a.checkpoint,root/'weights/best_checkpoint.pt')
    write(root/'release.json',{'checkpoint_sha256':EXPECTED_SHA,'stages':STAGES,'fields':entries,'settings_sha256':sha(root/'settings.json'),
        'phase_sha256':{s:sha(root/'phases'/(s+'.npy')) for s in STAGES},'simulation_metrics':result['metrics'],
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'amplitude_scale':16.,'encoding':'round(255*A/16), no clipping/peak normalization'})
    print('EXPORT COMPLETE',len(entries),flush=True)


def acquire(a):
    import numpy as np
    import torch
    from PIL import Image
    from .lab_runtime import load_model, phase_planes
    root=a.project.resolve(); output=a.output.resolve()
    if output.exists(): raise FileExistsError('Existing session: do not overwrite or blindly resume')
    release=read(root/'release.json')
    if release['checkpoint_sha256']!=EXPECTED_SHA or sha(root/'settings.json')!=release['settings_sha256']:
        raise ValueError('Wrong release')
    items=release['fields'][:a.limit]
    if len(release['fields'])!=1000 or len(items)!=a.limit: raise ValueError('Wrong sample count')
    torch.set_num_threads(4)
    model=load_model(root)
    planes=phase_planes(model)
    sys.path.insert(0,str(a.bench_root.resolve()))
    import four_image_flow as flow
    import shs_physical2400 as legacy
    for s in STAGES:
        if sha(root/'phases'/(s+'.npy'))!=release['phase_sha256'][s] or not np.allclose(planes[s],np.load(root/'phases'/(s+'.npy')),atol=1e-5):
            raise ValueError('Phase identity mismatch')
    # Independently audit the exact deployed phase representation before SDK.
    for item in items[:4]:
        p=root/item['file']
        if sha(p)!=item['sha256']:raise ValueError('Input cache SHA')
        batch=torch.load(p,map_location='cpu',weights_only=False)
        with torch.inference_mode(),FieldUnits(model,quantize=True,planes=planes) as tap: before=model(batch)
        with torch.inference_mode(),FieldUnits(model,quantize=True,measured=tap.detectors,planes=planes): after=model(batch)
        if not torch.allclose(before[0],after[0],atol=2e-5,rtol=2e-5):raise ValueError('Ideal CCD bridge failed')
    flow.BASE_CORNERS=legacy.CORNERS.copy()
    output.mkdir(parents=True);(output/'phase').mkdir()
    phases={}
    for s in STAGES:
        p=output/'phase'/(s+'.bmp');Image.fromarray(flow.phase_gray(planes[s],'hv',True)).save(p);phases[s]=p
    contract={'checkpoint_sha256':EXPECTED_SHA,'release_sha256':sha(root/'release.json'),'samples':a.limit,
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'amplitude_scale':16.,'detector_scale':1/255,
        'exposure_us':2000,'gain':'Gain_X4','wait_ms':240,'camera_orientation':'flip_v','phase_orientation':'hv','phase_inverse':True,
        'corners':legacy.CORNERS.tolist(),'phase_sha256':{s:sha(p) for s,p in phases.items()},
        'bench_source_sha256':sha(Path(legacy.__file__)),'full_run':a.limit==1000}
    write(output/'contract.json',contract)
    reused=0
    if a.reuse_session is not None:
        import shutil
        previous=read(a.reuse_session/'capture_report.json')
        prior=previous['contract']
        if previous['status']!='complete' or not previous['sdk_released']:raise ValueError('Pilot incomplete')
        for key in ('checkpoint_sha256','release_sha256','amplitude_scale','detector_scale','exposure_us','gain','wait_ms','camera_orientation','phase_orientation','phase_inverse','corners','phase_sha256','bench_source_sha256'):
            if prior[key]!=contract[key]:raise ValueError('Pilot identity differs: '+key)
        for stage in STAGES:
            dest=output/'ccd'/stage;dest.mkdir(parents=True)
            for item in items[:previous['samples']]:
                p=a.reuse_session/'ccd'/stage/(item['key']+'.png');rec=p.with_suffix('.json')
                if sha(p)!=read(rec)['ccd_sha256']:raise ValueError('Pilot CCD changed')
                shutil.copy2(p,dest/p.name);shutil.copy2(rec,dest/rec.name);reused+=1
    start=time.time()
    with legacy.SHSBench(output,2000,240,{}) as bench:
        for n,s in enumerate(STAGES):
            amp=output/'amplitude'/s;amp.mkdir(parents=True)
            for k,item in enumerate(items):
                p=root/item['file']
                if sha(p)!=item['sha256']:raise ValueError('Cache changed')
                batch=torch.load(p,map_location='cpu',weights_only=False)
                measured={}
                upstream={}
                for old in STAGES[:n]:
                    c=output/'ccd'/old/(item['key']+'.png');upstream[old]=sha(c)
                    measured[old]=torch.from_numpy(np.array(Image.open(c),dtype=np.float32))[None]/255
                with torch.inference_mode(),FieldUnits(model,quantize=True,measured=measured,planes=planes) as tap:model(batch)
                value=tap.amplitudes[s][0].cpu().numpy()
                # Verify phase planes exactly represent this model's input field.
                bmp=amp/(item['key']+'.bmp')
                Image.fromarray(flow.active_to_native(np.rint(value*255).astype(np.uint8))).save(bmp)
                c=output/'ccd'/s/(item['key']+'.png')
                if c.exists():
                    rec=read(c.with_suffix('.json'))
                    if sha(c)!=rec['ccd_sha256'] or sha(bmp)!=rec['amplitude_sha256'] or rec['upstream_ccd_sha256']!=upstream or rec['phase_sha256']!=sha(phases[s]):
                        raise ValueError('Reused CCD not identical to current input/phase')
                    continue
                values,_=bench.capture(s,phases[s],[bmp],[item['key']],'flip_v',save=False)
                row=bench.rows[-1]
                if row['p99']<15: raise RuntimeError('Dark CCD: stop and diagnose; no blind retry')
                c=output/'ccd'/s/(item['key']+'.png');c.parent.mkdir(parents=True,exist_ok=True)
                Image.fromarray(values[0]).save(c)
                write(c.with_suffix('.json'),dict(row,ccd_sha256=sha(c),upstream_ccd_sha256=upstream,checkpoint_sha256=EXPECTED_SHA,contract_sha256=sha(output/'contract.json')))
                write(output/'progress.json',{'status':'capturing','stage':s,'stage_completed':k+1,'per_stage':a.limit,'ccd_completed':n*a.limit+k+1,'elapsed_seconds':time.time()-start})
                if (k+1)%20==0:print('CAPTURED',s,k+1,'/',a.limit,flush=True)
    write(output/'capture_report.json',{'status':'complete','samples':a.limit,'ccd_count':a.limit*3,'reused_captures':reused,'sdk_released':True,'elapsed_seconds':time.time()-start,'contract':contract})
    print('CAPTURE COMPLETE',a.limit*3,flush=True)


def queued(a):
    """One bounded handoff, not a recurring task or a blind retry."""
    import tarfile
    deadline=time.monotonic()+7200
    state=a.output.with_suffix('.queue.json')
    write(state,{'status':'waiting_for_verified_cache','pilot':str(a.reuse_session)})
    while not a.cache_archive.exists() or a.cache_archive.stat().st_size!=a.archive_bytes:
        if time.monotonic()>deadline:raise TimeoutError('Cache transfer incomplete after two hours; SDK never opened')
        time.sleep(5)
    if sha(a.cache_archive)!=a.archive_sha256:raise ValueError('Cache archive SHA mismatch')
    root=a.project.resolve();release=read(root/'release.json')
    allowed={e['file']:e['sha256'] for e in release['fields']}
    with tarfile.open(a.cache_archive) as tar:
        for member in tar.getmembers():
            dest=(root/member.name).resolve()
            if not member.isfile() or not dest.is_relative_to(root) or member.name not in allowed:raise ValueError('Unexpected artifact member')
            if dest.exists():raise FileExistsError('Preserve existing cache member')
            with tar.extractfile(member) as source,dest.open('xb') as target:
                import shutil
                shutil.copyfileobj(source,target)
            if sha(dest)!=allowed[member.name]:raise ValueError('Extracted cache SHA mismatch')
    for item in release['fields']:
        if sha(root/item['file'])!=item['sha256']:raise ValueError('Incomplete full cache')
    if a.reuse_session is None:raise ValueError('Require verified three-layer pilot')
    report=read(a.reuse_session/'capture_report.json')
    if report['status']!='complete' or not report['sdk_released'] or report['ccd_count']!=12:raise ValueError('Pilot not complete')
    write(state,{'status':'capturing_full','samples':1000,'captures':3000})
    acquire(a)
    write(state,{'status':'capture_complete','samples':1000,'captures':3000,'sdk_released':True})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['export','acquire','queued'])
    p.add_argument('--project',type=Path,required=True);p.add_argument('--checkpoint',type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--bench-root',type=Path)
    p.add_argument('--reuse-session',type=Path)
    p.add_argument('--cache-archive',type=Path);p.add_argument('--archive-bytes',type=int)
    p.add_argument('--archive-sha256')
    p.add_argument('--limit',type=int,default=4);a=p.parse_args()
    if a.action=='export' and a.checkpoint is None:p.error('checkpoint required')
    if a.action in ('acquire','queued') and (a.output is None or a.bench_root is None or not 1<=a.limit<=1000):p.error('output/bench-root/limit required')
    if a.action=='queued' and (a.limit!=1000 or a.cache_archive is None or a.archive_bytes is None or not a.archive_sha256):p.error('full archive identity required')
    globals()[a.action](a)


if __name__=='__main__':main()
