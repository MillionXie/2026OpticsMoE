"""Export exact frozen-stem caches, pinned weights and a SHA-verified SHS release."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
import numpy as np
import torch
from .lab_runtime import (CHECKPOINT_SHA, REFERENCE_CC, TEST_IDS_SHA, STAGES,
                          CachedStudent, load_model, phase_planes, replay, write, sha)
from .settings import load_settings
from .modeling import build_student, load_vision_backbone
from .recheck_aligned import recheck_loader
from .reproduce_baseline import independent_cc
from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency.datasets import prepare_salicon
from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency.modeling import preprocess_vision
from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency.objectives import density_from_logits, SaliencyAccumulator


def build(a):
    root = Path(__file__).resolve().parents[3]
    out = a.output.resolve()
    if out.exists():
        raise FileExistsError('New output required; existing release will not be overwritten')
    if sha(a.checkpoint) != CHECKPOINT_SHA:
        raise ValueError('Wrong checkpoint')
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    if subprocess.check_output(['git','diff','HEAD','--','LightGenV2/tasks/t03_saliency'],cwd=root):
        raise ValueError('Commit/push runtime before building')
    torch.manual_seed(42)
    s = load_settings(a.config)
    s.local_files_only, s.download, s.inference_batch_size, s.num_workers = True, False, a.batch_size, 2
    # Data and HF locations are machine-local, not changed model parameters.
    if a.data_root: s.data_root = a.data_root.resolve()
    if a.cache_dir: s.cache_dir = a.cache_dir.resolve()
    s.output_dir = out/'export_evidence'
    out.mkdir(parents=True)
    bundle = prepare_salicon(s, persist=True)
    loader, expected = recheck_loader(bundle, s, 'test')
    loaded = load_vision_backbone(s, torch.device(a.device))
    student = build_student(loaded, s)
    payload = torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    student.core.load_state_dict(payload['core'],strict=True)
    student.head.load_state_dict(payload['saliency_head'],strict=True)
    student.core.set_phase_dropout_active(False)
    student.eval()
    (out/'weights').mkdir()
    shutil.copy2(a.checkpoint,out/'weights/best_checkpoint.pt')
    write(out/'settings.json',vars(s))
    cached = load_model(out,a.device)
    (out/'inputs').mkdir(); (out/'phases').mkdir()
    for stage, plane in phase_planes(cached).items(): np.save(out/'phases'/f'{stage}.npy',plane)
    captured = {}
    original_groups = student.core.forward_groups
    def capture_groups(groups, spatial_shapes):
        captured['tokens'] = torch.cat(groups).detach().cpu().clone()
        return original_groups(groups, spatial_shapes)
    student.core.forward_groups = capture_groups
    fields, rows, audits = [], [], []
    accumulator = SaliencyAccumulator()
    try:
        with torch.inference_mode():
            for batch in loader:
                inputs = preprocess_vision(loaded.processor,batch['images'],loaded.device)
                logits = student(inputs['pixel_values'], inputs['image_grid_thw'])[0]
                grid = inputs['image_grid_thw'].cpu()
                cache_batch = dict(tokens=captured['tokens'],grid=grid)
                repeated = cached(cache_batch)
                delta = float((repeated-logits).abs().max())
                if delta > 1e-5: raise RuntimeError(f'Frozen-stem replay mismatch {delta}')
                cc = independent_cc(density_from_logits(logits).cpu().numpy(),batch['density'].numpy())
                accumulator.update(logits,batch['density'].to(a.device),batch['fixation'].to(a.device))
                lengths = [int(row.prod()) for row in grid]
                groups = list(captured['tokens'].split(lengths))
                for i, sid in enumerate(batch['sample_ids']):
                    if a.max_fields and len(fields) >= a.max_fields: break
                    key = f'field_{len(fields):05d}'
                    data = dict(tokens=groups[i].clone(),grid=grid[i:i+1].clone(),
                                density=batch['density'][i:i+1].clone(),fixation=batch['fixation'][i:i+1].clone(),
                                sample_id=sid)
                    if len(fields)<8:
                        native,tap = replay(cached,data)
                        restored,_ = replay(cached,data,tap.detectors)
                        error = float((native-restored).abs().max())
                        single_delta = float((native-logits[i:i+1]).abs().max())
                        if error>1e-4 or single_delta>2e-4 or tuple(tap.amplitudes)!=STAGES:
                            raise RuntimeError(f'Three-CCD replay audit failed: {error}, {single_delta}')
                        audits.append(dict(key=key,three_ccd_max_abs=error,single_vs_batch_max_abs=single_delta,stem_max_abs=delta))
                        from PIL import Image
                        folder=out/'previews';folder.mkdir(exist_ok=True)
                        batch['images'][i].save(folder/(key+'_input.png'))
                    path=out/'inputs'/(key+'.pt');torch.save(data,path)
                    fields.append(dict(key=key,file=path.relative_to(out).as_posix(),sha256=sha(path),sample_id=sid,simulation_cc=float(cc[i])))
                    rows.append(float(cc[i]))
                print('EXPORTED',len(fields),'/',a.max_fields or expected,'CC',np.mean(rows),flush=True)
                if a.max_fields and len(fields)>=a.max_fields: break
    finally:
        student.core.forward_groups=original_groups
        student.restore_native()
    full=len(fields)==expected
    ids_sha=hashlib.sha256('\n'.join(f['sample_id'] for f in fields).encode()).hexdigest()
    if full and (ids_sha!=TEST_IDS_SHA or abs(np.mean(rows)-REFERENCE_CC)>2e-5):
        raise RuntimeError('Full split identities or fixed-weight simulation CC differ')
    # Git-tracked Python only: preserve dependency imports but no runs/caches/data.
    paths=subprocess.check_output(['git','ls-files','LightGenV2','experiments'],cwd=root,text=True).splitlines()
    for rel in paths:
        if not rel.endswith('.py') or any(x in Path(rel).parts for x in ('runs','data','vendor_sdk')): continue
        dst=out/'runtime'/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        dst.write_bytes(subprocess.check_output(['git','show',commit+':'+rel],cwd=root))
    entry="from pathlib import Path\nimport sys\nsys.path.insert(0,str(Path(__file__).resolve().parent/'runtime'))\nfrom LightGenV2.tasks.t03_saliency.lab_bench import main\nif __name__=='__main__': main()\n"
    (out/'run.py').write_text(entry,encoding='utf-8')
    shutil.copy2(root/'LightGenV2/tasks/t03_saliency/COMMAND_SHS.md',out/'COMMAND_SHS.md')
    release=dict(target='salicon',source_commit=commit,checkpoint_sha256=CHECKPOINT_SHA,reference_cc=REFERENCE_CC,
                 stages=list(STAGES),full_test=full,test_samples=len(fields),test_ids_sha256=ids_sha,
                 simulation_cc_float64=float(np.mean(rows)),simulation_metrics=accumulator.compute() if full else None,
                 frozen_stem_source='Qwen patch_embed+position, before block0; exact dtype retained',
                 replay_audits=audits,fields=fields)
    write(out/'release.json',release)
    subprocess.run([sys.executable,'-I',str(out/'run.py'),'--help'],cwd=out,check=True)
    write(out/'SHA256.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()})
    archive=out.with_suffix('.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in out.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(out).as_posix())
    write(out.with_suffix('.delivery.json'),dict(zip=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,
                                              source_commit=commit,simulation_cc_float64=release['simulation_cc_float64']))
    print('DELIVERED',archive,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('config','checkpoint','output'):p.add_argument('--'+key,type=Path)
    p.add_argument('--refresh-runtime',type=Path,help='Verified completed export: repackage control-code-only update without regenerating caches')
    p.add_argument('--archive',type=Path)
    p.add_argument('--data-root',type=Path);p.add_argument('--cache-dir',type=Path)
    p.add_argument('--max-fields',type=int,default=0);p.add_argument('--batch-size',type=int,default=48)
    p.add_argument('--device',default='cuda');a=p.parse_args()
    if a.refresh_runtime:
        if not a.archive:p.error('--refresh-runtime requires a NEW --archive')
        refresh_runtime(a.refresh_runtime,a.archive)
    else:
        if not all((a.config,a.checkpoint,a.output)):p.error('config/checkpoint/output required')
        build(a)


def refresh_runtime(out,archive):
    from .lab_runtime import read
    out=out.resolve();archive=archive.resolve()
    if archive.exists():raise FileExistsError(archive)
    manifest=read(out/'SHA256.json')
    for name,digest in manifest.items():
        if sha(out/name)!=digest:raise ValueError('Existing export changed: '+name)
    root=Path(__file__).resolve().parents[3]
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    # This whitelist changes only capture auditing/supervision, never model/cache.
    for name in ('lab_bench.py','lab_supervise.py'):
        rel='LightGenV2/tasks/t03_saliency/'+name
        dest=out/'runtime'/rel
        dest.write_bytes(subprocess.check_output(['git','show',commit+':'+rel],cwd=root))
    release=read(out/'release.json');release['control_runtime_commit']=commit
    write(out/'release.json',release)
    code="""import sys,json,torch
from pathlib import Path
p=Path(sys.argv[1]);sys.path.insert(0,str(p/'runtime'))
from LightGenV2.tasks.t03_saliency.lab_runtime import load_model,replay,read
from LightGenV2.tasks.t03_saliency.reproduce_baseline import independent_cc
from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency.objectives import density_from_logits
torch.set_num_threads(4)
r=read(p/'release.json');item=r['fields'][0]
m=load_model(p,'cpu');b=torch.load(p/item['file'],map_location='cpu',weights_only=False)
y,tap=replay(m,b);cc=float(independent_cc(density_from_logits(y).numpy(),b['density'].numpy())[0])
assert abs(cc-item['simulation_cc'])<1e-4,(cc,item['simulation_cc'])
print('ISOLATED_CPU_PACKAGE_REPLAY',cc,flush=True)
"""
    subprocess.run([sys.executable,'-I','-c',code,str(out)],cwd=out,check=True)
    write(out/'SHA256.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p.name!='SHA256.json'})
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in out.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(out).as_posix())
    write(archive.with_suffix('.delivery.json'),dict(zip=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,
           source_commit=release['source_commit'],control_runtime_commit=commit,simulation_cc_float64=release['simulation_cc_float64']))
    print('REPACKAGED',archive,flush=True)

if __name__=='__main__':main()
