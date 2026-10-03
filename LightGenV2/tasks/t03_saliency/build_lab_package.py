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
                        density_delta=float((density_from_logits(native)-density_from_logits(logits[i:i+1])).abs().max())
                        single_cc=float(independent_cc(density_from_logits(native).cpu().numpy(),data['density'].numpy())[0])
                        cc_delta=abs(single_cc-float(cc[i]))
                        if error>1e-4 or density_delta>1e-6 or cc_delta>1e-5 or tuple(tap.amplitudes)!=STAGES:
                            raise RuntimeError(f'Three-CCD replay audit failed: {error}, logits {single_delta}, density {density_delta}, CC {cc_delta}')
                        audits.append(dict(key=key,three_ccd_max_abs=error,single_vs_batch_max_abs=single_delta,stem_max_abs=delta,
                                           single_vs_batch_density_max_abs=density_delta,single_vs_batch_cc_abs=cc_delta))
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
    p.add_argument('--handoff-export',type=Path,help='Verified fixed export: add portable commands and explicit hardware/training limits in a NEW ZIP')
    p.add_argument('--archive',type=Path)
    p.add_argument('--atomic-writer-update',action='store_true',help='Small control-only ZIP for already deployed releases')
    p.add_argument('--data-root',type=Path);p.add_argument('--cache-dir',type=Path)
    p.add_argument('--max-fields',type=int,default=0);p.add_argument('--batch-size',type=int,default=48)
    p.add_argument('--device',default='cuda');a=p.parse_args()
    if a.handoff_export:
        if not a.archive:p.error('--handoff-export requires a NEW --archive')
        handoff_export(a.handoff_export,a.archive)
    elif a.atomic_writer_update:
        if not a.archive:p.error('A new --archive ZIP is required')
        root=Path(__file__).resolve().parents[3]
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
        rel='LightGenV2/tasks/t06_video_quality_assessment/lab_runtime.py'
        base='3dabf3bd33f3a16f3666c03491e4e6bb29a3dc80'
        old=subprocess.check_output(['git','show',base+':'+rel],cwd=root)
        new=subprocess.check_output(['git','show',commit+':'+rel],cwd=root)
        manifest=dict(source_commit=commit,base_commit=base,change='bounded retry for Windows atomic JSON replacement only; inference/hardware unchanged',
                      files={'runtime/'+rel:dict(base_sha256=hashlib.sha256(old).hexdigest(),sha256=hashlib.sha256(new).hexdigest())})
        a.archive.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(a.archive,'x',zipfile.ZIP_DEFLATED) as z:
            z.writestr('runtime/'+rel,new);z.writestr('runtime_update.json',json.dumps(manifest,indent=2))
        write(a.archive.with_suffix('.delivery.json'),dict(zip=str(a.archive.resolve()),sha256=sha(a.archive),bytes=a.archive.stat().st_size,source_commit=commit))
    elif a.refresh_runtime:
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

def handoff_export(out,archive):
    """Repackage exact cached inference without touching the source export."""
    import tempfile
    out=out.resolve();archive=archive.resolve()
    if archive.exists():raise FileExistsError(archive)
    root=Path(__file__).resolve().parents[3]
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    release=json.loads((out/'release.json').read_text())
    if release['checkpoint_sha256']!=CHECKPOINT_SHA or sha(out/'weights/best_checkpoint.pt')!=CHECKPOINT_SHA:
        raise ValueError('Wrong selected checkpoint')
    if release['test_samples']!=5000 or release['test_ids_sha256']!=TEST_IDS_SHA:
        raise ValueError('Not the complete verified test export')
    manifest=json.loads((out/'SHA256.json').read_text())
    for name,digest in manifest.items():
        path=(out/name).resolve()
        if not path.is_relative_to(out) or sha(path)!=digest:raise ValueError('Export integrity failed: '+name)
    base=release['source_commit']
    additions={}
    def git_bytes(ref,name):return subprocess.check_output(['git','show',ref+':'+name],cwd=root)
    additions['handoff.py']=git_bytes(commit,'LightGenV2/tasks/t03_saliency/handoff_cli.py')
    additions['00_START_HERE.md']=git_bytes(commit,'LightGenV2/tasks/t03_saliency/HANDOFF_08625.md')
    additions['meadowlark.py']=git_bytes(commit,'LightGenV2/tasks/t03_saliency/meadowlark_cli.py')
    additions['COMMAND_MEADOWLARK.md']=git_bytes(commit,'LightGenV2/tasks/t03_saliency/COMMAND_MEADOWLARK.md')
    additions['requirements-reference.txt']=git_bytes(base,'ABO_Lab_SHS_8um/requirements-gpu-tested.txt')
    paths=subprocess.check_output(['git','ls-tree','-r','--name-only',base,'LightGenV2','experiments'],cwd=root,text=True).splitlines()
    for rel in paths:
        if any(x in Path(rel).parts for x in ('runs','data','vendor_sdk','releases')):continue
        if rel.endswith(('.yaml','.yml')):additions['source_configs/'+rel]=git_bytes(base,rel)
        elif rel in ('LightGenV2/AI_RULES.md','LightGenV2/tasks/t03_saliency/reports/reproduction/README.md',
                     'LightGenV2/tasks/t03_saliency/reports/reproduction/CROSS_SAMPLE_BALANCE_20260913.md'):
            additions['source_notes/'+rel]=git_bytes(base,rel)
    # Apply only the already audited Windows atomic-write retry patch; preserve
    # the inference model/runtime of the verified export byte-for-byte.
    writer='runtime/LightGenV2/tasks/t06_video_quality_assessment/lab_runtime.py'
    additions[writer]=git_bytes('55df4f53',writer.removeprefix('runtime/'))
    information=dict(package_kind='fixed_weight_experiment_handoff',source_commit=commit,
                     original_model_runtime_commit=base,checkpoint_sha256=CHECKPOINT_SHA,
                     reference_cc=release['simulation_cc_float64'],test_fields=5000,train_fields=0,
                     hardware_binding='meadowlark.py: Meadowlark17/manual phase8/TUCam; existing calibrated native SDK config required',
                     meadowlark_adapter_validation='mock regression and offline replay; real hardware validation pending on recipient Windows machine',
                     one_command_measured_finetuning_included=False,
                     control_patch_commit='55df4f53',hardware_settings_from_other_lab_included=False)
    additions['handoff.json']=(json.dumps(information,indent=2)+'\n').encode()
    archive.parent.mkdir(parents=True,exist_ok=True)
    # Test the delivered CLI in an isolated temporary tree (no source mutation).
    with tempfile.TemporaryDirectory(prefix='salicon_handoff_',dir=archive.parent) as tmp:
        staged=Path(tmp)
        for name in manifest:
            if '__pycache__' in Path(name).parts:continue
            dest=staged/name;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(out/name,dest)
        for name,value in additions.items():
            dest=staged/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(value)
        write(staged/'SHA256.json',manifest)  # CLI reads identity before exporting.
        subprocess.run([sys.executable,'-I',str(staged/'meadowlark.py'),'--help'],cwd=staged,check=True)
        for profile in ('meadowlark17','shs8'):
            subprocess.run([sys.executable,'-I',str(staged/'handoff.py'),'export-reference-bmp',
                            '--profile',profile,'--fields','4','--device','cpu'],cwd=staged,check=True)
        actual={p.relative_to(staged).as_posix():sha(p) for p in staged.rglob('*')
                if p.is_file() and p.name!='SHA256.json' and '__pycache__' not in p.parts}
        write(staged/'SHA256.json',actual)
        subprocess.run([sys.executable,'-I',str(staged/'handoff.py'),'verify'],cwd=staged,check=True)
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
            for name in [*actual,'SHA256.json']:z.write(staged/name,name)
    write(archive.with_suffix('.delivery.json'),dict(zip=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,
          source_commit=commit,checkpoint_sha256=CHECKPOINT_SHA,simulation_cc_float64=release['simulation_cc_float64']))
    print('HANDOFF_READY',archive,flush=True)


if __name__=='__main__':main()
