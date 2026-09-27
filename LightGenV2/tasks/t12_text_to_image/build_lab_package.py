"""Export a versioned laboratory ZIP directly from Git, never dirty source files."""
import argparse
import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path
import torch
from .sealed_editor import build_sealed
from .audited_unified import architecture_report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--contract',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    root=Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],text=True).strip())
    subprocess.run(['git','diff','--quiet','HEAD'],cwd=root,check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    checkpoint=a.checkpoint.read_bytes();checkpoint_sha=hashlib.sha256(checkpoint).hexdigest()
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    model=build_sealed(saved);report=architecture_report(model)
    if report['counted_parameters']>20_000_000:raise ValueError('20M export budget exceeded')
    if model.bounded_amplitude!={'kind':'tanh','scale':.5}:raise ValueError('Amplitude contract changed')
    contract=json.loads(a.contract.read_text(encoding='utf-8-sig'))
    contract.update(checkpoint_sha256=checkpoint_sha,counted_parameters=report['counted_parameters'],
                    fusion_bounds=saved.get('fusion_bounds',dict(minimum=.4,maximum=.75)),
                    detector_correction=saved.get('detector_correction'),
                    bounded_amplitude=model.bounded_amplitude,source_commit=commit,
                    source_note='TRAIN adaptation; VAL-only selection; new weights require fresh CCD captures',
                    scope='new versioned candidate; cannot reuse CCDs from another weight',
                    selection=saved.get('channel_robust_training',{}))
    contract.pop('indices',None)
    names=subprocess.check_output(['git','ls-files'],cwd=root,text=True).splitlines()
    names=[n for n in names if n.startswith(('LightGenV2/','experiments/'))
           and Path(n).suffix in ('.py','.yaml','.yml','.json')
           and not any(s in n for s in ('/runs/','/reports/','/dataset/','/datasets/','/analysis_results/'))]
    archive=subprocess.check_output(['git','archive','--format=zip','HEAD','--',*names],cwd=root)
    records={}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.output,'w',compression=zipfile.ZIP_DEFLATED) as out:
        def put(name,content):
            out.writestr(name,content);records[name]=dict(bytes=len(content),sha256=hashlib.sha256(content).hexdigest())
        with zipfile.ZipFile(io.BytesIO(archive)) as source:
            for item in source.infolist():
                if not item.is_dir():put('source/'+item.filename,source.read(item.filename))
        put('assets/small.pt',checkpoint)
        put('assets/contract.json',json.dumps(contract,indent=2).encode())
        readme=f'''# T12 versioned physical robustness candidate
Source commit: {commit}
Checkpoint SHA256: {checkpoint_sha}
Parameters: {report['counted_parameters']}; frozen Qwen embeddings excluded by existing convention.
Amplitude is zero-preserving tanh(abs(E)/0.5), preserving complex phase. BMP uses round(255*a), no extra peak normalization.
Training channel perturbations are NOT enabled in deployment. Use complete six-stage physical replay.
Do not overwrite candidate_bounded or reuse its CCDs. Extract into a new candidate folder.
Reuse the existing read-only assets/datasets directory via directory junction, and existing optical hardware dependencies.
Set PYTHONPATH to this package's source. For a paired VAL96 capture, run lab_shs8um.run_layerwise with --split val --max-samples 96 --reuse a-new-empty-directory.
Source and weight integrity are recorded in manifest.json. Physical improvement remains unverified until fresh paired VAL/TEST captures.
'''
        put('README.md',readme.encode())
        out.writestr('manifest.json',json.dumps(dict(source_commit=commit,checkpoint_sha256=checkpoint_sha,files=records),indent=2))
    manifest=dict(zip_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),source_commit=commit,checkpoint_sha256=checkpoint_sha,
                  counted_parameters=report['counted_parameters'],filename=str(a.output),files=len(records))
    a.output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest),flush=True)


if __name__=='__main__':main()
