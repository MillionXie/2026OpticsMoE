"""Build the SDK-free F9 interface kit from published Git, not the old17M model."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile
from .scene_lab_native import F9,F6,E662
from .scene_lab_protocol import sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile',choices=['scene-f9-interface'],required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--asset-manifest',type=Path,required=True)
    p.add_argument('--asset-manifest-sha256',required=True);p.add_argument('--source-commit',required=True)
    a=p.parse_args();task=Path(__file__).resolve().parent;root=task.parents[2]
    if a.output.exists():raise FileExistsError(a.output)
    if sha(a.asset_manifest)!=a.asset_manifest_sha256:raise ValueError('Unverified persistent asset closure')
    assets=json.loads(a.asset_manifest.read_text());assert assets['complete']and assets['schema']=='scene_F9_persistent_asset_closure_v1'
    pin=subprocess.check_output(['git','rev-parse',a.source_commit],cwd=root,text=True).strip()
    # Sources always come from Git blobs. Dirty/untracked files are never exported.
    names=subprocess.check_output(['git','ls-tree','-r','--name-only',pin],cwd=root,text=True).splitlines()
    names=[n for n in names if (n.startswith(('LightGenV2/tasks/t12_text_to_image/','LightGenV2/tasks/t01_object_retrieval/','LightGenV2/common/','experiments/qwen'))or n in('LightGenV2/__init__.py','LightGenV2/tasks/__init__.py','experiments/__init__.py'))and Path(n).suffix in('.py','.json','.yaml','.yml')
           and not any(x in n for x in('/runs/','/reports/','/datasets/','/dataset/','/analysis_results/'))
           and not any(x in Path(n).name.lower()for x in('local.json','paths.local','secret','credential'))]
    guide='LightGenV2/tasks/t12_text_to_image/scene_lab_README.md'
    subprocess.check_output(['git','cat-file','-e',pin+':'+guide],cwd=root)
    names.append(guide)
    for required in('scene_lab_protocol.py','scene_lab_native.py','scene_lab_entry.py','scene_lab_session_template.json'):
        if 'LightGenV2/tasks/t12_text_to_image/'+required not in names:raise ValueError('Commit lacks the new interface')
    members={}
    archive=subprocess.check_output(['git','archive','--format=zip',pin,'--',*names],cwd=root)
    with zipfile.ZipFile(io.BytesIO(archive))as z:
        for name in names:members['source/'+name]=z.read(name)
    # Whitelisted task weights; no CCD, feature cache, targets, masks or TEST requests.
    for name,expected in [('scene_two_sentence_qwen_paired_20261029',E662),('qwen3_scene_LG_frontend_20261029',F6),('scene_natural_single_EG_joint12_training_20261030',F9)]:
        file=task/'runs/simulation'/name/'best_checkpoint.pt'
        if sha(file)!=expected:raise ValueError('Checkpoint identity mismatch')
        members['assets/'+name+'/best_checkpoint.pt']=file.read_bytes()
    members['assets/persistent_asset_manifest.json']=a.asset_manifest.read_bytes()
    note=('F9 SDK-free interface kit, NOT a runnable certified bench release.\n'
          'See source/LightGenV2/tasks/t12_text_to_image/scene_lab_README.md.\n'
          'Physical capture disabled; unreviewed session template rejects initialization.\n'
          'E662 joint_image_context=false; F9=true. Restore exact checkpoint head strictly.\n'
          'Base14.58GB assets are external, not copied or counted as deployed parameters.\n'
          'No TRAIN/TEST images, CCDs, calibration, SDK or credentials included.\n')
    members['README.txt']=note.encode()
    manifest=dict(schema='scene_F9_interface_package_v1',source_commit=pin,physical_capture_ready=False,
                  frozen_TEST_gate_passed=False,SDK_included=False,session_calibration_included=False,
                  external_assets_sha256=a.asset_manifest_sha256,
                  files={n:dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())for n,b in members.items()})
    members['manifest.json']=json.dumps(manifest,indent=2).encode();a.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.output,'w',zipfile.ZIP_DEFLATED)as z:
        for name,data in members.items():z.writestr(name,data)
    with zipfile.ZipFile(a.output)as z:
        if z.testzip()is not None:raise ValueError('Invalid ZIP')
    a.output.with_suffix('.zip.sha256').write_text(sha(a.output)+'  '+a.output.name+'\n')
    print(json.dumps(dict(output=str(a.output),bytes=a.output.stat().st_size,sha256=sha(a.output),source_commit=pin,physical_capture_ready=False)))
