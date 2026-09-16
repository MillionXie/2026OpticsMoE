"""Build a standalone two-model EuroSAT handoff from the canonical pilot source."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

HERE=Path(__file__).resolve().parent
TRAINING_COMMIT='8c48e5caacb5cd2a8f82d18eaad41d5cb7a80d5f'
SPLIT_SHA='cfe8373dd33cc0fe64f083b9ca32377e767c21b078c3f1d91f2dacecc25cb776'


def digest(data):return hashlib.sha256(data).hexdigest()
def json_bytes(value):return (json.dumps(value,indent=2,ensure_ascii=False)+'\n').encode('utf-8')


def replace_once(text,old,new):
    assert text.count(old)==1,repr(old)
    return text.replace(old,new,1)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--variant',choices=['pure_optical','shared_frontend'],default='pure_optical')
    parser.add_argument('--split',type=Path)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--data',type=Path)
    parser.add_argument('--data-manifest',type=Path)
    parser.add_argument('--known-test-run',type=Path)
    args=parser.parse_args()
    if args.variant=='shared_frontend':
        from shared_frontend.package_builder import build
        build(args)
        return
    assert args.split is not None,'--split is required for the pure_optical package'
    assert not args.out.exists(),'Refusing to replace a release'
    assert digest(args.split.read_bytes())==SPLIT_SHA
    metadata=json.loads((args.run/'metadata.json').read_text())
    assert metadata['git_commit']==TRAINING_COMMIT
    assert json.loads((args.run/'independent_verification.json').read_text())['passed']
    pilot=HERE/'pure_optical';sources={}
    def read(path):
        data=path.read_bytes();sources[path.relative_to(HERE).as_posix()]=digest(data)
        return data.decode('utf-8').replace('\r\n','\n')
    files={}
    model=read(pilot/'models.py')
    for source_name in ['models.py','run.py','prepare.py','config.json']:
        assert digest((pilot/source_name).read_bytes())==metadata['source_sha256'][source_name] or digest((pilot/source_name).read_bytes().replace(b'\r\n',b'\n'))==metadata['source_sha256'][source_name]
    model=replace_once(model,"from pathlib import Path\nimport sys\n",'')
    model=replace_once(model,"ARCHIVE = Path(__file__).resolve().parents[1] / 'adrenal_softsign_code_export_20260915_145336/code'\nsys.path.insert(0, str(ARCHIVE))\nfrom optical_reference.optics import AngularSpectrumPropagator",'from optics import AngularSpectrumPropagator')
    files['models.py']=model.encode()
    run=read(pilot/'run.py')
    run=replace_once(run,'from models import PhaseOnly, objective, ARCHIVE','from models import PhaseOnly, objective')
    run=replace_once(run,"sources['archived_optics.py']=sha(ARCHIVE/'optical_reference/optics.py')","sources['optics.py']=sha(HERE/'optics.py')")
    run=replace_once(run,"subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip()","json.loads((HERE/'PROVENANCE.json').read_text())['export_commit']")
    run=replace_once(run,'results=[];initial_main=None','results=[]')
    for line in ["            hashes=tensors_sha(dict(model.named_parameters()))\n",
                 "            if architecture=='dynamic_four':initial_main={n:h for n,h in hashes.items() if n!='router_phase'}\n",
                 "            if architecture=='fixed_four':assert hashes==initial_main\n"]:
        run=replace_once(run,line,'')
    files['run.py']=run.encode()
    cfg=json.loads(read(pilot/'config.json'));cfg['architectures']=['dynamic_four','full_d2nn']
    files['config.json']=json_bytes(cfg)
    prepare=read(pilot/'prepare.py')
    prepare=replace_once(prepare,'import sys\n','')
    prepare=replace_once(prepare,"sys.path.insert(0, str(HERE.parents[0] / 'EuroSAT_MoE_D2NN/code'))\n",'')
    files['prepare.py']=prepare.encode()
    optics=read(HERE/'adrenal_softsign_code_export_20260915_145336/code/optical_reference/optics.py')
    node=next(x for x in ast.parse(optics).body if isinstance(x,ast.ClassDef) and x.name=='AngularSpectrumPropagator')
    files['optics.py']=('import math\nfrom typing import Tuple, Union\nimport torch\nfrom torch import nn\nGridSize=Union[int,Tuple[int,int]]\n\n'+ast.get_source_segment(optics,node)+'\n').encode()
    downloader=read(HERE/'EuroSAT_MoE_D2NN/code/download_archives.py')
    downloader=replace_once(downloader,"sys.path.insert(0,str(ROOT/'.netdeps'))\n",'')
    files['download_archives.py']=downloader.encode()
    files['README.md']=read(pilot/'HANDOFF_README.md').encode()
    files['requirements.txt']=b'numpy==1.26.4\npillow==12.2.0\nrasterio==1.4.4\npyproj==3.7.2\naffine==3.0.1\nrequests==2.34.2\n'
    files['data/SPLIT.json']=args.split.read_bytes()
    files['reference_data_manifest.json']=(args.run/'dataset_manifest.json').read_bytes()
    results=json.loads((args.run/'results.json').read_text())
    files['reported_results.json']=json_bytes([r for r in results if r['architecture'] in cfg['architectures']])
    files['training_environment.json']=json_bytes({k:metadata[k] for k in ['python','torch','gpu','environment','git_commit','data_sha256']})
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip()
    files['PROVENANCE.json']=json_bytes(dict(export_commit=commit,training_commit=TRAINING_COMMIT,
        canonical_source_sha256=sources,export_models=cfg['architectures'],builder_sha256=digest(Path(__file__).read_bytes()),
        modifications=['self-contained imports','extract identical ASM class','run two architectures only','Git-free provenance metadata','remove unused fixed-four smoke assertion'],
        results_origin='runs/simulation/pure_optical_20260916',split_sha256=SPLIT_SHA))
    for name,payload in files.items():
        if name.endswith('.py'):
            ast.parse(payload.decode(),filename=name)
            assert 'fixed_four' not in payload.decode()
    manifest={name:dict(bytes=len(data),sha256=digest(data)) for name,data in sorted(files.items())}
    files['MANIFEST.json']=json_bytes(dict(files=manifest,manifest_scope='all other ZIP files; manifest itself excluded'))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.out,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name,data in sorted(files.items()):
            info=zipfile.ZipInfo('eurosat_phase_only_moe_d2nn/'+name,date_time=(2026,9,16,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,data,compresslevel=9)
    args.out.with_suffix('.manifest.json').write_bytes(files['MANIFEST.json'])
    args.out.with_suffix('.sha256').write_text(digest(args.out.read_bytes())+'  '+args.out.name+'\n',encoding='utf-8')
    print(json.dumps(dict(zip=str(args.out),bytes=args.out.stat().st_size,sha256=digest(args.out.read_bytes()),files=len(files))))


if __name__=='__main__':main()
