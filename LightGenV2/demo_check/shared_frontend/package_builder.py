"""Pinned, standalone export of the requested 77.20/73.65 validation baseline."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

HERE=Path(__file__).resolve().parent
TASK=HERE.parent
TRAINING='8cb125cfaa4829e7714b5b23389f0c96e4844a72'
EXPORT_SOURCE='cc4a0f17'


def digest(data):return hashlib.sha256(data).hexdigest()
def js(value):return (json.dumps(value,indent=2,ensure_ascii=False)+'\n').encode()
def replace(s,a,b):assert s.count(a)==1,repr(a);return s.replace(a,b,1)


def build(args):
    assert not args.out.exists(),'Refusing to overwrite an existing package'
    assert args.data is not None and args.data_manifest is not None
    metadata=json.loads((args.run/'metadata.json').read_text());assert metadata['git_commit']==TRAINING
    assert json.loads((args.run/'independent_verification.json').read_text())['passed']
    assert digest(args.data.read_bytes())==metadata['data_sha256']
    assert digest(args.data_manifest.read_bytes())==metadata['data_manifest_sha256']
    sources={};files={}
    def source(path):
        data=subprocess.check_output(['git','show',EXPORT_SOURCE+':LightGenV2/demo_check/'+path],cwd=TASK)
        sources[path]=digest(data);return data.decode()
    def extract(text,names):
        tree=ast.parse(text);segments=[];lines=text.splitlines(True)
        for name in names:
            node=next(n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name)
            first=min([node.lineno]+[d.lineno for d in node.decorator_list])
            segments.append(''.join(lines[first-1:node.end_lineno]))
        return '\n\n'.join(segments)+'\n'
    optical=source('pure_optical/models.py')
    files['optical_model.py']=('import torch\nfrom torch import nn\nfrom torch.nn import functional as F\nfrom optics import AngularSpectrumPropagator\n\n'+extract(optical,['encode','PhaseOnly','objective'])).encode()
    optics=source('adrenal_softsign_code_export_20260915_145336/code/optical_reference/optics.py')
    files['optics.py']=('import math\nfrom typing import Tuple,Union\nimport torch\nfrom torch import nn\nGridSize=Union[int,Tuple[int,int]]\n\n'+extract(optics,['AngularSpectrumPropagator'])).encode()
    electronic=source('frozen_electronic/model.py')
    files['cnn.py']=('import torch\nfrom torch import nn\n\n'+extract(electronic,['Electronic'])).encode()
    frontend=source('shared_frontend/model.py')
    files['models.py']=('import torch\nfrom torch import nn\nimport cnn\nfrom optical_model import PhaseOnly\n\n'+extract(frontend,['SharedFrontend','encode_features','FrontendOptics'])).encode()
    utilities=source('pure_optical/run.py')
    files['utils.py']=('import hashlib,json\nfrom pathlib import Path\nimport numpy as np\nimport torch\nfrom optical_model import objective\n\n'+extract(utilities,['save','sha','tensors_sha','batches','evaluate'])).encode()
    runner=source('shared_frontend/run.py');training=extract(runner,['write_predictions','train'])
    training=replace(training,'def train(frontend,cfg,optical_cfg,arrays,train_data,val,out):','def train(frontend,cfg,optical_cfg,arrays,train_data,val,out,resume=None):')
    training=replace(training,'        initial=tensors_sha(dict(model.optical.named_parameters()))',
        "        start_epoch=0;resume_state=None\n        if resume is not None:\n            resume_state=torch.load(resume/architecture/'last_checkpoint.pt',map_location='cpu',weights_only=False)\n            assert resume_state['frontend_tensors_sha256']==frozen\n            model.optical.load_state_dict(resume_state['model']);start_epoch=resume_state['epoch']\n            assert start_epoch<cfg['epochs']\n        initial=tensors_sha(dict(model.optical.named_parameters()))")
    training=replace(training,"        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,cfg['epochs'],eta_min=cfg['learning_rate']*cfg['minimum_lr_ratio'])",
        "        if resume_state is not None:\n            optimizer.load_state_dict(resume_state['optimizer'])\n            for group in optimizer.param_groups:group['lr']=cfg['learning_rate']\n        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,cfg['epochs']-start_epoch,eta_min=cfg['learning_rate']*cfg['minimum_lr_ratio'])")
    training=replace(training,"for epoch in range(1,cfg['epochs']+1):","for epoch in range(start_epoch+1,cfg['epochs']+1):")
    files['training.py']=('import csv,hashlib,json,time\nimport numpy as np\nimport torch\nimport utils as base\nfrom utils import save,sha,tensors_sha,batches\nfrom optical_model import objective\nfrom models import FrontendOptics\n\n'+training).encode()
    files['run.py']=(HERE/'handoff_entry.py').read_bytes()
    files['README.md']=(HERE/'HANDOFF_README.md').read_bytes()
    files['config.json']=js(metadata['config']);ocfg=metadata['optical_config'].copy();ocfg['architectures']=metadata['config']['architectures'];files['optical_config.json']=js(ocfg)
    files['requirements.txt']=b'numpy==1.26.4\n# Install torch==2.6.0 with CUDA separately; see README.\n'
    files['data/data.npz']=args.data.read_bytes();files['data/manifest.json']=args.data_manifest.read_bytes()
    files['data/classes.json']=js(['AnnualCrop','Forest','HerbaceousVegetation','Highway','Industrial','Pasture','PermanentCrop','Residential','River','SeaLake'])
    expected={'dynamic_four':(.772,20),'full_d2nn':(.7365,15)};best_hashes={};last_hashes={}
    for architecture,(score,epoch) in expected.items():
        summary=json.loads((args.run/architecture/'summary.json').read_text())
        assert summary['validation']['accuracy']==score and summary['selected_epoch']==epoch
        for checkpoint in ['best_checkpoint.pt','last_checkpoint.pt']:
            content=(args.run/architecture/checkpoint).read_bytes();files['checkpoints/'+architecture+'/'+checkpoint]=content
            if checkpoint.startswith('best'):assert digest(content)==summary['checkpoint_sha256'];best_hashes[architecture]=digest(content)
            else:last_hashes[architecture]=digest(content)
        for name in ['summary.json','history.json','validation_predictions.csv','initial_validation.json']:
            files['reference/'+architecture+'/'+name]=(args.run/architecture/name).read_bytes()
    frontend_bytes=(args.run/'frontend/best_checkpoint.pt').read_bytes();files['checkpoints/frontend/best_checkpoint.pt']=frontend_bytes
    for name in ['metadata.json','results.json','fairness.json','independent_verification.json']:
        files['reference/'+name]=(args.run/name).read_bytes()
    for name in ['validation_curves.png','training_diagnostics.png']:
        if (args.run/name).exists():files['reference/'+name]=(args.run/name).read_bytes()
    if args.known_test_run is not None:
        test=json.loads((args.known_test_run/'results.json').read_text())
        files['reference/known_test_results.json']=js([dict(architecture=x['architecture'],checkpoint_sha256=x['checkpoint_sha256'],test=x['test'],scope='2000 images from the original spatial test split; previously evaluated, not an untouched future test set') for x in test if x['label']=='original'])
    for p in (TASK/'EuroSAT_MoE_D2NN/code/licenses').glob('*'):
        if p.suffix in ['.txt','.md']:files['licenses/'+p.name]=p.read_bytes()
    for name in ['DATA_LICENSES.json','DATA_PREPROCESSING.json']:
        files['data/'+name]=(TASK/'EuroSAT_MoE_D2NN/code'/name).read_bytes()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=TASK,text=True).strip()
    files['PROVENANCE.json']=js(dict(export_commit=commit,training_commit=TRAINING,canonical_export_source_commit=EXPORT_SOURCE,
        source_run='eurosat_shared_frontend_20260916',canonical_source_sha256=sources,data_sha256=metadata['data_sha256'],
        frontend_checkpoint_sha256=digest(frontend_bytes),best_checkpoint_sha256=best_hashes,last_checkpoint_sha256=last_hashes,
        changes=['self-contained module imports; model and propagation bodies copied from pinned source',
                 'Git-free standalone entry; load the supplied frozen frontend directly',
                 'optional last-checkpoint resume preserving Adam; original 20-epoch fresh-phase path unchanged'],
        builder_sha256=digest(Path(__file__).read_bytes())))
    files['verify_package.py']=b'''from pathlib import Path
import hashlib,json
r=Path(__file__).resolve().parent
m=json.loads((r/'MANIFEST.json').read_text())['files']
for n,v in m.items():
 p=r/n
 assert p.stat().st_size==v['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==v['sha256'],n
print('Verified',len(m),'files')
'''
    for name,data in files.items():
        if name.endswith('.py'):ast.parse(data.decode(),filename=name)
    files['MANIFEST.json']=js(dict(files={n:dict(bytes=len(b),sha256=digest(b)) for n,b in sorted(files.items())},scope='all other package files'))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.out,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,data in sorted(files.items()):
            info=zipfile.ZipInfo('eurosat_shared_frontend_moe_d2nn/'+name,date_time=(2026,9,16,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,data,compresslevel=6)
    args.out.with_suffix('.manifest.json').write_bytes(files['MANIFEST.json'])
    args.out.with_suffix('.sha256').write_text(digest(args.out.read_bytes())+'  '+args.out.name+'\n')
    print(json.dumps(dict(path=str(args.out),bytes=args.out.stat().st_size,files=len(files),sha256=digest(args.out.read_bytes()))))
