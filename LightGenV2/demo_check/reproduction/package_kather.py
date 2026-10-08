"""Task builder backend: immutable source files, four 6-layer reference weights."""
import hashlib,json,subprocess,zipfile
from pathlib import Path

def build_code(a):
    """Code-only handoff while retraining is active; never imply selected results."""
    task=Path(__file__).resolve().parents[1];assert not a.out.exists()
    meta=json.loads((a.run/'metadata.json').read_text());files={}
    def add(p,name=None):files[name or p.relative_to(task).as_posix()]=p.read_bytes()
    for relative,digest in meta['sources'].items():
        p=task/relative;assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,relative;add(p)
    for name in ['handoff_kather.py','prepare_kather_handoff.py','package_kather.py','plot_kather_architecture.py']:
        add(task/'reproduction'/name)
    add(task/'build_lab_package.py')
    cfg=json.loads((task/'reproduction/bloodmnist_profile.json').read_text())
    spec=json.loads((task/'reproduction/kather2016_profiles.json').read_text())
    cfg.update(spec['candidates']['base']);cfg.pop('url');cfg.pop('md5')
    cfg.update(dataset=spec['dataset'],classes=spec['classes'],scope=spec['scope'],encoding=spec['encoding'],deduplication=spec['data_split'])
    for name,values in meta['specification']['candidates'].items():
        c=dict(cfg,**values);assert c['expert_input_coverage']=='full'
        files['configs/'+name+'.json']=(json.dumps(c,indent=2)+'\n').encode()
    add(a.data_manifest,'data_manifest.json')
    add(task/'reproduction/KATHER_CODE_HANDOFF.md','README.md')
    add(task/'reports/reproduction/KATHER_FOLLOWUP_20260917.md','ARCHITECTURE_AND_PROTOCOL.md')
    add(a.run/'metadata.json','provenance/training_metadata.json')
    add(a.run/'status.json','provenance/training_status_at_export.json')
    smoke=task/'runs/smoke/kather_coverage_20260917'
    for name in ['smoke.json','coverage_smoke.json']:add(smoke/name,'verification/'+name)
    import importlib.metadata
    versions={n:importlib.metadata.version(n) for n in ['torch','numpy','scikit-learn','PyYAML','Pillow','pyarrow','matplotlib']}
    files['requirements.txt']=''.join(n+'=='+v.split('+')[0]+'\n' for n,v in versions.items()).encode()
    files['PROVENANCE.json']=json.dumps(dict(export_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),training_sources=meta['sources'],data_sha256=meta['data_sha256'],environment_packages=versions,scope='Code-only handoff: full-expert MoE and full-aperture D2NN, OEO on/off, 2/4/6 layers. No selected configuration, final metrics or weights are included. Training remains active.'),indent=2).encode()
    files['verify_manifest.py']=b"import pathlib,json,hashlib\nr=pathlib.Path(__file__).resolve().parent\nm=json.loads((r/'MANIFEST.json').read_text())['files']\nfor n,d in m.items():\n p=r/n;assert p.stat().st_size==d['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==d['sha256'],n\nprint('Verified',len(m),'files')\n"
    manifest={n:dict(bytes=len(v),sha256=hashlib.sha256(v).hexdigest()) for n,v in sorted(files.items())}
    files['MANIFEST.json']=json.dumps(dict(files=manifest),indent=2).encode()
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.out,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for n,v in sorted(files.items()):z.writestr('kather_coverage_code/'+n,v)
    a.out.with_suffix('.manifest.json').write_bytes(files['MANIFEST.json'])
    a.out.with_suffix('.sha256').write_text(hashlib.sha256(a.out.read_bytes()).hexdigest()+'  '+a.out.name+'\n')
    print(json.dumps(dict(zip=str(a.out),files=len(files),bytes=a.out.stat().st_size)))

def build(a):
    task=Path(__file__).resolve().parents[1]
    assert not a.out.exists()
    lock=json.loads((a.run/'selection_lock.json').read_text());results=json.loads((a.run/'results.json').read_text())
    assert json.loads((a.run/'status.json').read_text())['state']=='complete'
    files={}
    def add(p,name=None):files[name or p.relative_to(task).as_posix()]=p.read_bytes()
    for relative,digest in lock['sources'].items():
        p=task/relative;assert hashlib.sha256(p.read_bytes()).hexdigest()==digest;add(p)
    for name in ['handoff_kather.py','prepare_kather_handoff.py','verify_bloodmnist.py','analyze_oeo_suite.py','package_kather.py']:add(task/'reproduction'/name)
    add(task/'build_lab_package.py')
    for p in (task/'adrenal_softsign_code_export_20260915_145336/code').rglob('*.json'):
        if p.parent.name=='code':add(p)
    for e in lock['entries']:
        x=e['result']
        if x['depth']!=6 or x['seed']!=17:continue
        folder=Path(e['folder']);p=folder/'best_checkpoint.pt';assert hashlib.sha256(p.read_bytes()).hexdigest()==x['checkpoint_sha256'];add(p,'reference_weights/'+x['name']+'.pt')
        for n in ['summary.json','history.json','val_predictions.csv']:add(folder/n,'reference_results/'+x['name']+'/'+n)
    for n in ['selection_lock.json','results.json','validation_replay.json','metadata.json']:add(a.run/n,'reference_results/'+n)
    add(a.data_manifest,'data_manifest.json')
    files['selected_config.json']=(json.dumps(lock['config'],indent=2)+'\n').encode()
    add(task/'reproduction/KATHER_HANDOFF.md','README.md')
    for name in ['OEO_METHODS_20260917.md','KATHER_FOLLOWUP_20260917.md']:
        p=task/'reports/reproduction'/name
        if p.exists():add(p)
    files['requirements.txt']=b'torch==2.6.0\nnumpy==1.26.4\nscikit-learn==1.6.1\nPyYAML\nPillow\npyarrow\nmatplotlib\n'
    files['PROVENANCE.json']=json.dumps(dict(export_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),training_sources=lock['sources'],data_sha256=lock['data_sha256'],selection_lock_sha256=hashlib.sha256((a.run/'selection_lock.json').read_bytes()).hexdigest(),scope='Four arms, main depth6 seed17 reference weights; all36 numeric results. Source files are unchanged; use handoff_kather.py outside Git.'),indent=2).encode()
    manifest={n:dict(bytes=len(v),sha256=hashlib.sha256(v).hexdigest()) for n,v in sorted(files.items())}
    files['MANIFEST.json']=json.dumps(dict(files=manifest),indent=2).encode()
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.out,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for n,v in sorted(files.items()):z.writestr('kather_optical_comparison/'+n,v)
    a.out.with_suffix('.manifest.json').write_bytes(files['MANIFEST.json']);a.out.with_suffix('.sha256').write_text(hashlib.sha256(a.out.read_bytes()).hexdigest()+'  '+a.out.name+'\n')
    print(json.dumps(dict(zip=str(a.out),files=len(files),bytes=a.out.stat().st_size)))
