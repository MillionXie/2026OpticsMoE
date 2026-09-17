"""Task builder backend: immutable source files, four 6-layer reference weights."""
import hashlib,json,subprocess,zipfile
from pathlib import Path

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
