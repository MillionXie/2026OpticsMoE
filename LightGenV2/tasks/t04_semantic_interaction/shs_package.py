"""Small Git-pinned hardware overlay for the existing verified simulation ZIP."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

BASE_SHA='304b7c457c6f6c8e4e7f41f18deaef6f65270f86e09194f69c7a3e8f45b0294f'


def build_shs(base,output):
    root=Path(__file__).resolve().parents[3]
    if output.exists():raise FileExistsError(output)
    digest=hashlib.file_digest(base.open('rb'),'sha256').hexdigest()
    if digest!=BASE_SHA:raise ValueError('Unexpected original simulation package')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    scope='LightGenV2/tasks/t04_semantic_interaction'
    if subprocess.check_output(['git','diff','HEAD','--',scope],cwd=root):raise RuntimeError('Commit/push source first')
    entries={}
    paths=subprocess.check_output(['git','ls-files','LightGenV2','experiments'],cwd=root,text=True).splitlines()
    for rel in paths:
        if not rel.endswith(('.py','.yaml')) or any(x in Path(rel).parts for x in ('runs','data','vendor_sdk')):continue
        entries['runtime/'+rel]=subprocess.check_output(['git','show',commit+':'+rel],cwd=root)
    for name in ('phase_hdmi.py','phase_owner.py','phase_display.py'):
        entries['runtime/phase_control/'+name]=subprocess.check_output(['git','show',commit+':ABO_Lab_SHS_8um/'+name],cwd=root)
    entries['run.py']=b'''import os,sys,time
from pathlib import Path
root=Path(__file__).resolve().parent
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
if sys.stdout is None:
    (root/'logs').mkdir(exist_ok=True)
    log=(root/'logs'/('job_'+time.strftime('%Y%m%d_%H%M%S')+'.log')).open('x',encoding='utf-8',buffering=1)
    sys.stdout=sys.stderr=log
sys.path.insert(0,str(root/'runtime'))
if __name__=='__main__':
    from LightGenV2.tasks.t04_semantic_interaction.lab_bench import main
    main()
'''
    entries['COMMAND_SHS.md']=subprocess.check_output(['git','show',commit+':'+scope+'/COMMAND_SHS.md'],cwd=root)
    entries['install_shs.py']=subprocess.check_output(['git','show',commit+':'+scope+'/install_shs.py'],cwd=root)
    manifest=dict(source_commit=commit,base_simulation_zip_sha256=BASE_SHA,
        checkpoint_sha256='a69ddcee827749fb9202f9aef11ea45011e433d8b2f0151be2eec3db7dbff9eb',
        files={k:hashlib.sha256(v).hexdigest() for k,v in entries.items()})
    entries['SHS_SOURCE.json']=json.dumps(manifest,indent=2).encode()
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'x',zipfile.ZIP_DEFLATED) as z:
        for name,value in entries.items():z.writestr(name,value)
    result=dict(zip=str(output),sha256=hashlib.file_digest(output.open('rb'),'sha256').hexdigest(),source_commit=commit)
    output.with_suffix('.manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result
