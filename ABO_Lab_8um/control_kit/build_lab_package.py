"""Build a commit-pinned, standalone Holoeye/DVP control ZIP; never harvest sessions."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
VERIFIED={
 'device_driver.py':'ba858591d2d71ad5e6fdea34427a503093ae60d742b54951503671f648bb3df7',
 'legacy/dvp_capture_worker.py':'baeda8f71b6b60e1c847beb237a924780ae2daff0705d8810fb8c850c9b36512',
 'vendor/dvp_py36_x64/dvp.pyd':'d5adc160a6006fece32b3bf2dc0ad9abcb446d4347492957e906baee63475d06',
 'vendor/dvp_py36_x64/DVPCamera64.dll':'b2d200a02ef0b068e822c5fd3c96931bbdd078256cd1ed8bbfc240807c17cd0b'}


def build(ref='HEAD'):
    def git(*args): return subprocess.check_output(['git',*args],cwd=REPO)
    commit=git('rev-parse',ref+'^{commit}').decode().strip()
    files={}
    for name in ['control.py','config.json','requirements.txt','README.md','example_api.py','test_control.py']:
        files[name]=git('show',commit+':ABO_Lab_8um/control_kit/'+name)
    for dst,src in [('device_driver.py','devices.py'),('legacy/dvp_capture_worker.py','legacy/dvp_capture_worker.py')]:
        files[dst]=git('show',commit+':experiments/hardware_sdk/'+src)
    for name in ['dvp.pyd','DVPCamera64.dll']:
        files['vendor/dvp_py36_x64/'+name]=(REPO/'experiments/hardware_sdk/vendor_sdk/camera_dvp_legacy/lib/windows/python3.6/x64'/name).read_bytes()
    for name,expected in VERIFIED.items():
        actual=hashlib.sha256(files[name]).hexdigest()
        if actual!=expected: raise RuntimeError('Not the laboratory-verified driver: '+name)
    files['LAB_VERIFIED_SOURCE.json']=json.dumps({'checked_date':'2026-09-10','method':'read-only SSH hashes matched current laboratory files',
        'main_python_observed':'3.12.4 x64','dvp_python_observed':'3.6.13 x64','files_sha256':VERIFIED,
        'hardware_retested_for_this_zip':False},indent=2).encode()
    manifest={'schema_version':1,'source_commit':commit,'kit':'Holoeye8um_DVP_control',
        'files':{n:{'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)} for n,b in files.items()}}
    files['MANIFEST.json']=json.dumps(manifest,indent=2).encode()
    out=REPO/'ABO_Lab_8um/releases'; out.mkdir(parents=True,exist_ok=True)
    target=out/f'Holoeye8um_DVP_control_{commit[:8]}.zip'
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,b in files.items(): z.writestr('SLM_CCD/'+name,b)
    h=hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_suffix('.zip.sha256').write_text(h+'  '+target.name+'\n',encoding='ascii')
    with zipfile.ZipFile(target) as z:
        if z.testzip() is not None: raise RuntimeError('ZIP CRC failed')
        for name,entry in manifest['files'].items():
            if hashlib.sha256(z.read('SLM_CCD/'+name)).hexdigest()!=entry['sha256']: raise RuntimeError(name)
    print(json.dumps({'zip':str(target),'bytes':target.stat().st_size,'sha256':h,'commit':commit},indent=2))
    return target


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ref',default='HEAD');a=p.parse_args();build(a.ref)
