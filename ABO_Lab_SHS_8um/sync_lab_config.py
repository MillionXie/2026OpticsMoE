"""Pull/edit/push the junior computer's hardware config; preserve a backup, reject concurrent edits."""
import argparse,json,time,uuid
from pathlib import Path
from dual_run import Remote
from guarded_workflow import read,write,require_geometry,ROOT
from phase_fingerprint import digest

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['pull','push'])
    p.add_argument('--link-config',type=Path,default=ROOT/'dual.local.json')
    p.add_argument('--file',type=Path,default=ROOT/'LAB.review.json');a=p.parse_args()
    side=a.file.with_suffix('.source.json')
    with Remote(read(a.link_config)) as remote:
        current=remote.read('LAB.local.json')
        if a.action=='pull':
            if a.file.exists() or side.exists():raise FileExistsError('Use a new --file; do not overwrite local edits')
            write(a.file,current);write(side,{'source_sha256':digest(current),'host':remote.c['host'],'project':remote.root});print(a.file);return
        source=read(side)
        if source['source_sha256']!=digest(current) or source['host']!=remote.c['host'] or source['project']!=remote.root:raise ValueError('Remote config changed since pull; fetch to a new file and reconcile')
        c=read(a.file)
        if [c['model_active_pixels'],c['model_pitch_um'],c['distance_m'],c['wavelength_nm']]!=[478,17,.1,532]:raise ValueError('Fixed model physical geometry must not change')
        if c['capture_input_range']!=[0,255] or c['camera']['expected_pixel_format']!='Mono8':raise ValueError('SHS profile requires Mono8 fixed scale [0,255]')
        if c['phase_slm']['gray_encoding'] not in ('normal','inverted_255_minus_g'):raise ValueError('Invalid phase encoding')
        if c.get('geometry_confirmed'):require_geometry(c)
        _,out,_=remote.ssh.exec_command('tasklist /FO CSV');tasks=out.read().decode(errors='replace').lower()
        if any(n in tasks for n in ('python.exe','pythonw.exe','faststream.exe','holoeye-slideshowplayer.exe')):raise RuntimeError('Close hardware apps/jobs before changing hardware config')
        if any(x in remote.root for x in "'<>\r\n"):raise ValueError('Unsafe project path')
        tag=time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]
        tmp='LAB.upload_'+tag+'.json';remote.putjson(tmp,c)
        backup='results/config_backups/LAB_'+tag+'.json'
        remote.ps(f"New-Item -ItemType Directory -Force -Path '{remote.root}/results/config_backups' | Out-Null\nCopy-Item -LiteralPath '{remote.root}/LAB.local.json' -Destination '{remote.root}/{backup}'\nMove-Item -LiteralPath '{remote.root}/{tmp}' -Destination '{remote.root}/LAB.local.json' -Force")
        if digest(remote.read('LAB.local.json'))!=digest(c):raise RuntimeError('Config upload verification failed')
        write(side,{'source_sha256':digest(c),'host':remote.c['host'],'project':remote.root})
        print('Config updated; old copy:',backup,'; use NEW reference bank and NEW session.')

if __name__=='__main__':main()
