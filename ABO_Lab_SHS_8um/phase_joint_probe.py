"""Bounded phase SDK + remote amplitude/camera orientation evidence.

Outputs raw frames. Does NOT automatically approve geometry or phase linearity.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from PIL import Image
from dual_run import Remote
from phase_hdmi import PhaseHDMI,sha
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--link-config',type=Path,default=ROOT/'dual.example.json')
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    a.out.mkdir(parents=True);out=a.out.resolve();pattern=out/'patterns';pattern.mkdir()
    link=json.loads(a.link_config.read_text(encoding='utf-8-sig'))
    amplitudes={}
    for name,centers in [('center',[(960,540)]),('xplus',[(1152,540)]),('yplus',[(960,732)]),
                         ('four',[(710,290),(1210,290),(710,790),(1210,790)])]:
        ar=np.zeros((1080,1920),np.uint8)
        for x,y in centers:ar[y-64:y+64,x-64:x+64]=255
        dest=pattern/('A_'+name+'.bmp');Image.fromarray(ar).save(dest);amplitudes[name]=dest
    yy,xx=np.indices((1200,1920));gx=((xx%32)/31*255).astype(np.uint8);gy=((yy%32)/31*255).astype(np.uint8)
    fields={'flat':np.zeros((1200,1920),np.uint8),'gx':gx,'gx_inverse':255-gx,'gy':gy,'gy_inverse':255-gy}
    for name,(x0,x1,y0,y1) in {'TL':(0,960,0,600),'TR':(960,1920,0,600),'BL':(0,960,600,1200),'BR':(960,1920,600,1200)}.items():
        ar=np.zeros((1200,1920),np.uint8);ar[y0:y1,x0:x1]=gx[y0:y1,x0:x1];fields[name]=ar
    phases={}
    for name,field in fields.items():
        dest=pattern/('P_'+name+'.bmp');Image.fromarray(field).save(dest);phases[name]=dest
    seq=[('center','flat'),('xplus','flat'),('yplus','flat')]+[('center',x) for x in ['gx','gx_inverse','gy','gy_inverse']]+[('four',x) for x in ['flat','TL','TR','BL','BR']]
    report={'scope':'Raw device-raster orientation/phase modulation diagnostic; not a phase LUT calibration',
            'patterns':'128px amplitude patches; 32px phase sawtooth; no spatial flip; inverse explicitly named',
            'complete':False,'rows':[]}
    with PhaseHDMI(link['phase_sdk'],link['phase_lut'],link.get('phase_settle_s',1)) as phase:
      try:
        with Remote(link) as remote:
            remote_dir='results/phase_joint_'+time.strftime('%Y%m%d_%H%M%S')
            remote.ps(f"New-Item -ItemType Directory -Path '{remote.root}/{remote_dir}' | Out-Null")
            for name,path in amplitudes.items():remote.sftp.put(str(path),remote.root+'/'+remote_dir+'/'+path.name)
            report['remote_directory']=remote_dir
            for i,(an,pn) in enumerate(seq):
                receipt=phase.show(phases[pn]);capout=remote_dir+f'/{i:02d}_{an}_{pn}'
                remote.job({'action':'probe','bmp':remote_dir+'/'+amplitudes[an].name,'out':capout,'phase_receipt':receipt})
                frame=out/f'{i:02d}_{an}_{pn}.png';remote.download(capout+'/raw.png',frame)
                remote.download(capout+'/capture.json',frame.with_suffix('.json'))
                report['rows'].append({'amplitude':an,'phase':pn,'phase_receipt':receipt,'file':frame.name,'sha256':sha(frame)})
                (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
                print(i,an,pn,flush=True)
            report['complete']=True
      finally:
        # Return to uniform BLACK while SDK is alive. Closing may restore desktop.
        phase.show(phases['flat'])
        (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Phase SDK closed after restoring uniform black; reopen GUI if you need to hold it.')

if __name__=='__main__':main()
