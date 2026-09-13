"""Deterministic 8um MNIST assets and raw CCD comparison (no image enhancement).

Geometry is supplied from independent physical markers, never optimized on
classification labels. Two pilot digits may select phase orientation/sign;
the remaining held-out measurement digits are not used for that selection.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np,cv2
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def raster(a):
    n=round(a.shape[0]*17/8)
    ix=np.floor((np.arange(n)+.5-n/2)*8/17+a.shape[0]/2).astype(int).clip(0,a.shape[0]-1)
    return a[np.ix_(ix,ix)]
def canvas(a,height,background=0):
    v=np.full((height,1920),background,np.uint8);h,w=a.shape
    y=(height-h)//2;x=(1920-w)//2;v[y:y+h,x:x+w]=a;return v
def pcc(a,b):
    a=np.asarray(a,float).ravel();b=np.asarray(b,float).ravel()
    if min(a.std(),b.std())<1e-10:return None
    return float(np.corrcoef(a,b)[0,1])
def energy(a,bounds):return [float(a[y0:y1,x0:x1].sum()) for x0,y0,x1,y1 in bounds]

def prepare(reference,geometry,out,variant=None):
    ref=read(reference/'reference.json');assert sha(reference/'reference.npz')==ref['npz_sha256']
    z=np.load(reference/'reference.npz',allow_pickle=False);out.mkdir(parents=True,exist_ok=False)
    g=read(geometry);H=g['variants'][g['selected']]['H']
    rows=[]
    def add(name,a,p,**kwargs):
        rows.append(dict(name=name,amplitude=str(a.resolve()),phase=str(p.resolve()),amplitude_sha256=sha(a),phase_sha256=sha(p),**kwargs))
    phi=np.floor(np.mod(z['phase'],2*np.pi)/(2*np.pi)*256).clip(0,255).astype(np.uint8)
    variants=[f'{f}_{s}' for f in ['none','x','y','xy'] for s in ['normal','inverse']] if variant is None else [variant]
    indices=[0,1] if variant is None else [i for i,r in enumerate(ref['rows']) if r['split']=='heldout_measurement']
    for i in indices:
        a=np.rint(z['amplitude'][i]*255).clip(0,255).astype(np.uint8)
        Image.fromarray(canvas(raster(a),1080)).save(out/(ref['rows'][i]['key']+'.bmp'))
    for v in variants:
        orient,sign=v.split('_');a=phi.copy()
        if 'x' in orient:a=np.fliplr(a)
        if 'y' in orient:a=np.flipud(a)
        phase=canvas(raster(a),1200)
        if sign=='inverse':phase=255-phase
        pp=out/('phase_'+v+'.bmp');Image.fromarray(phase).save(pp)
        for i in indices:
            key=ref['rows'][i]['key'];add(v+'_'+key,out/(key+'.bmp'),pp,reference_index=i,variant=v)
    # Same-image repeat is evidence of stability, not a new accuracy sample.
    row=dict(rows[0]);row['name']='repeat_'+row['name'];row['repeat_of']=rows[0]['name'];rows.append(row)
    result=dict(rows=rows,reference=str(reference.resolve()),reference_json_sha256=sha(reference/'reference.json'),
                geometry_file=str(geometry.resolve()),geometry_sha256=sha(geometry),H=H,
                phase_center_xy=[960,600],amplitude_center_xy=[960,540],phase_size_wh=[1920,1200],amplitude_size_wh=[1920,1080],
                pitch_conversion='17um logical -> 8um native nearest physical pixel centers; 478 -> 1016',
                selection_scope='pilot only' if variant is None else 'fixed held-out measurement',
                photometric_processing='none; single geometric homography only')
    (out/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(out/'manifest.json')

def evaluate(run):
    report=read(run/'report.json');m=report['manifest'];refdir=Path(m['reference']);ref=read(refdir/'reference.json')
    assert sha(refdir/'reference.json')==m['reference_json_sha256']
    assert sha(refdir/'reference.npz')==ref['npz_sha256']
    z=np.load(refdir/'reference.npz',allow_pickle=False);H=np.array(m['H']);bounds=ref['detector_bounds'];rows=[];imgs=[]
    for r in report['rows']:
        raw=np.array(Image.open(run/(r['name']+'.png')));assert sha(run/(r['name']+'.png'))==r['raw_sha256']
        a=cv2.warpPerspective(raw.astype(np.float32),H,(478,478),flags=cv2.INTER_LINEAR)
        # Numeric CCD stays linear float32. Separate display PNG uses fixed 0..255.
        np.save(run/(r['name']+'_canonical.npy'),a)
        b=z['ccd'][r['reference_index']];truth=ref['rows'][r['reference_index']];en=energy(a,bounds)
        result=dict(name=r['name'],variant=r['variant'],reference_index=r['reference_index'],label=truth['label'],
                    measured_prediction=int(np.argmax(en)),raw_detector_energy=en,simulation_prediction=truth['simulation_prediction'],
                    pcc=pcc(a,b),pcc_blur3=pcc(cv2.GaussianBlur(a,(0,0),3),cv2.GaussianBlur(b,(0,0),3)),
                    mean=float(a.mean()),p99=float(np.percentile(a,99)),max=float(a.max()),saturation_fraction=float((a>=254.5).mean()),
                    repeat_of=r.get('repeat_of'))
        rows.append(result)
        im=Image.fromarray(np.rint(a).clip(0,255).astype(np.uint8)).convert('RGB');d=ImageDraw.Draw(im)
        for k,(x0,y0,x1,y1) in enumerate(bounds):d.rectangle((x0,y0,x1-1,y1-1),outline='red');d.text((x0,y0-12),str(k),fill='red')
        d.text((8,8),f"{r['variant']} y={truth['label']} sim={truth['simulation_prediction']} real={result['measured_prediction']}",fill='red')
        im.save(run/(r['name']+'_boxes.png'));imgs.append(im)
    groups={}
    for key in sorted({r['variant'] for r in rows}):
        vs=[r for r in rows if r['variant']==key and not r['repeat_of']]
        groups[key]=dict(n=len(vs),accuracy=float(np.mean([r['measured_prediction']==r['label'] for r in vs])),
                         simulation_accuracy=float(np.mean([r['simulation_prediction']==r['label'] for r in vs])),
                         mean_pcc=float(np.mean([r['pcc'] for r in vs if r['pcc'] is not None])))
    result=dict(scope=m['selection_scope'],complete=report['complete'],groups=groups,rows=rows,normalization_applied=False)
    for r in rows:
        if r['repeat_of']:
            result['repeat_pcc']=pcc(np.load(run/(r['name']+'_canonical.npy')),np.load(run/(r['repeat_of']+'_canonical.npy')))
    (run/'comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    thumb=239;sheet=Image.new('RGB',(thumb*4,thumb*((len(imgs)+3)//4)),(40,40,40))
    for i,im in enumerate(imgs):sheet.paste(im.resize((thumb,thumb)),((i%4)*thumb,(i//4)*thumb))
    sheet.save(run/'measured_contact_sheet.png');print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    a=sub.add_parser('prepare');a.add_argument('--reference',type=Path,required=True);a.add_argument('--geometry',type=Path,required=True);a.add_argument('--out',type=Path,required=True)
    a.add_argument('--variant',choices=[f'{f}_{s}' for f in ['none','x','y','xy'] for s in ['normal','inverse']])
    a=sub.add_parser('evaluate');a.add_argument('--run',type=Path,required=True)
    a=p.parse_args()
    if a.action=='prepare':prepare(a.reference,a.geometry,a.out,a.variant)
    else:evaluate(a.run)
