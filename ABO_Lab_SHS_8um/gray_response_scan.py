"""Uniform-aperture gray sweep and independent pattern timing at fixed exposure.

No LUT fit, normalization of inference pixels, or automatic full-run approval.
Uses the bounded raw capture worker; all frames and metadata retained.
"""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from dual_run import Remote
from guarded_workflow import read,write,require_geometry
from mnist_joint_capture import run as capture_run,sha
from exposure_batch import image_stats
from phase_fingerprint import pcc
ROOT=Path(__file__).resolve().parent


def validate_settings(exposure_us,wait_ms):
    if not np.isfinite(exposure_us) or not 1<=exposure_us<=2000:raise ValueError('Exposure must be1..2000us')
    if not np.isfinite(wait_ms) or not 200<=wait_ms<=1000:raise ValueError('Wait must be200..1000ms')


def aperture(c):
    pitch=float(c['amplitude_slm']['pixel_pitch_um'])
    n=int(round(c['model_active_pixels']*c['model_pitch_um']/pitch))
    cx,cy=c['amplitude_slm']['center_xy'];x=int(round(cx-n/2));y=int(round(cy-n/2))
    if not (0<=x<x+n<=1920 and 0<=y<y+n<=1080):raise ValueError('Aperture outside SLM')
    return x,y,n


def uniform(c,g):
    x,y,n=aperture(c);a=np.zeros((1080,1920),np.uint8);a[y:y+n,x:x+n]=g
    return a


def analyze(out):
    manifest=read(out/'manifest.json');cap=read(out/'capture/report.json')
    if not cap['complete']:raise ValueError('Capture incomplete; do not infer missing measurements')
    c=cap['remote_config_snapshot'];im=Image.new('1',(1920,1080))
    points=[tuple(c['logical_corners_full_sensor_xy'][k]) for k in ['top_left','top_right','bottom_right','bottom_left']]
    ImageDraw.Draw(im).polygon(points,fill=1);mask=np.asarray(im,dtype=bool)
    data=[];references={};raws={};metas={}
    for r in cap['rows']:
        path=out/'capture'/(r['name']+'.png')
        if sha(path)!=r['raw_sha256']:raise ValueError('CCD hash mismatch')
        a=np.asarray(Image.open(path));meta=read(path.with_suffix('.json'))
        cam=meta['camera']
        if abs(float(cam['ExposureTime']['value'])-c['camera']['exposure_us'])>1 or cam['Gain']['value']!='Gain_X4' or float(cam['AcquisitionFrameRate']['value'])!=100 or meta['settle_delay_ms']!=c['settle_delay_ms']:
            raise ValueError('Actual exposure/gain/fps/wait differs from this run protocol')
        v=a[mask].astype(np.float32);raws[r['name']]=v;metas[r['name']]=meta
        data.append(dict(name=r['name'],kind=r['kind'],gray=r.get('gray'),digit=r.get('digit'),stats=image_stats(a,mask)))
        if r['kind']=='reference' and r['hold_index']==2:references[r['digit']]=v
    sweep=[]
    for g in manifest['gray_values']:
        rr=[r for r in data if r['kind']=='gray' and r['gray']==g]
        sweep.append(dict(gray=g,mean=float(np.median([r['stats']['mean'] for r in rr])),
            mean_min=min(r['stats']['mean'] for r in rr),mean_max=max(r['stats']['mean'] for r in rr),
            p99=max(r['stats']['p99'] for r in rr),p999=max(r['stats']['p999'] for r in rr),
            maximum=max(r['stats']['maximum'] for r in rr),
            saturation_fraction=max(r['stats']['saturation_fraction'] for r in rr)))
    separation=pcc(references[0],references[1]);tests=[]
    for r in data:
        if r['kind']!='timing':continue
        scores={d:pcc(raws[r['name']],v) for d,v in references.items()}
        tests.append(dict(name=r['name'],digit=r['digit'],prediction=max(scores,key=scores.get),scores=scores))
    holds={}
    for d in [0,1]:
        first=metas[f'ref_d{d}_0']['host_receive_monotonic_ns'];last=metas[f'ref_d{d}_2']['host_receive_monotonic_ns']
        holds[d]=(last-first)/1e6
    refstats=[r['stats'] for r in data if r['name'] in ['ref_d0_2','ref_d1_2']]
    signal=all(s['p99']>=16 and s['std']>=2 for s in refstats)
    correct=sum(t['prediction']==t['digit'] for t in tests)
    minimum=min(t['scores'][t['digit']] for t in tests)
    timing_passed=signal and separation<.95 and min(holds.values())>=400 and correct==len(tests) and minimum>=.97
    timing_ms={}
    for key in ['bmp_validate_ms','slm_preload_ms','slm_show_to_visible_ms','settle_actual_ms','final_fresh_ms','capture_total_ms']:
        values=[metas[t['name']][key] for t in tests]
        timing_ms[key]=dict(mean=float(np.mean(values)),p95=float(np.percentile(values,95)),maximum=float(np.max(values)))
    result=dict(status='complete',config=c,sweep=sweep,frames=data,timing=dict(n=len(tests),correct=correct,
        minimum_target_pcc=minimum,reference_pair_pcc=separation,reference_hold_ms=holds,
        valid_reference_signal=signal,pattern_timing_passed=timing_passed,rows=tests),
        timing_ms=timing_ms,timing_scope='Prepared BMP to raw host frame; excludes inference, PNG save, SSH and phase switching. Decode overlaps acquisition; do not sum it twice.',
        raw_polygon_roi_before_warp=True,full_dataset_qualified=False,
        note='Uniform-gray response is not sufficient to certify focused network CCDs or fit a phase/amplitude LUT.')
    write(out/'analysis.json',result);plot(out,result)
    print(json.dumps(dict(sweep=sweep,timing={k:v for k,v in result['timing'].items() if k!='rows'})),flush=True)
    return result


def plot(out,r):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=r['sweep'];x=[s['gray'] for s in rows]
    fig,ax=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    ax[0].plot(x,[s['mean'] for s in rows],'o-');ax[0].fill_between(x,[s['mean_min'] for s in rows],[s['mean_max'] for s in rows],alpha=.2)
    ax[0].set_ylabel('Raw ROI mean (0-255); median / range of 3')
    ax[1].plot(x,[s['p99'] for s in rows],'o-',label='max p99');ax[1].plot(x,[s['p999'] for s in rows],'o-',label='max p99.9');ax[1].axhline(255,ls='--',color='red');ax[1].set_ylim(0,260);ax[1].legend()
    ax[2].plot(x,[s['saturation_fraction']*100 for s in rows],'o-');ax[2].set_ylabel('Raw pixels equal to 255 (%)')
    for a in ax:a.set_xlabel('Amplitude SLM gray');a.grid(alpha=.2)
    fig.suptitle(f"Uniform 1016-pixel aperture | {r['config']['camera']['exposure_us']:g} us, Gain X4, 100 fps, {r['config']['settle_delay_ms']:g} ms wait | flat phase")
    fig.savefig(out/'gray_response.png',dpi=150);plt.close(fig)
    selected=[0,64,128,191,255];fig,axes=plt.subplots(1,5,figsize=(15,4),constrained_layout=True)
    c=r['config'];pts=np.array([c['logical_corners_full_sensor_xy'][k] for k in ['top_left','top_right','bottom_right','bottom_left','top_left']])
    for ax,g in zip(axes,selected):
        raw=Image.open(out/'capture'/f'g{g:03d}_r0.png');ax.imshow(raw,cmap='gray',vmin=0,vmax=255)
        ax.plot(pts[:,0],pts[:,1],color='red',lw=.7);ax.set_title(f'gray={g}; fixed 0-255');ax.axis('off')
    fig.savefig(out/'gray_previews.png',dpi=150);plt.close(fig)


def run(link_path,source_config,out,exposure_us=400,wait_ms=200,switch_count=40):
    validate_settings(exposure_us,wait_ms)
    if not 40<=switch_count<=80:raise ValueError('40..80 switches; total stays within 128-capture diagnostic bound')
    out=Path(out).resolve()
    if not out.is_relative_to(ROOT/'results'):raise ValueError('Output outside results')
    if not out.name.replace('_','').isalnum() or len(out.name)>55:raise ValueError('Use short safe run name')
    run_id='_'.join(out.relative_to(ROOT/'results').parts)
    if len(run_id)>60:raise ValueError('Combined run id too long')
    out.mkdir(parents=True,exist_ok=False);link=read(link_path)
    with Remote(link) as remote:
        remote.ps("$busy=Get-Process | Where-Object {$_.ProcessName -match 'FastStream|Slideshow|python|Viewer'};if($busy){throw 'Remote hardware busy; close GUIs first'}")
        c=remote.read(source_config);require_geometry(c)
        c.update(diagnostic_only=True,diagnostic_session='smoke_'+run_id,settle_delay_ms=wait_ms)
        c.pop('diagnostic_query_indices',None);c['camera'].update(exposure_us=exposure_us,gain='Gain_X4',frame_rate_hz=100)
        c['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        c['exposure_selection']=f'Fixed {exposure_us:g}us uniform-gray and{wait_ms:g}ms timing diagnostic; no full-run approval'
        cfg=f"results/smoke_configs/{c['diagnostic_session']}.json"
        if remote.exists(cfg):raise FileExistsError('Config exists')
        write(ROOT/cfg,c);remote.putjson(cfg,c)
    generated=ROOT/'generated'/run_id;generated.mkdir(exist_ok=False)
    # Already-exported uniform phase uses the current 255-g convention exactly once.
    phase=ROOT/'results/pre_full_400us_200ms_20260913/flat.bmp'
    if not phase.exists():raise FileNotFoundError(phase)
    gray=[int(v) for v in np.rint(np.linspace(0,255,13))]
    for g in gray:Image.fromarray(uniform(c,g)).save(generated/f'g{g:03d}.bmp')
    # Freeze the installed bench's digit assets into this run's own namespace.
    # Do not assume an older local generated/ file has the same geometry/hash.
    with Remote(link) as remote:
        for d in [0,1]:remote.download(f'generated/phase_inverted/cal/A_DIGIT_{d}.bmp',generated/f'A_DIGIT_{d}.bmp')
    rows=[]
    def add(name,path,**info):rows.append(dict(name=name,amplitude=str(path),amplitude_sha256=sha(path),phase=str(phase),phase_sha256=sha(phase),**info))
    for repeat,values in enumerate([gray,gray[::-1],np.random.default_rng(20260913).permutation(gray)]):
        for g in values:add(f'g{g:03d}_r{repeat}',generated/f'g{g:03d}.bmp',kind='gray',gray=int(g))
    for d in [0,1]:
        for i in range(3):add(f'ref_d{d}_{i}',generated/f'A_DIGIT_{d}.bmp',kind='reference',digit=d,hold_index=i)
    for i in range(switch_count):
        d=i%2;add(f'test{i:02d}_d{d}',generated/f'A_DIGIT_{d}.bmp',kind='timing',digit=d)
    write(out/'manifest.json',dict(kind='uniform gray response and pattern timing; not MNIST accuracy',gray_values=gray,switch_count=switch_count,rows=rows,aperture_xy_size=aperture(c),config=cfg))
    capture_run(out/'manifest.json',link_path,out/'capture',batch=True,config_rel=cfg)
    analyze(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--link',type=Path,required=True);p.add_argument('--source-config',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--exposure-us',type=float,default=400);p.add_argument('--wait-ms',type=float,default=200)
    p.add_argument('--switch-count',type=int,default=40)
    p.add_argument('--analyze-only',action='store_true');a=p.parse_args()
    if a.analyze_only:analyze(a.out)
    else:run(a.link,a.source_config,a.out,a.exposure_us,a.wait_ms,a.switch_count)
