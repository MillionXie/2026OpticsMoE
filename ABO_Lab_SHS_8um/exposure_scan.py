"""Local six-phase exposure audit; two fixed-input probes per phase, no long bank sweep."""
import argparse,json,re
from pathlib import Path
import numpy as np
from PIL import Image
from dual_run import Remote,STAGES
from phase_owner import PhaseOwner
from phase_hdmi import sha
from phase_fingerprint import pcc
from guarded_workflow import read,write,require_geometry
from exposure_batch import recommend
ROOT=Path(__file__).resolve().parent

def select_entries(entries):
    # All four image queries + fixed first/middle/last title; not chosen by outcome.
    images=[e for e in entries if e['id'].startswith('image_')]
    titles=[e for e in entries if e['id'].startswith('title_')]
    selected=images[:4]
    if titles:selected += [titles[i] for i in sorted({0,len(titles)//2,len(titles)-1})]
    if not selected:raise ValueError('No recognized network samples')
    return selected

def plot_report(out,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    for ax,s in zip(axes.flat,report['stages']):
        rows=s['data']['rows'];exps=sorted({r['exposure_us'] for r in rows})
        for key,label in [('p99','max p99'),('p999','max p99.9')]:
            ax.plot(exps,[max(r['stats'][key] for r in rows if r['exposure_us']==e) for e in exps],marker='o',label=label)
        ax.plot(exps,[min(r['stats']['p99'] for r in rows if r['exposure_us']==e) for e in exps],marker='s',label='min p99')
        ax.axhline(245,ls='--',color='r',lw=1);ax.axhline(16,ls=':',color='gray');ax.set_ylim(0,260)
        ax.set_title(s['stage']);ax.set_xlabel('Exposure (us)');ax.set_ylabel('Raw ROI pixel value');ax.legend(fontsize=7)
    fig.suptitle('Fixed gain/frame rate; raw sensor ROI BEFORE warp; sampled network inputs only')
    fig.savefig(out/'exposure_range.png',dpi=150);plt.close(fig)

def run(link_path,config_rel,session,out,exposures,stages=None,test_wait_ms=200,capture_wait_ms=None):
    for wait in [test_wait_ms,capture_wait_ms]:
        if wait is not None and (not np.isfinite(wait) or not 200<=wait<=1000):raise ValueError('Audit wait must be200..1000ms')
    if not re.fullmatch('[A-Za-z0-9_-]{1,80}',session):raise ValueError('Invalid source session')
    out=Path(out).resolve()
    if not out.is_relative_to(ROOT/'results'):raise ValueError('Output outside results')
    out.mkdir(parents=True,exist_ok=False);link=read(link_path)
    report=dict(status='running',source_session=session,source_sha256=sha(__file__),stages=[],
        full_dataset_qualified=False,phase_check='same fixed checker before/after; previous-phase PCC informational, no repeated flat challenges')
    try:
        with Remote(link) as remote:
            c=remote.read(config_rel);require_geometry(c)
            remote.ps("$busy=Get-Process | Where-Object {$_.ProcessName -match 'FastStream|Slideshow|Viewer' -or ($_.ProcessName -match '^python(w)?$' -and $_.Path -like '*ABO_Lab_SHS_8um*')};if($busy){throw 'Hardware busy; no takeover'}")
            report['config']=c;write(out/'report.json',report)
            state=remote.read(f'sessions/{session}/session.json')
            if state['hardware_config']!=c:raise ValueError('Source config must match captured session; scan overrides only exposure in memory')
            flat=out/'flat.bmp';remote.download(f'generated/{session}/cal/P_ZERO.bmp',flat)
            probe=dict(bmp=link['diagnostic_probe_bmp'],sha256=link['diagnostic_probe_sha256'])
            # Known real digit patterns for timing, not an image-similarity test on nearly identical features.
            timing=[]
            for d in [0,1]:
                rel=f'generated/phase_inverted/cal/A_DIGIT_{d}.bmp';local=out/f'A_DIGIT_{d}.bmp'
                remote.download(rel,local);timing.append(dict(id=f'digit{d}',bmp=rel,sha256=sha(local)))
            prev=None
            with PhaseOwner(link,flat,flat) as owner:
                for stage in (stages or STAGES):
                    mf=remote.read(f'sessions/{session}/play/{stage}/manifest.json')
                    phase=out/(stage+'.bmp');remote.download(mf['phase_file'].replace('\\','/'),phase)
                    receipt=owner.show(phase,mf['phase_sha256'])
                    selected=select_entries(mf['entries'])
                    dest=out.relative_to(ROOT).as_posix()+'/'+stage
                    spec=dict(action='exposure_batch',config=config_rel,out=dest,phase_receipt=receipt,
                        phase_sha256=mf['phase_sha256'],probe=probe,probe_exposure_us=150,
                        exposure_rows=[dict(id=e['id'],bmp=f"sessions/{session}/play/{stage}/{e['bmp']}",sha256=e['sha256']) for e in selected],
                        exposures_us=exposures,repeats=2,timing_repeats=5 if stage=='vision_global' else 0,
                        timing_rows=timing,test_wait_ms=test_wait_ms,capture_wait_ms=capture_wait_ms)
                    remote.job(spec)
                    local=out/stage;local.mkdir();remote.download(dest+'/report.json',local/'report.json')
                    data=read(local/'report.json')
                    for name in ['probe_before.png','probe_after.png']:
                        remote.download(dest+'/'+name,local/name)
                    v=np.asarray(Image.open(local/'probe_before.png'))
                    row=dict(stage=stage,data=data,previous_phase_pcc=None if prev is None else pcc(prev,v));prev=v
                    report['stages'].append(row);write(out/'report.json',report)
                    print(stage,data['recommendation'],'hold PCC',data['phase_hold_pcc'],flush=True)
                    if data['phase_hold_pcc']<.97:raise RuntimeError('Fixed-input phase hold unstable; stop audit')
            allrows=[r for s in report['stages'] for r in s['data']['rows']]
            report.update(status='complete',recommendation=recommend(allrows),phase_sdk_audit=owner.audit)
            timingrows=[r for s in report['stages'] for r in s['data']['timing']]
            report['timing_summary']=dict(n=len(timingrows),correct=sum(r['correct'] for r in timingrows),
                test_wait_ms=test_wait_ms,
                minimum_target_pcc=min((r['scores'][r['id']] for r in timingrows),default=None),
                all_references_discriminative=all(r['valid_reference'] for r in timingrows),
                note='Small-sample configured-wait recheck against independent400ms references; no hard-trigger guarantee')
            write(out/'report.json',report);plot_report(out,report)
    except BaseException as e:
        report.update(status='failed',error=str(e));write(out/'report.json',report);raise
    print('COMPLETE',out,report['recommendation'],report['timing_summary'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--link-config',type=Path,required=True);p.add_argument('--remote-config',required=True)
    p.add_argument('--source-session',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--exposures-us',type=float,nargs='+',default=[150,300,450,600,900])
    p.add_argument('--stages',nargs='+',choices=STAGES);p.add_argument('--test-wait-ms',type=float,default=200)
    p.add_argument('--capture-wait-ms',type=float,help='Diagnostic replay only; no source session/config mutation');a=p.parse_args()
    run(a.link_config,a.remote_config,a.source_session,a.out,a.exposures_us,a.stages,a.test_wait_ms,a.capture_wait_ms)
