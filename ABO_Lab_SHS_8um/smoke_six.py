"""Few-query REAL six-stage bring-up; never certifies production geometry/LUT.

100 measured title candidates are retained. Fresh session only. Per-stage flat
challenge, target repeat and post-capture PCC checks; fail closed, never swap in
theoretical CCD. No bank approval is manufactured. Formal runs use dual_run.py.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from PIL import Image
from dual_run import Remote,STAGES
from phase_owner import PhaseOwner
from phase_hdmi import sha
from phase_fingerprint import pcc,settings_signature,fingerprint
from guarded_workflow import write,read,require_geometry
from diagnostic_config import validate

ROOT=Path(__file__).resolve().parent

def run(remote,link,c,config_rel,out,limit=4):
    validate(c);roi=require_geometry(c);session=c['diagnostic_session']
    if not 1<=limit<=8:raise ValueError('Smoke only: 1..8 queries; all 100 titles retained')
    if not c['camera'].get('gain'):raise ValueError('Explicit camera gain required')
    if remote.exists(f'sessions/{session}/session.json'):raise ValueError('Fresh diagnostic session required; old data not overwritten')
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    report={'status':'running','production_qualified':False,'session':session,'queries':limit,'candidates':100,
        'expected_stage_counts':[limit]*3+[100+limit]*3,'stages':[],
        'geometry_evidence':c['geometry_evidence'],'note':'Small-sample REAL flow check. Provisional geometry, phase LUT not certified.'}
    def save():write(out/'report.json',report)
    save()
    def job(spec):return remote.job(dict(spec,config=config_rel))
    try:
        job({'action':'init','session':session,'limit':limit})
        base='generated/'+session
        flat=out/'flat.bmp';remote.download(base+'/cal/P_ZERO.bmp',flat)
        probe_remote=c.get('diagnostic_probe_bmp',base+'/cal/A_CHECK_64.bmp')
        if 'diagnostic_probe_bmp' in c:
            p=Path(probe_remote)
            if p.is_absolute() or '..' in p.parts or not p.parts or p.parts[0]!='results':
                raise ValueError('Diagnostic probe must be a relative results artifact')
            probe_file=out/'diagnostic_probe.bmp';remote.download(probe_remote,probe_file)
            if sha(probe_file)!=c.get('diagnostic_probe_sha256'):raise ValueError('Probe SHA mismatch')
            with Image.open(probe_file) as im:
                if im.size!=(1920,1080) or im.mode!='L' or im.format!='BMP':raise ValueError('Probe must be native Mono8 amplitude BMP')
        actual=None
        def probe(receipt,label):
            nonlocal actual
            dest=f'results/smoke_checks/{session}/{label}'
            job({'action':'probe','bmp':probe_remote,'out':dest,'phase_receipt':receipt})
            path=out/(label+'.png');remote.download(dest+'/raw.png',path);remote.download(dest+'/capture.json',path.with_suffix('.json'))
            a=np.array(Image.open(path));meta=read(path.with_suffix('.json'));now=settings_signature(meta)
            if actual is not None and actual!=now:raise RuntimeError('Camera settings drift')
            actual=now;v,stats=fingerprint(a,roi)
            if stats['p99']<16 or stats['std']<2 or stats['saturation_fraction']>.01:raise RuntimeError('Diagnostic probe signal invalid: '+str(stats))
            return v,stats
        with PhaseOwner(link,flat,flat) as owner:
            for index,stage in enumerate(STAGES):
                started=time.monotonic();row={'stage':stage,'status':'preparing'};report['stages'].append(row);save()
                job({'action':'prepare','session':session,'stage':stage})
                mf=remote.read(f'sessions/{session}/play/{stage}/manifest.json')
                if len(mf['entries'])!=report['expected_stage_counts'][index]:raise ValueError('Unexpected sample count')
                phase=out/(stage+'.bmp');remote.download(mf['phase_file'].replace('\\','/'),phase)
                if sha(phase)!=mf['phase_sha256']:raise ValueError('Phase hash mismatch')
                v0,_=probe(owner.show(flat),stage+'_flat')
                receipt=owner.show(phase,mf['phase_sha256']);v1,s1=probe(receipt,stage+'_before')
                receipt=owner.show(phase,mf['phase_sha256']);v2,s2=probe(receipt,stage+'_repeat')
                repeat=pcc(v1,v2);challenge=pcc(v0,v2)
                row.update(phase_sha256=mf['phase_sha256'],count=len(mf['entries']),repeat_pcc=repeat,flat_pcc=challenge)
                save()
                if repeat<.97 or repeat-challenge<.01:raise RuntimeError('Phase repeat/challenge check failed: '+str(row))
                row['status']='capturing';save()
                # This is deliberately NOT a fabricated approved-bank receipt.
                receipt['diagnostic_probe']={'repeat_pcc':repeat,'flat_pcc':challenge,'production_verified':False}
                job({'action':'capture','session':session,'stage':stage,'phase_receipt':receipt})
                after,sa=probe({'phase_sha256':mf['phase_sha256'],'no_phase_write':True},stage+'_after')
                post=pcc(v2,after);ratio=abs(sa['mean']/max(s2['mean'],1e-9)-1)
                row.update(post_pcc=post,post_mean_ratio_error=ratio,elapsed_s=time.monotonic()-started)
                if post<.97 or ratio>.2:
                    row['status']='rejected_postcheck';save();raise RuntimeError('Stage drift; no subsequent preparation/evaluation')
                # Capture creates a record only after raw/route/file checks pass.
                for e in mf['entries']:
                    record=remote.read(f"sessions/{session}/ccd/{e['id']}/{stage}.record.json")
                    if record['phase_sha256']!=mf['phase_sha256'] or record['capture_mode']!='real':raise ValueError('Capture record mismatch')
                row['status']='real_capture_complete';save();print('COMPLETED',stage,row,flush=True)
            job({'action':'evaluate','session':session})
            remote.download(f'sessions/{session}/results/metrics.json',out/'metrics.json')
            remote.download(f'sessions/{session}/results/compute_memory.json',out/'compute_memory.json')
            report.update(status='completed',camera_actual=actual,total_real_captures=sum(report['expected_stage_counts']))
        report['display_audit']=owner.display.audit;save()
    except BaseException as e:
        report.update(status='stopped',error=str(e));save();raise

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--link-config',type=Path,required=True)
    p.add_argument('--remote-config',required=True);p.add_argument('--limit',type=int,default=4);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();link=read(a.link_config)
    with Remote(link) as remote:
        c=remote.read(a.remote_config)
        run(remote,link,c,a.remote_config,a.out,a.limit)

if __name__=='__main__':main()
