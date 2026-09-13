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
from phase_fingerprint import pcc,settings_signature,fingerprint,digest
from guarded_workflow import write,read,require_geometry
from diagnostic_config import validate

ROOT=Path(__file__).resolve().parent

def post_state(post,ratio,brightness_warning_only=False):
    if post<.97:return 'rejected_postcheck'
    if ratio>.2:return 'real_capture_complete_photometric_warning' if brightness_warning_only else 'rejected_postcheck'
    return 'real_capture_complete'

def run(remote,link,c,config_rel,out,limit=4,resume=False,brightness_warning_only=False):
    validate(c);roi=require_geometry(c);session=c['diagnostic_session']
    if not 1<=limit<=8:raise ValueError('Smoke only: 1..8 queries; all 100 titles retained')
    if not c['camera'].get('gain'):raise ValueError('Explicit camera gain required')
    existing=remote.exists(f'sessions/{session}/session.json')
    if existing and not resume:raise ValueError('Fresh diagnostic session required; old data not overwritten')
    if resume:
        if not existing or not out.exists():raise ValueError('Both local journal and remote session required to resume')
        report=read(out/'report.json');state=remote.read(f'sessions/{session}/session.json')
        if report['session']!=session or report['queries']!=limit or digest(state['hardware_config'])!=digest(c):raise ValueError('Resume identity mismatch')
        write(out/('report_before_resume_'+time.strftime('%Y%m%d_%H%M%S')+'.json'),report)
        for row in list(report['stages']):
            if row['status']=='preparing':
                # No camera capture for this stage has started. Retain the
                # failed preflight evidence but allow deterministic reprepare.
                for s in state['samples']:
                    prefix=f"sessions/{session}/ccd/{s['id']}/{row['stage']}"
                    if any(remote.exists(prefix+ext) for ext in ('.png','.raw.png','.record.json','.capture.json')):
                        raise ValueError('Preflight resume found capture files; audit them instead')
                report.setdefault('incomplete_preflight_attempts',[]).append(row)
                report['stages'].remove(row);continue
            if row['status']=='rejected_postcheck' and brightness_warning_only and row.get('post_pcc',-1)>=.97:
                row.update(original_status='rejected_postcheck',status='real_capture_complete_photometric_warning')
            if not row['status'].startswith('real_capture_complete'):
                raise ValueError('Only completed capture with a recorded passing shape postcheck can resume; no partial-stage bypass')
        report.setdefault('protocol_amendments',[]).append({'time':time.strftime('%Y-%m-%dT%H:%M:%S'),
          'brightness_warning_only':brightness_warning_only,'scope':'Diagnostic flow only; unchanged pixels/config, formal acceptance NOT granted'})
        report['status']='running';report.pop('error',None)
    else:
        if out.exists():raise FileExistsError(out)
        out.mkdir(parents=True)
        report={'status':'running','production_qualified':False,'session':session,'queries':limit,'candidates':100,
        'expected_stage_counts':[limit]*3+[100+limit]*3,'stages':[],
          'geometry_evidence':c['geometry_evidence'],'note':'Small-sample REAL flow check. Provisional geometry, phase LUT not certified.'}
    report['brightness_warning_only']=brightness_warning_only
    report['coordinator_source_sha256']=sha(__file__)
    def save():write(out/'report.json',report)
    save()
    def job(spec):return remote.job(dict(spec,config=config_rel))
    try:
        if not resume:job({'action':'init','session':session,'limit':limit})
        base='generated/'+session
        flat=out/'flat.bmp';remote.download(base+'/cal/P_ZERO.bmp',flat)
        probe_remote=link.get('diagnostic_probe_bmp',c.get('diagnostic_probe_bmp',base+'/cal/A_CHECK_64.bmp'))
        probe_config=link.get('diagnostic_probe_config',config_rel)
        if probe_config!=config_rel:
            pc=remote.read(probe_config);validate(pc)
            cc=json.loads(json.dumps(c));cc['camera']['exposure_us']=pc['camera']['exposure_us']
            if digest(cc)!=digest(pc):raise ValueError('Separate probe config may ONLY differ in camera.exposure_us')
        if 'diagnostic_probe_bmp' in c or 'diagnostic_probe_bmp' in link:
            p=Path(probe_remote)
            if p.is_absolute() or '..' in p.parts or not p.parts or p.parts[0] not in ('results','generated'):
                raise ValueError('Diagnostic probe must be a relative results/generated artifact')
            expected_probe=link.get('diagnostic_probe_sha256',c.get('diagnostic_probe_sha256'))
            if not isinstance(expected_probe,str) or len(expected_probe)!=64:raise ValueError('Probe SHA required')
            probe_file=out/('diagnostic_probe_'+expected_probe[:12]+'.bmp');remote.download(probe_remote,probe_file)
            if sha(probe_file)!=expected_probe:raise ValueError('Probe SHA mismatch')
            with Image.open(probe_file) as im:
                if im.size!=(1920,1080) or im.mode!='L' or im.format!='BMP':raise ValueError('Probe must be native Mono8 amplitude BMP')
        report['probe_protocol']={'config':probe_config,'bmp':probe_remote,'network_config':config_rel,
            'separate_exposure':probe_config!=config_rel,'probe_only_never_used_as_model_feature':True};save()
        actual=None
        def probe(receipt,label):
            nonlocal actual
            if resume:label+='_resume'+str(len(report['protocol_amendments']))
            dest=f'results/smoke_checks/{session}/{label}'
            remote.job({'action':'probe','config':probe_config,'bmp':probe_remote,'out':dest,'phase_receipt':receipt})
            path=out/(label+'.png');remote.download(dest+'/raw.png',path);remote.download(dest+'/capture.json',path.with_suffix('.json'))
            a=np.array(Image.open(path));meta=read(path.with_suffix('.json'));now=settings_signature(meta)
            if actual is not None and actual!=now:raise RuntimeError('Camera settings drift')
            actual=now;v,stats=fingerprint(a,roi)
            if stats['p99']<16 or stats['std']<2 or stats['saturation_fraction']>.01:raise RuntimeError('Diagnostic probe signal invalid: '+str(stats))
            return v,stats
        with PhaseOwner(link,flat,flat) as owner:
            for index,stage in enumerate(STAGES):
                prior=[r for r in report['stages'] if r['stage']==stage]
                if prior:
                    if not prior[0]['status'].startswith('real_capture_complete'):raise ValueError('Incomplete prior stage')
                    print('RETAIN completed measured stage',stage,prior[0]['status'],flush=True);continue
                started=time.monotonic();row={'stage':stage,'status':'preparing'};report['stages'].append(row);save()
                job({'action':'prepare','session':session,'stage':stage})
                mf=remote.read(f'sessions/{session}/play/{stage}/manifest.json')
                if len(mf['entries'])!=report['expected_stage_counts'][index]:raise ValueError('Unexpected sample count')
                phase=out/(stage+'.bmp');remote.download(mf['phase_file'].replace('\\','/'),phase)
                if sha(phase)!=mf['phase_sha256']:raise ValueError('Phase hash mismatch')
                row['preflight_attempts']=[]
                for attempt in range(3):
                    tag=stage+f'_try{attempt+1}'
                    v0,_=probe(owner.show(flat),tag+'_flat')
                    receipt=owner.show(phase,mf['phase_sha256']);v1,s1=probe(receipt,tag+'_before')
                    receipt=owner.show(phase,mf['phase_sha256']);v2,s2=probe(receipt,tag+'_repeat')
                    repeat=pcc(v1,v2);challenge=pcc(v0,v2)
                    row.update(phase_sha256=mf['phase_sha256'],count=len(mf['entries']),repeat_pcc=repeat,flat_pcc=challenge)
                    row['preflight_attempts'].append({'attempt':attempt+1,'repeat_pcc':repeat,'flat_pcc':challenge});save()
                    if repeat>=.97 and repeat-challenge>=.01:break
                    if attempt==2:raise RuntimeError('Phase repeat/challenge failed after 3 attempts; no data capture: '+str(row))
                    print('Phase challenge failed; reconnect SDK, keeping display origin fixed.',flush=True)
                    owner.restart()
                row['status']='capturing';save()
                # This is deliberately NOT a fabricated approved-bank receipt.
                receipt['diagnostic_probe']={'repeat_pcc':repeat,'flat_pcc':challenge,'production_verified':False}
                job({'action':'capture','session':session,'stage':stage,'phase_receipt':receipt})
                after,sa=probe({'phase_sha256':mf['phase_sha256'],'no_phase_write':True},stage+'_after')
                post=pcc(v2,after);ratio=abs(sa['mean']/max(s2['mean'],1e-9)-1)
                row.update(post_pcc=post,post_mean_ratio_error=ratio,elapsed_s=time.monotonic()-started)
                result_state=post_state(post,ratio,brightness_warning_only)
                if result_state=='rejected_postcheck':
                    row['status']='rejected_postcheck';save();raise RuntimeError('Stage drift; no subsequent preparation/evaluation')
                # Capture creates a record only after raw/route/file checks pass.
                for e in mf['entries']:
                    record=remote.read(f"sessions/{session}/ccd/{e['id']}/{stage}.record.json")
                    if record['phase_sha256']!=mf['phase_sha256'] or record['capture_mode']!='real':raise ValueError('Capture record mismatch')
                row['status']=result_state;save();print('COMPLETED',stage,row,flush=True)
            job({'action':'evaluate','session':session})
            remote.download(f'sessions/{session}/results/metrics.json',out/'metrics.json')
            remote.download(f'sessions/{session}/results/compute_memory.json',out/'compute_memory.json')
            warnings=sum(r['status']=='real_capture_complete_photometric_warning' for r in report['stages'])
            report.update(status='completed_with_photometric_warnings' if warnings else 'completed',
                camera_actual=actual,total_real_captures=sum(report['expected_stage_counts']),photometric_stability_passed=warnings==0)
            metrics=read(out/'metrics.json');metrics['photometric_stability_passed']=warnings==0
            metrics['photometric_warning_stages']=[r['stage'] for r in report['stages'] if r['status'].endswith('photometric_warning')]
            write(out/'metrics.json',metrics)
        report['display_audit']=owner.display.audit;save()
        report['sdk_audit']=owner.audit;save()
        remote.putjson(f'sessions/{session}/results/smoke_validation.json',report)
    except BaseException as e:
        report.update(status='stopped',error=str(e));save();raise

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--link-config',type=Path,required=True)
    p.add_argument('--remote-config',required=True);p.add_argument('--limit',type=int,default=4);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--resume',action='store_true',help='Only completed stages with passing shape postchecks; never arbitrary partial capture')
    p.add_argument('--brightness-warning-only',action='store_true',help='Diagnostic flow only: retain >20%% brightness-drift warning, never certify photometric stability')
    a=p.parse_args();link=read(a.link_config)
    with Remote(link) as remote:
        c=remote.read(a.remote_config)
        run(remote,link,c,a.remote_config,a.out,a.limit,a.resume,a.brightness_warning_only)

if __name__=='__main__':main()
