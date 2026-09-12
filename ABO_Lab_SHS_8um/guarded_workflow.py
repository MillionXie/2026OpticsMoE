"""Local coordinator: alignment, reviewed fingerprint bank, bounded retries, six stages.

Never uses theoretical CCDs to replace experimental captures. GPU runs remotely.
"""
import argparse
import json
import time
import uuid
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from dual_run import Remote, STAGES
from phase_hdmi import sha
from phase_owner import PhaseOwner
from phase_fingerprint import digest, fingerprint, settings_signature, classify, validate_bank

ROOT=Path(__file__).resolve().parent

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2),encoding='utf-8');tmp.replace(path)

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def require_geometry(c):
    names=('top_left','top_right','bottom_right','bottom_left')
    points=np.asarray([c['logical_corners_full_sensor_xy'][n] for n in names],dtype=float)
    if not c.get('geometry_confirmed') or points.shape!=(4,2) or not np.isfinite(points).all():
        raise ValueError('First fill and confirm the four SHS full-sensor corners; see START_HERE.md')
    if (points<0).any() or (points[:,0]>=1920).any() or (points[:,1]>=1080).any():raise ValueError('SHS corners outside 1920x1080')
    cross=np.cross(np.roll(points,-1,axis=0)-points,np.roll(points,-2,axis=0)-np.roll(points,-1,axis=0))
    if not ((cross>1).all() or (cross< -1).all()):raise ValueError('Logical corner order is crossed/degenerate')
    return [int(np.floor(points[:,0].min())),int(np.floor(points[:,1].min())),
            min(1920,int(np.ceil(points[:,0].max()))+1),min(1080,int(np.ceil(points[:,1].max()))+1)]

def generated(c):return 'generated/phase_inverted' if c['phase_slm']['gray_encoding']=='inverted_255_minus_g' else 'generated'

def prepare_artifacts(remote,c,out):
    base=generated(c);paths={s:f'{base}/P/{i+1:02d}_{s}.bmp' for i,s in enumerate(STAGES)}
    paths.update(flat=f'{base}/cal/P_ZERO.bmp')
    # Strong lens diagnostic is independent of training masks, never encoded twice.
    strong=ROOT/'generated/phase_response_strong_20260913'
    if not (strong/'P_lens_10cm.bmp').exists():
        import subprocess,sys
        subprocess.run([sys.executable,str(ROOT/'generate_phase_response_patterns.py'),'--out',str(strong)],check=True)
    local={}
    for key,path in paths.items():
        dest=out/'patterns'/(key+'.bmp');remote.download(path,dest);local[key]=dest
    local['lens']=strong/'P_lens_10cm.bmp'
    probe=out/'patterns/A_probe.bmp';remote.download(f'{base}/cal/A_CHECK_64.bmp',probe)
    return local,probe,f'{base}/cal/A_CHECK_64.bmp'

class CaptureProbe:
    def __init__(self,remote,out,probe_remote):
        self.remote=remote;self.out=out;self.probe_remote=probe_remote
        self.remote_dir='results/phase_guard/'+out.name
        remote.ps(f"New-Item -ItemType Directory -Force -Path '{remote.root}/{self.remote_dir}' | Out-Null")
    def capture(self,receipt,label):
        tag=label+'_'+uuid.uuid4().hex[:10];dest=self.remote_dir+'/'+tag
        self.remote.job({'action':'probe','bmp':self.probe_remote,'out':dest,'phase_receipt':receipt})
        frame=self.out/(tag+'.png');self.remote.download(dest+'/raw.png',frame)
        self.remote.download(dest+'/capture.json',frame.with_suffix('.json'))
        return np.asarray(Image.open(frame)).copy(),read(frame.with_suffix('.json')),frame

def gallery(out,rows):
    thumb=(320,210);canvas=Image.new('RGB',(4*thumb[0],((len(rows)+3)//4)*thumb[1]),'white');draw=ImageDraw.Draw(canvas)
    for i,row in enumerate(rows):
        im=Image.open(out/row['file']).convert('RGB');im.thumbnail((310,180))
        x=(i%4)*320;y=(i//4)*210;canvas.paste(im,(x,y+25));draw.text((x+4,y+4),row['label'],fill='black')
    canvas.save(out/'review.png')

class Guard:
    def __init__(self,owner,probe,bank,out,link):
        self.owner=owner;self.probe=probe;self.bank=bank;self.out=out;self.link=link
        self.refs={k:np.asarray(v,dtype=float) for k,v in bank['references'].items()}
    def check(self,target,path,attempt_label):
        expected=self.bank['phase_sha256'][target]
        if sha(path)!=expected:raise ValueError('Phase changed since reference enrollment')
        receipt=self.owner.show(path,expected)
        raw,meta,file=self.probe.capture(receipt,attempt_label)
        vec,stats=fingerprint(raw,self.bank['roi_xyxy'])
        result=classify(vec,stats,self.refs,target,self.bank.get('thresholds'))
        if settings_signature(meta)!=self.bank['camera_actual']:
            result['passed']=False;result['reasons'].append('camera_actual_settings_changed')
        result.update(file=str(file),phase_sha256=expected,receipt=receipt)
        write(file.with_suffix('.verification.json'),result)
        return result
    def ensure(self,target,path,label):
        attempts=[]
        for i in range(int(self.link.get('phase_max_attempts',3))):
            # Distinct known challenge before target: a stuck target cannot pass
            # merely because two consecutive captures are identical.
            challenge='lens' if target!='lens' else 'flat'
            challenge_path=Path(self.bank['phase_paths'][challenge])
            first=self.check(challenge,challenge_path,f'{label}_try{i+1}_challenge');attempts.append(first)
            if first['passed']:
                last=self.check(target,path,f'{label}_try{i+1}_target');attempts.append(last)
                if last['passed']:
                    receipt=last['receipt'];receipt['optical_verification']={'passed':True,'bank_id':self.bank['bank_id'],
                        'target':target,'file':last['file'],'scores':last['scores'],'attempt':i+1}
                    return receipt,attempts
            print(f'Phase verification failed ({target}, attempt {i+1}); bounded recovery.',flush=True)
            self.owner.recover(int(self.link.get('phase_retry_cycles',4)))
        raise RuntimeError(f'Phase {target} failed all attempts; no formal capture. Inspect {self.out}')
    def postcheck(self,target,path,label):
        # Re-send target after batch would hide a mid-batch failure. Capture
        # current display WITHOUT writing phase, then compare to target bank.
        raw,meta,file=self.probe.capture({'phase_sha256':sha(path),'postcheck_no_phase_write':True},label)
        vec,stats=fingerprint(raw,self.bank['roi_xyxy'])
        result=classify(vec,stats,self.refs,target,self.bank.get('thresholds'))
        if settings_signature(meta)!=self.bank['camera_actual']:result['passed']=False;result['reasons'].append('camera_actual_settings_changed')
        result['file']=str(file);write(file.with_suffix('.verification.json'),result);return result

def enroll(remote,c,link,out):
    roi=require_geometry(c)
    if c['camera'].get('gain') is None:raise ValueError('Pin camera.gain explicitly (e.g. Gain_X4) before enrolling')
    local,probe_path,probe_remote=prepare_artifacts(remote,c,out)
    probe=CaptureProbe(remote,out,probe_remote);samples={k:[] for k in local};stats={k:[] for k in local};rows=[];actual=None
    report={'status':'collecting','rows':rows};write(out/'enrollment.json',report)
    with PhaseOwner(link,local['flat'],local['lens']) as owner:
        for repeat in range(3):
            names=list(local)
            if repeat==1:names.reverse()
            if repeat==2:names=names[3:]+names[:3]
            for key in names:
                receipt=owner.show(local[key]);raw,meta,file=probe.capture(receipt,f'enroll_r{repeat}_{key}')
                now=settings_signature(meta)
                if actual is not None and actual!=now:raise RuntimeError('Camera settings changed during enrollment')
                actual=now;v,s=fingerprint(raw,roi);samples[key].append(v);stats[key].append(s)
                rows.append({'label':f'r{repeat} {key}','file':file.name,'phase_sha256':sha(local[key]),'camera_actual':now})
                write(out/'enrollment.json',report)
        refs,checks=validate_bank(samples,stats,link.get('phase_thresholds'))
        report.update(status='repeatability_passed' if checks['passed'] else 'failed',checks=checks,sdk_audit=owner.audit)
        write(out/'enrollment.json',report);gallery(out,rows)
    if not checks['passed']:raise RuntimeError('Reference bank is unstable or phases indistinguishable; review enrollment.json, no bank approved')
    bank={'schema':1,'approved':False,'hardware_config_sha256':digest(c),'camera_actual':actual,'roi_xyxy':roi,
          'lut_sha256':sha(link['phase_lut']),'phase_sha256':{k:sha(v) for k,v in local.items()},
          'phase_paths':{k:str(v.resolve()) for k,v in local.items()},'probe_remote':probe_remote,'probe_sha256':sha(probe_path),
          'references':{k:v.tolist() for k,v in refs.items()},'thresholds':link.get('phase_thresholds',{}),
          'enrollment_sha256':sha(out/'enrollment.json'),'review_note':'Repeatability does not prove correct physical phase mapping; operator must review.'}
    bank['bank_id']=digest(bank);write(out/'bank.json',bank)
    print('Review',out/'review.png','then explicitly approve bank; it is NOT approved automatically.',flush=True)

def load_bank(path,c,link):
    b=read(path);base={k:v for k,v in b.items() if k not in ('bank_id','approval')};base['approved']=False
    if digest(base)!=b['bank_id']:raise ValueError('Reference bank was edited; re-enroll instead of changing reference/thresholds')
    if not b.get('approved') or b.get('approval',{}).get('bank_id')!=b['bank_id']:raise ValueError('Reference bank needs explicit review approval')
    if b['hardware_config_sha256']!=digest(c) or b['lut_sha256']!=sha(link['phase_lut']):raise ValueError('Hardware/LUT changed; new bank and session required')
    for k,p in b['phase_paths'].items():
        if sha(p)!=b['phase_sha256'][k]:raise ValueError('Reference phase file changed')
    return b

def alignment(remote,c,link,out):
    base=generated(c);strong=ROOT/'generated/phase_response_strong_20260913'
    if not (strong/'P_lens_10cm.bmp').exists():raise FileNotFoundError('Run generate_phase_response_patterns.py first')
    flat=out/'P_ZERO.bmp';remote.download(base+'/cal/P_ZERO.bmp',flat)
    pairs=[('A_WHITE','P_F4')]+[('A_WHITE','P_F_'+k) for k in ('TL','TR','BR','BL')]+[(k,'P_ZERO') for k in ('A_L','A_TL','A_TR','A_BR','A_BL')]
    rows=[]
    with PhaseOwner(link,flat,strong/'P_lens_10cm.bmp') as owner:
        for amp,phase in pairs:
            p=out/(phase+'.bmp')
            if not p.exists():remote.download(base+'/cal/'+phase+'.bmp',p)
            receipt=owner.show(p);probe=CaptureProbe(remote,out,base+'/cal/'+amp+'.bmp')
            raw,meta,file=probe.capture(receipt,amp+'_'+phase)
            rows.append({'label':amp+' '+phase,'file':file.name,'receipt':receipt})
            write(out/'alignment.json',{'verified':False,'rows':rows,'note':'Raw evidence only; identify corners/direction manually, do not trust SDK receipt alone.'})
    gallery(out,rows);print('Mark full-sensor corners from',out,'; no geometry flags were changed.')

def execute(remote,c,link,out,session,limit,bank_path):
    require_geometry(c)
    if not link.get('orientation_and_phase_response_verified'):raise ValueError('Confirm direction and phase encoding after calibration, see START_HERE.md')
    bank=load_bank(bank_path,c,link)
    if c['phase_slm'].get('lut_sha256')!=bank['lut_sha256']:raise ValueError('Pin phase_slm.lut_sha256 in LAB.local.json before enrollment')
    if c['camera'].get('gain') is None:raise ValueError('Explicit camera gain required')
    remote_probe=out/'probe.bmp';remote.download(bank['probe_remote'],remote_probe)
    if sha(remote_probe)!=bank['probe_sha256']:raise ValueError('Probe amplitude changed')
    if not remote.exists('sessions/'+session+'/session.json'):remote.job({'action':'init','session':session,'limit':limit})
    else:
        state=remote.read('sessions/'+session+'/session.json')
        if state['test_queries']!=(limit or 2400):raise ValueError('Existing session has different query count')
    probe=CaptureProbe(remote,out,bank['probe_remote']);journal=out/'journal.json'
    audit=read(journal) if journal.exists() else {'schema':1,'session':session,'bank_id':bank['bank_id'],'batches':[]}
    if audit['bank_id']!=bank['bank_id']:raise ValueError('Session bound to a different reference bank')
    # An interruption after acquisition but before verification must never make
    # unchecked records look completed on the next prepare.
    for row in audit['batches']:
        if row['state']=='pending':
            rel=f"sessions/{session}/phase_batches/{row['batch_id']}.json"
            if remote.exists(rel) and remote.read(rel)['state']=='accepted':
                row['state']='accepted_reconciled'
            else:
                remote.job({'action':'quarantine_batch','session':session,'batch_id':row['batch_id']})
                row['state']='quarantined_after_interruption'
            write(journal,audit)
    with PhaseOwner(link,bank['phase_paths']['flat'],bank['phase_paths']['lens']) as owner:
        guard=Guard(owner,probe,bank,out,link)
        for stage in STAGES:
            remote.job({'action':'prepare','session':session,'stage':stage})
            mf=remote.read(f'sessions/{session}/play/{stage}/manifest.json')
            if not mf['entries']:continue
            phase=out/(stage+'.bmp');remote.download(mf['phase_file'].replace('\\','/'),phase)
            if sha(phase)!=bank['phase_sha256'][stage]:raise ValueError('Prepared phase differs from reviewed bank')
            size=int(link.get('capture_batch_size',32))
            for offset in range(0,len(mf['entries']),size):
                ids=[e['id'] for e in mf['entries'][offset:offset+size]]
                for retry in range(int(link.get('phase_max_attempts',3))):
                    receipt,attempts=guard.ensure(stage,phase,f'{stage}_{offset}_r{retry}')
                    bid=uuid.uuid4().hex;row={'batch_id':bid,'stage':stage,'ids':ids,'state':'pending','precheck':receipt['optical_verification']}
                    audit['batches'].append(row);write(journal,audit)
                    try:
                        remote.job({'action':'capture_batch','session':session,'stage':stage,'ids':ids,'batch_id':bid,'phase_receipt':receipt})
                        post=guard.postcheck(stage,phase,f'{stage}_{offset}_post_r{retry}');row['postcheck']=post
                        if not post['passed']:raise RuntimeError('Phase postcheck failed; batch cannot be used')
                        remote.job({'action':'accept_batch','session':session,'batch_id':bid,'verification':{'bank_id':bank['bank_id'],'postcheck':post}})
                        row['state']='accepted';write(journal,audit);break
                    except BaseException:
                        rel=f'sessions/{session}/phase_batches/{bid}.json'
                        if remote.exists(rel) and remote.read(rel)['state']=='accepted':
                            row['state']='accepted_reconciled';write(journal,audit);break
                        remote.job({'action':'quarantine_batch','session':session,'batch_id':bid})
                        row['state']='quarantined';write(journal,audit)
                        if retry+1>=int(link.get('phase_max_attempts',3)):raise
                        if not isinstance(__import__('sys').exc_info()[1],Exception):raise
                        owner.recover(int(link.get('phase_retry_cycles',4)))
        remote.job({'action':'evaluate','session':session})
        remote.download(f'sessions/{session}/results/metrics.json',out/'metrics.json')
        result=read(out/'metrics.json')
        result['phase_guard']={'bank_id':bank['bank_id'],'batch_size':int(link.get('capture_batch_size',32)),
            'accepted_batches':sum(x['state'].startswith('accepted') for x in audit['batches']),
            'quarantined_batches':sum(x['state'].startswith('quarantined') for x in audit['batches']),
            'verification_scope':'before and after each batch; NOT a per-frame hardware acknowledgement'}
        write(out/'metrics.json',result)
        write(out/'sdk_audit.json',owner.audit)
    print('Real six-stage results:',out/'metrics.json')

def main(default_action=None):
    p=argparse.ArgumentParser(description=__doc__)
    if default_action is None:p.add_argument('action',choices=['alignment','enroll','approve','run'])
    p.add_argument('--link-config',type=Path,default=ROOT/'dual.local.json');p.add_argument('--out',type=Path)
    p.add_argument('--bank',type=Path);p.add_argument('--confirm-reference-images',action='store_true')
    p.add_argument('--session');p.add_argument('--limit',type=int,default=4);a=p.parse_args();action=default_action or a.action
    if action=='approve':
        if not a.bank or not a.confirm_reference_images:raise ValueError('Review images first; explicitly pass --bank and --confirm-reference-images')
        b=read(a.bank)
        if sha(a.bank.parent/'enrollment.json')!=b['enrollment_sha256']:raise ValueError('Enrollment report changed')
        if not read(a.bank.parent/'enrollment.json')['checks']['passed']:raise ValueError('Failed bank cannot be approved')
        b['approved']=True;b['approval']={'bank_id':b['bank_id'],'operator_confirmed_reference_images':True,'time':time.strftime('%Y-%m-%dT%H:%M:%S')};write(a.bank,b);print('Approved',b['bank_id']);return
    link=read(a.link_config)
    if not 1<=int(link.get('phase_max_attempts',3))<=5 or not 1<=int(link.get('capture_batch_size',32))<=128:raise ValueError('Retries 1..5, batch size 1..128')
    if action=='run':
        import re
        if not a.session or not re.fullmatch('[A-Za-z0-9_-]{1,80}',a.session) or not 0<=a.limit<=2400:raise ValueError('Valid session and limit 0..2400 required')
        out=ROOT/'results/guarded_runs'/a.session;out.mkdir(parents=True,exist_ok=True)
        bank=a.bank or (Path(link['phase_reference_bank']) if link.get('phase_reference_bank') else None)
        if not bank:raise ValueError('Enroll and approve reference bank first')
    else:
        out=a.out or ROOT/'results'/('phase_'+action+'_'+time.strftime('%Y%m%d_%H%M%S'))
        out.mkdir(parents=True,exist_ok=False)
    with Remote(link) as remote:
        c=remote.read('LAB.local.json')
        if action in ('alignment','enroll'):remote.job({'action':'calibrate'})
        if action=='alignment':alignment(remote,c,link,out)
        elif action=='enroll':enroll(remote,c,link,out)
        else:execute(remote,c,link,out,a.session,a.limit,bank)

if __name__=='__main__':main()
