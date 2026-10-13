"""Durable once-only final protocol primitives. Caller must independently CPU-audit fits."""
import hashlib,json,math,os,re,time
from pathlib import Path
KEYS=tuple(f'L{d}_{r}' for d in (2,3,4) for r in ('rho0','rho03'))
DATA_SHA='21436ad0b269d5319176e3e4d40fd382b21e1d34d41345ec979176231fb0e3c0'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def identity(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def valid_sha(value):return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None

def durable(path,value):
    if os.name!='posix':raise RuntimeError('Linux directory-fsync runtime required; no durability downgrade')
    payload=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode();path=Path(path)
    # Any partial claim survives errors; never unlink a failed or existing marker.
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'wb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)

def validate_audit(receipt):
    assert receipt['receipt_kind']=='formal_repaired_saved_CPU_fit_audit' and receipt['passed'] and receipt['all_six_fit_audited'] and receipt['all_fit_processes_exited']
    assert receipt['test_evaluations']==0 and receipt['test_arrays_materialized']==0 and receipt['data_sha256']==DATA_SHA
    assert valid_sha(receipt['formal_profile_sha256']) and set(receipt['fits'])==set(KEYS)
    fits=receipt['fits']
    for key in KEYS:
        f=fits[key];depth=int(key[1]);rho=0. if key.endswith('rho0') else .3
        assert f['depth']==depth and f['rho']==rho and 1<=f['selected_epoch']<=f['epochs_completed']<=150
        assert f['selection_split']=='val' and f['selected_state_kind']=='ema' and f['saved_CPU_verified'] and f['test_evaluations']==0
        assert valid_sha(f['checkpoint_sha256']) and valid_sha(f['initial_parameter_sha256']) and f['data_sha256']==DATA_SHA
        assert f['sources'] and all(valid_sha(x) for x in f['sources'].values())
    for depth in (2,3,4):
        left,right=[fits[f'L{depth}_{r}'] for r in ('rho0','rho03')]
        assert left['initial_parameter_sha256']==right['initial_parameter_sha256'] and left['sources']==right['sources']
        configs=[dict(f['config']) for f in (left,right)]
        for c,rho in zip(configs,(0.,.3)):
            assert c['depth']==depth and c['residual_rho']==rho;c.pop('residual_rho');c['contract']=dict(c['contract']);assert c['contract'].pop('rho')==rho
        assert configs[0]==configs[1]
        n=min(left['epochs_completed'],right['epochs_completed']);assert len(left['orders'])==left['epochs_completed'] and len(right['orders'])==right['epochs_completed'] and left['orders'][:n]==right['orders'][:n]
        assert len(left['transforms'])==left['epochs_completed'] and len(right['transforms'])==right['epochs_completed'] and left['transforms'][:n]==right['transforms'][:n]
    return receipt

def seal(out,receipt):
    out=Path(out);assert not out.exists(),'Never retry a completed/partial lock directory'
    validate_audit(receipt);out.mkdir(parents=True,exist_ok=False)
    fd=os.open(out.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    durable(out/'cpu_fit_audit.json',receipt)
    for depth in (2,3,4):durable(out/f'pair_lock_L{depth}.json',dict(depth=depth,keys=[f'L{depth}_rho0',f'L{depth}_rho03'],config_only_rho_verified=True))
    locks={}
    for key in KEYS:
        f=receipt['fits'][key];(out/key).mkdir();value=dict(key=key,epoch=f['selected_epoch'],checkpoint_sha256=f['checkpoint_sha256'],data_sha256=DATA_SHA,formal_profile_sha256=receipt['formal_profile_sha256'],config=f['config'],sources=f['sources'],selection_split='val',selected_state_kind='ema',test_evaluations_before_lock=0)
        durable(out/key/'selection_lock.json',value);locks[key]=sha(out/key/'selection_lock.json')
    durable(out/'global_selection_lock.json',dict(receipt_kind='formal_repaired_all6_selection_lock',locks=locks,data_sha256=DATA_SHA,formal_profile_sha256=receipt['formal_profile_sha256'],all_six_fit_audited=True,cpu_fit_audit_sha256=sha(out/'cpu_fit_audit.json'),test_evaluations_before_lock=0))

def validate_locks(out):
    out=Path(out);global_lock=read(out/'global_selection_lock.json');assert global_lock['receipt_kind']=='formal_repaired_all6_selection_lock' and global_lock['all_six_fit_audited'] and global_lock['test_evaluations_before_lock']==0 and global_lock['data_sha256']==DATA_SHA and set(global_lock['locks'])==set(KEYS)
    assert sha(out/'cpu_fit_audit.json')==global_lock['cpu_fit_audit_sha256'];proof=validate_audit(read(out/'cpu_fit_audit.json'));assert proof['formal_profile_sha256']==global_lock['formal_profile_sha256']
    for key,h in global_lock['locks'].items():
        assert sha(out/key/'selection_lock.json')==h;s=read(out/key/'selection_lock.json');f=proof['fits'][key]
        assert s['key']==key and s['checkpoint_sha256']==f['checkpoint_sha256'] and s['epoch']==f['selected_epoch'] and s['data_sha256']==DATA_SHA and s['formal_profile_sha256']==global_lock['formal_profile_sha256'] and s['config']==f['config'] and s['sources']==f['sources'] and s['selected_state_kind']=='ema' and s['selection_split']=='val'
    return global_lock

def claim_suite(out):
    lock=validate_locks(out);durable(Path(out)/'suite_started.json',dict(pid=os.getpid(),time_unix=time.time(),locks=lock['locks'],data_sha256=DATA_SHA,formal_profile_sha256=lock['formal_profile_sha256']))

def test_once(out,key,checkpoint,data,load_and_evaluate):
    # Callback may materialize test arrays only after this durable exclusive claim.
    out=Path(out);assert key in KEYS;lock=validate_locks(out);suite=read(out/'suite_started.json');assert suite['locks']==lock['locks'] and suite['data_sha256']==DATA_SHA and suite['formal_profile_sha256']==lock['formal_profile_sha256']
    assert not (out/key/'test_completed.json').exists() and not (out/key/'test_failure.json').exists()
    s=read(out/key/'selection_lock.json')
    assert sha(checkpoint)==s['checkpoint_sha256'] and sha(data)==s['data_sha256']
    durable(out/key/'test_started.json',dict(pid=os.getpid(),time_unix=time.time(),checkpoint_sha256=s['checkpoint_sha256'],data_sha256=s['data_sha256'],formal_profile_sha256=s['formal_profile_sha256'],selection_lock_sha256=sha(out/key/'selection_lock.json')))
    try:
        result=load_and_evaluate(s)
        durable(out/key/'test_completed.json',dict(test_evaluations=1,checkpoint_sha256=s['checkpoint_sha256'],data_sha256=s['data_sha256'],test_used_for_selection=False,result=result))
        return result
    except BaseException as e:
        durable(out/key/'test_failure.json',dict(error=repr(e),automatic_retry=False,test_evaluations_attempted=1));raise
