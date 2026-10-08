"""Validation-only optimization controls; archived trainer/model remain unchanged."""
import argparse,copy,json,os,subprocess,sys,traceback
from pathlib import Path
import kather2016_experiment as k
from kather2016_experiment import b,m,r,torch,np

HERE=Path(__file__).resolve()
PROFILE=HERE.with_name('kather_followup_profiles.json')
SPEC=r.read(PROFILE)
old_config=k.config
old_sources=k.sources
BASE_SOURCES=old_sources()
old_encode=b.encode
drop_probability=0.
drop_generator=None
mask_hashes=[]

def config(name):
    c=old_config('base');c.update(SPEC['candidates'][name]);return c

def sources():
    d=dict(BASE_SOURCES)
    for p in [HERE,PROFILE]:d[p.relative_to(b.TASK).as_posix()]=r.sha(p)
    return d

def encode(x,theta=None,cnn=False):
    x=old_encode(x,theta,cnn)
    if theta is not None and drop_probability:
        mask=(torch.rand(x.shape,generator=drop_generator)>drop_probability)
        if len(mask_hashes)<8:mask_hashes.append(r.sha_tensor(mask))
        y=x*mask.to(x.device)
        x=y*(x.square().sum((1,2,3),keepdim=True)/y.square().sum((1,2,3),keepdim=True).clamp_min(1e-12)).sqrt()
    return x

b.encode=encode
k.config=config
k.sources=sources
m.sources=sources

def train_one(a):
    global drop_probability,drop_generator,mask_hashes
    drop_probability=config(a.candidate)['input_dropout'];drop_generator=torch.Generator().manual_seed(a.seed+901234);mask_hashes=[]
    k.train_one(a)
    r.save(a.out/'input_dropout_audit.json',dict(probability=drop_probability,first_batch_mask_hashes=mask_hashes,contract=SPEC['dropout_contract']))

def coordinator(a):
    a.out.mkdir(parents=True,exist_ok=False)
    r.save(a.out/'metadata.json',dict(command=sys.argv,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),sources=sources(),data_sha256=r.sha(a.data),specification=SPEC,time=r.now(),test_read=False,test_previously_observed=True))
    try:
        # Reuse the queue implementation, but launch this entry point.
        original_file=k.__file__;k.__file__=str(HERE)
        if a.phase=='pilot':
            jobs=[k.job(c,arch,SPEC['pilot_depth'],17) for c in SPEC['candidate_order'] for arch in SPEC['architectures']]
            entries=k.run_jobs(a,jobs)
            means={c:float(np.mean([e['result']['metrics']['val']['balanced_nll'] for e in entries if e['candidate']==c])) for c in SPEC['candidate_order']}
            chosen=min(SPEC['candidate_order'],key=means.get)
            r.save(a.out/'candidate_selection.json',dict(chosen=chosen,mean_validation_balanced_nll=means,entries=entries,sources=sources(),data_sha256=r.sha(a.data),test_read=False,time=r.now()))
            r.save(a.out/'status.json',dict(state='pilot_complete_test_not_read',chosen=chosen,time=r.now()))
        else:
            sel=r.read(a.pilot/'candidate_selection.json');assert sel['sources']==sources() and sel['data_sha256']==r.sha(a.data)
            chosen=sel['chosen'];reuse=[dict(e,reused=True) for e in sel['entries'] if e['candidate']==chosen]
            jobs=[k.job(chosen,arch,d,s) for s in SPEC['seeds'] for d in SPEC['depths'] for arch in SPEC['architectures'] if not(s==17 and d==SPEC['pilot_depth'])]
            entries=reuse+k.run_jobs(a,jobs);assert len(entries)==36
            r.save(a.out/'selection_lock.json',dict(entries=entries,config=config(chosen),candidate=chosen,pilot_selection_sha256=r.sha(a.pilot/'candidate_selection.json'),sources=sources(),data_sha256=r.sha(a.data),time=r.now()))
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(a.gpus[0])
            subprocess.run([sys.executable,'-u',str(HERE),'--phase','evaluate','--data',str(a.data),'--out',str(a.out)],env=env,check=True)
            r.save(a.out/'status.json',dict(state='complete',time=r.now()))
        k.__file__=original_file
    except Exception:
        r.save(a.out/'status.json',dict(state='failed',traceback=traceback.format_exc(),time=r.now()));raise

def smoke(a):
    global drop_probability,drop_generator,mask_hashes
    a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);data=k.load_data(a.data,'train');x=data[0][:8];theta=b.affine_parameters(8,17,1,config('long')['augmentation'])
    drop_probability=.05;drop_generator=torch.Generator().manual_seed(901251)
    clean=old_encode(x,theta);masked=encode(x,theta)
    err=float(((masked.square().sum((1,2,3))-clean.square().sum((1,2,3))).abs()/clean.square().sum((1,2,3))).max());assert err<1e-6
    assert torch.equal(encode(x),old_encode(x))
    drop_generator=torch.Generator().manual_seed(901251);assert torch.equal(masked,encode(x,theta))
    checks=[]
    for arch in SPEC['architectures']:
        r.setseed(17);model=m.build(arch,6,config('long'));prob,c,_=b.forward(model,masked,arch);loss=-prob[torch.arange(8),data[1][:8]].clamp_min(1e-12).log().mean()-.02*c.log().mean();loss.backward()
        grad={n:float(p.grad.norm()) for n,p in model.named_parameters()};assert all(np.isfinite(v) and v>0 for v in grad.values());checks.append(dict(arch=arch,parameters=sum(p.numel() for p in model.parameters()),gradients=grad));del model;torch.cuda.empty_cache()
    r.save(a.out/'smoke.json',dict(passed=True,power_relative_error=err,evaluation_identity=True,paired_masks_equal=True,models=checks,sources=sources(),time=r.now()))

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['smoke','pilot','suite','worker','train-one','evaluate'],required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--pilot',type=Path);p.add_argument('--gpus',nargs='+',default=['0']);p.add_argument('--candidate',choices=SPEC['candidate_order'],default='long');p.add_argument('--arch',choices=SPEC['architectures']);p.add_argument('--depth',type=int);p.add_argument('--seed',type=int);a=p.parse_args()
    assert len(a.gpus)<=5 and len(a.gpus)==len(set(a.gpus))
    if a.phase in ['pilot','suite']:coordinator(a)
    elif a.phase=='worker':k.__file__=str(HERE);k.worker(a)
    elif a.phase=='train-one':train_one(a)
    elif a.phase=='evaluate':k.evaluate(a)
    else:smoke(a)

if __name__=='__main__':main()
