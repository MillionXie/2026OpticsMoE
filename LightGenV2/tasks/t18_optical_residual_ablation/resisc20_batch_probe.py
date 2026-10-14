"""User-authorized bounded microbatch speed/parity probe; original runs untouched."""
import argparse,io,json,os,statistics,subprocess,time
from pathlib import Path
import numpy as np
import torch
import resisc20 as s
HERE=s.HERE;ROOT=s.ROOT
BASE=HERE/'runs/simulation/resisc20_s17_mainL4_rho0rho03_fullres_validate30_sharedgpu256_20261014'
MICROS=(1,2,4,8)
def sources():
 src=s.sources();src[Path(__file__).relative_to(ROOT).as_posix()]=s.sha(__file__);return src

def compute(model,x,y,weights,cfg,micro,backward=True):
 model.zero_grad(set_to_none=True);ps=[];captures=[]
 with torch.set_grad_enabled(backward):
  for start in range(0,len(x),micro):
   p,c=model(x[start:start+micro]);ps.append(p.detach());captures.append(c.detach())
   if backward:(s.engine.objective(model,p,c,y[start:start+micro],weights,cfg)*len(p)/len(x)).backward()
 gradients={n:p.grad.detach().clone() for n,p in model.named_parameters()} if backward else {}
 return torch.cat(ps),torch.cat(captures),gradients

def compare(reference,candidate):
 p,c,g=reference;q,d,h=candidate
 error=float((p-q).abs().max());capture_error=float((c-d).abs().max())
 rel=max(float((g[n]-v).norm()/g[n].norm().clamp_min(1e-30)) for n,v in h.items())
 cos=min(float((g[n].double()*v.double()).sum()/(g[n].double().norm()*v.double().norm()).clamp_min(1e-30)) for n,v in h.items())
 finite=all(torch.isfinite(v).all() and v.norm()>0 for v in h.values())
 passed=error<=1e-6 and capture_error<=1e-6 and rel<=.02 and cos>=.999 and bool(finite)
 return dict(passed=passed,probability_max_error=error,capture_max_error=capture_error,gradient_max_relative_l2=rel,gradient_min_cosine=cos,all11phase_gradients_finite_nonzero=bool(finite))

def run(a):
 assert a.gpu==s.ALLOWED[5] and os.environ.get('CUDA_VISIBLE_DEVICES')==a.gpu and not a.out.exists()
 s.slots();cfg=s.bind();meta=s.read(BASE/'metadata.json');assert meta['git_commit']=='513f0e884a16be5f37846974deaeea55fac95f94'
 s.engine.verify_sources(meta['sources'],meta['git_commit']);src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();s.engine.verify_sources(src,commit)
 s.gpu_budget.ALLOWED.update(s.ALLOWED.values());policy=s.gpu_budget.configure(torch,a.gpu,shared=True,memory_gib=2);torch.set_num_threads(4)
 a.out.mkdir(parents=True,exist_ok=False);s.save(a.out/'claim.json',dict(pid=os.getpid(),command=list(__import__('sys').argv),git_commit=commit,sources=src,automatic_retry=False,test_evaluations=0),exclusive=True)
 m=s.manifest(Path(cfg['data']));data=s.engine.load_fit(Path(cfg['data']),'train',m);weights=(len(data[1])/(20*torch.bincount(data[1],minlength=20).float())).cuda()
 class_idx=np.asarray([int((data[1]==k).nonzero()[0]) for k in range(20)]);_,theta=s.engine.epoch_plan(1,len(data[1]),cfg)
 class_x,class_y,_=s.engine.mixed_batch(data,class_idx,theta,1,0,cfg,'cuda');cases=[]
 for rho,key in ((0.,'L4_rho0'),(.3,'L4_rho03')):
  blob=(BASE/key/'last_checkpoint.pt').read_bytes();snap=a.out/(key+'_immutable_last.pt')
  with snap.open('xb') as f:f.write(blob)
  ck=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=False);assert ck['sources']==meta['sources'] and ck['config']['microbatch']==1 and ck['data_sha256']==m['cache_sha256']
  s.engine.seed(cfg['seed']);model=s.r.CorrectedMoE(4,rho).cuda();fresh={n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
  epoch=ck['epoch']+1;order,theta_epoch=s.engine.epoch_plan(epoch,len(data[1]),cfg)
  inputs=[]
  for label,idx,batch in (('macro16',order[:16],0),('tail14',order[-14:],699)):
   x,y,_=s.engine.mixed_batch(data,idx,theta_epoch,epoch,batch,cfg,'cuda');inputs.append((label,x,y))
  records={micro:dict(microbatch=micro,gates=[],timings=[],eligible=True) for micro in MICROS}
  for state_name,state in (('fresh',fresh),('saved_raw',ck['model'])):
   model.load_state_dict(state,strict=True)
   for label,x,y in [('classes20',class_x,class_y)]+inputs:
    reference=compute(model,x,y,weights,cfg,1)
    for micro in MICROS:
     if not records[micro]['eligible']:continue
     try:
      torch.cuda.reset_peak_memory_stats();candidate=compute(model,x,y,weights,cfg,micro);value=compare(reference,candidate);value.update(state=state_name,samples=label,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
     except torch.cuda.OutOfMemoryError as error:
      value=dict(passed=False,state=state_name,samples=label,error=repr(error),automatic_retry=False);model.zero_grad(set_to_none=True);torch.cuda.empty_cache()
     records[micro]['gates'].append(value);records[micro]['eligible']&=value['passed']
  model.load_state_dict(ck['ema'],strict=True)
  with torch.no_grad():
   reference=compute(model,class_x,class_y,weights,cfg,1,False)
   for micro in MICROS:
    if not records[micro]['eligible']:continue
    candidate=compute(model,class_x,class_y,weights,cfg,micro,False);error=float((reference[0]-candidate[0]).abs().max());capture_error=float((reference[1]-candidate[1]).abs().max())
    value=dict(passed=error<=1e-6 and capture_error<=1e-6,state='saved_EMA_eval',probability_max_error=error,capture_max_error=capture_error)
    records[micro]['gates'].append(value);records[micro]['eligible']&=value['passed']
  model.load_state_dict(ck['model'],strict=True);x,y=inputs[0][1:]
  for micro in MICROS:
   if not records[micro]['eligible']:continue
   compute(model,x,y,weights,cfg,micro);torch.cuda.synchronize();times=[]
   for repeat in range(3):
    begin=time.perf_counter();compute(model,x,y,weights,cfg,micro);torch.cuda.synchronize();times.append(time.perf_counter()-begin)
   records[micro]['timings']=times;records[micro]['median_macro16_forward_backward_seconds']=statistics.median(times)
  one=records[1]['median_macro16_forward_backward_seconds']
  for value in records.values():
   if value['eligible']:value['speedup_vs_micro1']=one/value['median_macro16_forward_backward_seconds']
  cases.append(dict(rho=rho,source_epoch=ck['epoch'],source_last_sha256=s.sha(snap),source_initial_parameter_sha256=ck['initial_parameter_sha256'],checks=list(records.values())))
  del model;torch.cuda.empty_cache();s.save(a.out/'progress.json',dict(cases=cases,test_evaluations=0))
 common=[micro for micro in MICROS if all(next(x for x in c['checks'] if x['microbatch']==micro)['eligible'] for c in cases)]
 chosen=min(common,key=lambda micro:sum(next(x for x in c['checks'] if x['microbatch']==micro)['median_macro16_forward_backward_seconds'] for c in cases))
 proof=dict(passed=True,checks=cases,common_eligible_microbatches=common,chosen_microbatch=chosen,policy=policy,training_source_commit=meta['git_commit'],git_commit=commit,sources=src,train_samples=len(data[1]),effective_batch_size=16,optimizer_steps_per_epoch=700,tail_batch=14,maximum_probe_gpu_children=1,optimizer_steps=0,test_arrays_read=0,test_evaluations=0,old_runs_unchanged=True,scope='Bounded speed/numerical probe on GPU5 only; no fit candidate, optimizer update, stop, migration, reduced epoch data or test. Timings omit encoding/optimizer/evaluation/I/O and are not measured full-epoch speed.')
 s.save(a.out/'checks.json',proof,exclusive=True);return proof
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--gpu',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 try:print(json.dumps(run(a)))
 except BaseException as e:
  if a.out.exists():s.save(a.out/'failure.json',dict(error=repr(e),automatic_retry=False,test_evaluations=0),exclusive=True)
  raise
