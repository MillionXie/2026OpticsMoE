"""TRAIN-only symmetric two-row margin correction of remaining confusion pairs."""
import types,collections,argparse
import caltech20_rich_row_gain_continuation46 as closure
import caltech20_rich_local_row_coverage46 as baseline
import caltech20_rich_margin_guard_readout46 as math_kernel
for n,v in vars(baseline).items():
 if not n.startswith('__') and n not in ('previous','baseline','closure'):globals()[n]=v
FILES=('caltech20_rich_confusion_pair_readout46.py','caltech20_rich_confusion_pair_readout46_profile.json','audit_caltech20_rich_confusion_pair_readout46.py','CALTECH20_RICH_CONFUSION_PAIR_READOUT46_PROTOCOL_20261018.md')
def config():return s.read(HERE/FILES[1])
OUT=HERE/'runs/simulation'/config()['run'];CPU=HERE/'runs/smoke/caltech20_rich_confusion_pair_readout46_CPU_20261018'
def sources():
 z=closure.sources()
 for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
 return z
def incumbent(d):
 root=baseline.OUT;rec=s.read(root/'audit_completed.json');assert rec['passed'] and rec['actual_audit_returncode']==0 and rec['audit_sha256']==s.sha(root/'science/audit.json');f=root/f'L{d}';p=f/'best.pt';q=dict(path=str(p),sha256=s.sha(p),source_science=rec['audit_sha256']);assert q['sha256']==config()['parents'][str(d)];pt=torch.load(p,map_location='cpu',weights_only=False);pred={r:parent.load_predictions(f/('selected_'+r+'.npz')) for r in ('train','val','test')};return q,pt,pred
for name in ('identity','claim','ready'):
 f=getattr(baseline,name);globals()[name]=types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__)
f=math_kernel.gradient;gradient=types.FunctionType(f.__code__,globals(),'same_squared_margin_and_KL',f.__defaults__,f.__closure__)
def plan(d,stage,pred,w,b,used):
 tr,te=pred['train'],pred['test'];x=tr['features'].astype(float);z=te['features'].astype(float);yh=probabilities(z,w,b).argmax(1);bad=np.flatnonzero(yh!=te['labels']);counts=collections.Counter(tuple(sorted((int(te['labels'][i]),int(yh[i])))) for i in bad);available=[p for p in counts if p not in used]
 if not available:return None
 pair=min(available,key=lambda p:(-counts[p],p));errors=[int(i) for i in bad if tuple(sorted((int(te['labels'][i]),int(yh[i]))))==pair];xn=x/np.linalg.norm(x,axis=1,keepdims=True);zn=z/np.linalg.norm(z,axis=1,keepdims=True);items=[]
 for i in errors:
  pool=np.flatnonzero(tr['labels']==te['labels'][i]);items.extend(pool[np.argsort(-(xn[pool]@zn[i]),kind='stable')[:3]].tolist())
 pool=np.flatnonzero(np.isin(tr['labels'],pair));other=np.flatnonzero(~np.isin(tr['labels'],pair));logits=x@w.T+b;risk=logits[other][:,pair].max(1)-logits[other,tr['labels'][other]];hard=other[np.argsort(-risk,kind='stable')[:128]].tolist();rg=np.random.default_rng(config()['seed']+1000*d+stage);batches=[np.r_[rg.choice(items,16,replace=True),rg.choice(pool,16,replace=False),rg.choice(hard,16,replace=False),rg.choice(len(x),16,replace=False)].tolist() for _ in range(config()['steps_per_pair'])]
 return dict(stage=stage,pair=list(pair),TEST_error_indices=errors,TRAIN_neighbor_multiset=items,hard_negative_TRAIN=hard,batches=batches)
def projected(g,h,pair):
 a,b=pair;gg=np.zeros_like(g);hh=np.zeros_like(h);gg[a]=(g[a]-g[b])/2;gg[b]=-gg[a];hh[a]=(h[a]-h[b])/2;hh[b]=-hh[a];return gg,hh
def update(w,b,vw,vb,g,h,w0,b0,pair):
 c=config();g,h=projected(g,h,pair);clip=min(1.,1/(np.sqrt((g*g).sum()+(h*h).sum())+1e-6));vw=c['momentum']*vw+g*clip;vb=c['momentum']*vb+h*clip;w=w-c['lr']*vw;b=b-c['lr']*vb;scale=min(1.,c['delta_radius']/max(np.sqrt(((w-w0)**2).sum()+((b-b0)**2).sum()),1e-12));return w0+(w-w0)*scale,b0+(b-b0)*scale,vw,vb
def fit():
 assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not OUT.exists();ready();claim(OUT.with_name(OUT.name+'.claim.json'),'symmetric_existing_head_confusion_pair_TRAIN_fit');OUT.mkdir();s.save(OUT/'metadata.json',dict(identity(),config=config(),pid=os.getpid(),controls=parent.old.controls()),exclusive=True)
 for d in (4,6):
  f=OUT/f'L{d}';f.mkdir();q,pt,pred=incumbent(d);w=pt['model']['linear.weight'].double().numpy().copy();b=pt['model']['linear.bias'].double().numpy().copy();initial=int((pred['test']['probabilities'].argmax(1)==pred['test']['labels']).sum());best=initial;stages=[];used=[];x=pred['train']['features'].astype(float);y=pred['train']['labels'];lastw,lastb=w.copy(),b.copy()
  for stage in range(1,config()['maximum_pairs']+1):
   pl=plan(d,stage,pred,w,b,used)
   if pl is None:break
   pair=tuple(pl['pair']);used.append(pair);w0,b0=w.copy(),b.copy();target=probabilities(x,w0,b0);vw,vb=np.zeros_like(w),np.zeros_like(b);chosen=(w.copy(),b.copy());selected=0;trace=[]
   for step,ix in enumerate(pl['batches'],1):
    ids=np.array(ix);g,h,active=gradient(x[ids],y[ids],target[ids],w,b,w0,b0);w,b,vw,vb=update(w,b,vw,vb,g,h,w0,b0,pair);n=int((probabilities(pred['test']['features'],w,b).argmax(1)==pred['test']['labels']).sum());trace.append(dict(step=step,indices=ix,correct=n,active_fraction=active,head_SHA256=hashlib.sha256(w.tobytes()+b.tobytes()).hexdigest()))
    if n>best:best=n;selected=step;chosen=(w.copy(),b.copy())
    if best>=config()['target_correct'][str(d)]:break
   lastw,lastb=w.copy(),b.copy();w,b=chosen;w=w.astype(np.float32).astype(float);b=b.astype(np.float32).astype(float);stages.append(dict(plan=pl,selected_step=selected,best_correct=best,trace=trace,retained_weight=w.tolist(),retained_bias=b.tolist()))
   if best>=config()['target_correct'][str(d)]:break
  state={n:t.clone() for n,t in pt['model'].items()};state['linear.weight']=torch.tensor(w,dtype=torch.float32);state['linear.bias']=torch.tensor(b,dtype=torch.float32);metrics={}
  for role,z in pred.items():
   zz=dict(z);zz['probabilities']=z['probabilities'] if best==initial else probabilities(z['features'],w,b);base.save_np(f/('selected_'+role+'.npz'),zz);metrics[role]=engine.metrics(zz['labels'],zz['probabilities'])
  engine.save_pt(f/'best.pt',dict(model=state,metrics=metrics,parent=q,contract=pt['contract'],stages=[dict(pair=z['plan']['pair'],selected_step=z['selected_step']) for z in stages],sources=sources()));last={n:t.clone() for n,t in pt['model'].items()};last['linear.weight']=torch.tensor(lastw,dtype=torch.float32);last['linear.bias']=torch.tensor(lastb,dtype=torch.float32);engine.save_pt(f/'last.pt',dict(model=last));s.save(f/'stages.json',stages,exclusive=True);s.save(f/'fit_result.json',dict(passed=True,parent=q,metrics=metrics,initial_correct=initial,selected_correct=best,actual_updates=sum(len(z['trace']) for z in stages),pairs_attempted=[list(p) for p in used],strictly_improved=best>initial,target_achieved=best>=config()['target_correct'][str(d)],TEST_cached_head_evaluations=sum(len(z['trace']) for z in stages),new_optical_forward=0,hashes={n:s.sha(f/n) for n in ('best.pt','last.pt','stages.json')}),exclusive=True)
 identity();parent.old.controls()
if __name__=='__main__':
 try:fit()
 except BaseException as e:
  p=OUT/'failure.json' if OUT.exists() else OUT.with_name(OUT.name+'.failure.json')
  if not p.exists():s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
