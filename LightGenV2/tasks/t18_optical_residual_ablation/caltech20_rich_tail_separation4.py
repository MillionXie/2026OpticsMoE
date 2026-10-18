"""L4 adapter to the same reviewed tail-training kernels, warm from82.48%."""
import types,inspect
import caltech20_rich_tail_separation6 as baseline
for n,v in vars(baseline).items():
 if not n.startswith('__') and n not in ('baseline','closure'):globals()[n]=v
FILES=('caltech20_rich_tail_separation4.py','caltech20_rich_tail_separation4_profile.json','check_caltech20_rich_tail_separation4.py','audit_caltech20_rich_tail_separation4.py','CALTECH20_RICH_TAIL_SEPARATION4_PROTOCOL_20261018.md')
TRAINABLE=('phase_sets.9','electronic.3.local.weight','fusion_logits.3','linear.weight','linear.bias')
def config():return s.read(HERE/FILES[1])
OUT=HERE/'runs/simulation'/config()['run'];CPU=HERE/'runs/smoke/caltech20_rich_tail_separation4_CPU_20261018'
def sources():
 z=baseline.sources()
 for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
 return z
def parent_data():
 q,pt,pred=baseline.closure.incumbent();assert q['sha256']==config()['parent_SHA256'];q=dict(q,module='caltech20_rich_prototype_consistency',class_name='Alpha40MoE');return q,pt,pred
def model_for():
 q,pt,pred=parent_data();mod=importlib.import_module(q['module']);model=getattr(mod,q['class_name'])(4,.3);model.load_state_dict(pt['model']);model.full_rms_gradient=True;model.robust=None;model.linear.mask=None
 for n,p in model.named_parameters():p.requires_grad_(n in TRAINABLE)
 assert tuple(n for n,p in model.named_parameters() if p.requires_grad)==TRAINABLE
 return model,q,pt,pred
for name in ('identity','claim','frozen_digest','plan','triplet_scale','inputs','gradients','targets','gate','ready'):
 f=getattr(baseline,name);globals()[name]=types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__)
text=inspect.getsource(baseline.fit);changes={"gpu==resources.GPU[3]":"gpu==resources.GPU[4]","'phase_sets.14'":"'phase_sets.9'","'electronic.5.local.weight'":"'electronic.3.local.weight'","'fusion_logits.5'":"'fusion_logits.3'","phase_sets[14]":"phase_sets[9]","L6_last_global_only_class_separation":"L4_last_global_only_class_separation"}
for a,b in changes.items():assert a in text,a;text=text.replace(a,b)
exec(compile(text,__file__+':same_tail_fit_bound_to_L4_stage3_phase9','exec'),globals())
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--gpu',required=True);a=p.parse_args()
 try:fit(a.gpu)
 except BaseException as e:
  mark=OUT/'failure.json' if OUT.exists() else OUT.with_name(OUT.name+'.failure.json')
  if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
