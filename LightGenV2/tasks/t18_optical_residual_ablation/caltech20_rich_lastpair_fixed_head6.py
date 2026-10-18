"""L6 feature adaptation: freeze readout, train only the final EG pair."""
import types,inspect
import caltech20_rich_tail_separation6 as baseline
for n,v in vars(baseline).items():
 if not n.startswith('__') and n not in ('baseline','closure'):globals()[n]=v
FILES=('caltech20_rich_lastpair_fixed_head6.py','caltech20_rich_lastpair_fixed_head6_profile.json','check_caltech20_rich_lastpair_fixed_head6.py','audit_caltech20_rich_lastpair_fixed_head6.py','CALTECH20_RICH_LASTPAIR_FIXED_HEAD6_PROTOCOL_20261018.md')
TRAINABLE=tuple('phase_sets.'+str(i) for i in range(10,15))+('electronic.4.local.weight','electronic.5.local.weight','fusion_logits.4','fusion_logits.5')
def config():return s.read(HERE/FILES[1])
OUT=HERE/'runs/simulation'/config()['run'];CPU=HERE/'runs/smoke/caltech20_rich_lastpair_fixed_head6_CPU_20261018'
def sources():
 z=baseline.sources()
 for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
 return z
for name in ('identity','claim','frozen_digest','parent_data','model_for','plan','inputs','gradients','targets','gate','ready'):
 f=getattr(baseline,name);globals()[name]=types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__)
text=inspect.getsource(baseline.triplet_scale);assert 'beta=min(2.,.2*' in text;text=text.replace('beta=min(2.,.2*',"beta=min(config()['triplet_max_beta'],.2*");exec(compile(text,__file__+':TRAIN_calibrated_20percent_not_cap2','exec'),globals())
text=inspect.getsource(baseline.fit);site="groups=[['phase_sets.14'],['electronic.5.local.weight','linear.weight','linear.bias'],['fusion_logits.5']]";assert text.count(site)==1;text=text.replace(site,"groups=[['phase_sets.'+str(i) for i in range(10,15)],['electronic.4.local.weight','electronic.5.local.weight'],['fusion_logits.4','fusion_logits.5']]");text=text.replace('L6_last_global_only_class_separation','L6_last_EG_pair_fixed_head_class_separation');exec(compile(text,__file__+':same_admitted_steps_with_fixed_head_final_EG_scope','exec'),globals())
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--gpu',required=True);a=p.parse_args()
 try:fit(a.gpu)
 except BaseException as e:
  mark=OUT/'failure.json' if OUT.exists() else OUT.with_name(OUT.name+'.failure.json')
  if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
