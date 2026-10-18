"""More augmented supervised CE, same total weight and inference; admitted micro4."""
import types,inspect
import caltech20_rich_view_target_consistency46 as baseline
import caltech20_rich_microbatch_throughput46 as source_closure
for n,v in vars(baseline).items():
    if not n.startswith('__') and n not in ('baseline','source_closure','original','old'):globals()[n]=v
original=baseline.original
EXTRA=('caltech20_rich_augmented_supervision46.py','check_caltech20_rich_augmented_supervision46.py','audit_caltech20_rich_augmented_supervision46.py','CALTECH20_RICH_AUGMENTED_SUPERVISION46_PROTOCOL_20261018.md')
FILES=(EXTRA[0],baseline.FILES[1],EXTRA[1],EXTRA[2],EXTRA[3])
def config():
    c=baseline.config();case=dict(fresh=False,epochs=5,electronic_warm_epochs=0,phase_lr=.00015,full_rms_gradient=True,consistency_target='clean_targets')
    c.update(run='caltech20_rich20_augmentedCE80_micro4_46_local5_20261018',cases={f'L{d}_augmentedCE80':dict(case,depth=d) for d in (4,6)},microbatch_size=4,clean_CE_coefficient=.2,augmented_CE_coefficient=.8,teacher_and_prototype_student_view='clean_only',gpu_order=[3,4,0,1,2])
    return c
OUT=HERE/'runs/simulation'/config()['run'];CPU=HERE/'runs/smoke/caltech20_rich_augmented_supervision46_CPU_20261018'
def sources():
    z=source_closure.sources()
    for n in EXTRA:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return z
for name in ('identity','claim','rates','seed_for','load_predictions','groups','js_loss','numeric_gate_kernel','campaign','feature_loss','target_data','initialize','gate_kernel','complete_gate_contract','fit_kernel','select_kernel','select','teacher_targets','gradient_record_kernel','fit','zoom_plan','plan_hash','transform','mixed_kernel','record_kernel','mixed','gate_original','gate'):
    f=getattr(baseline,name);globals()[name]=types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__)
for name in ('incumbent','prototype_data','teacher_data'):
    f=getattr(baseline,name).__wrapped__;globals()[name]=lru_cache(None)(types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__))
old=types.SimpleNamespace(**vars(baseline.old));old.kd=types.SimpleNamespace(**vars(baseline.old.kd));old.kd.teacher_targets=teacher_targets
ns=dict(baseline.old.targets.__globals__);ns['kd']=old.kd;old.targets=types.FunctionType(baseline.old.targets.__code__,ns,'same_clean_teacher_targets',baseline.old.targets.__defaults__,baseline.old.targets.__closure__)
def allocated_terms(kc,ka,pc,pa,mode):
    assert mode=='clean_targets';return kc,pc
text=inspect.getsource(baseline.gradient_kernel);before='.5*clean+.5*aug';assert text.count(before)==1;text=text.replace(before,"cfg['clean_CE_coefficient']*clean+cfg['augmented_CE_coefficient']*aug");exec(compile(text,__file__+':reviewed_supervised_view_reweight','exec'),globals())
f=baseline.gradients;original_gradients=types.FunctionType(f.__code__,globals(),'same_exact_zoomout_and_teacher_trace',f.__defaults__,f.__closure__)
def gradients(model,x,y,q,mass,mask,weights,cfg,micro):
    result=original_gradients(model,x,y,q,mass,mask,weights,cfg,micro);model.view_info.update(clean_CE_coefficient=cfg['clean_CE_coefficient'],augmented_CE_coefficient=cfg['augmented_CE_coefficient'],teacher_and_prototype_student_view='clean_only');return result
f=baseline.ready_kernel;ready_kernel=types.FunctionType(f.__code__,globals(),'unchanged_retained_source_and_teacher_admission',f.__defaults__,f.__closure__)
def ready():
    q=s.read(CPU/'checks.json');assert q['independent_weighted_view_gradient'] and q['independent_numpy_zoomout_pixels'] and q['micro4_TRAIN_throughput_admission_SHA256']=='f2d4149faea8aae24e947232942344ca747cece0aef4b64d38ca5ca0502ae535' and q['direct_TEST_only_selection'] and not CPU.with_name(CPU.name+'.failure.json').exists();ready_kernel()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','campaign'],required=True);p.add_argument('--key');p.add_argument('--gpu');a=p.parse_args()
    if a.key:select(a.key)
    try:campaign() if a.phase=='campaign' else fit(a.key,a.gpu)
    except BaseException as e:
        f=OUT/a.key if a.key else OUT;mark=f/'failure.json' if f.exists() else f.with_name(f.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
