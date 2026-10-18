"""Profile-only sparse input-amplitude TRAIN dropout, unchanged inference."""
import types,inspect,hashlib
import caltech20_rich_augmented_supervision46 as baseline
for n,v in vars(baseline).items():
    if not n.startswith('__') and n not in ('baseline','original','old'):globals()[n]=v
original=baseline.original
EXTRA=('caltech20_rich_input_dropout46.py','check_caltech20_rich_input_dropout46.py','audit_caltech20_rich_input_dropout46.py','CALTECH20_RICH_INPUT_DROPOUT46_PROTOCOL_20261018.md');FILES=(EXTRA[0],baseline.FILES[1],EXTRA[1],EXTRA[2],EXTRA[3])
def config():
    c=baseline.config();c.update(run='caltech20_rich20_augmentedCE80_inputDrop03_micro4_46_local5_20261018',input_dropout=.03,cases={key+'_inputDrop03':dict(row) for key,row in c['cases'].items()});return c
OUT=HERE/'runs/simulation'/config()['run'];CPU=HERE/'runs/smoke/caltech20_rich_input_dropout46_CPU_20261018'
def sources():
    z=baseline.sources()
    for n in EXTRA:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return z
for name in ('identity','claim','rates','seed_for','load_predictions','groups','js_loss','numeric_gate_kernel','campaign','feature_loss','target_data','initialize','gate_kernel','complete_gate_contract','fit_kernel','select_kernel','select','teacher_targets','gradient_record_kernel','fit','zoom_plan','plan_hash','transform','mixed_kernel','record_kernel','mixed','gate_original','gate','allocated_terms','gradient_kernel','original_gradients','gradients','ready_kernel','ready'):
    f=getattr(baseline,name);globals()[name]=types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__)
for name in ('incumbent','prototype_data','teacher_data'):
    f=getattr(baseline,name).__wrapped__;globals()[name]=lru_cache(None)(types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__))
old=types.SimpleNamespace(**vars(baseline.old));old.kd=types.SimpleNamespace(**vars(baseline.old.kd));old.kd.teacher_targets=teacher_targets
ns=dict(baseline.old.targets.__globals__);ns['kd']=old.kd;old.targets=types.FunctionType(baseline.old.targets.__code__,ns,'unchanged_clean_teacher_targets',baseline.old.targets.__defaults__,baseline.old.targets.__closure__)
LAST_INPUT_KEEP=None
def record_input_keep(keep):
    global LAST_INPUT_KEEP
    a=(keep.detach().cpu().numpy()>0).astype('uint8');LAST_INPUT_KEEP=dict(sha256=hashlib.sha256(a.tobytes()).hexdigest(),kept=int(a.sum()),n=int(a.shape[0]))
# Tap the actual existing input-mask tensor without changing its application.
core=types.SimpleNamespace(**vars(baseline.core));f=core.mixed;ns=dict(f.__globals__);ns['record_input_keep']=record_input_keep;text=inspect.getsource(f);site=";x*=F.interpolate(keep,(224,224),mode='nearest')[:,0]";assert text.count(site)==1;text=text.replace(site,";record_input_keep(keep)"+site);exec(compile(text,__file__+':same_input_mask_with_actual_saved_trace','exec'),ns);core.mixed=ns['mixed']
f=baseline.gradients;trace_gradients=types.FunctionType(f.__code__,globals(),'same_loss_add_actual_dropout_trace',f.__defaults__,f.__closure__)
def gradients(model,x,y,q,mass,mask,weights,cfg,micro):
    result=trace_gradients(model,x,y,q,mass,mask,weights,cfg,micro);assert LAST_INPUT_KEEP is not None and LAST_INPUT_KEEP['n']==len(x);model.view_info.update(actual_input_keep=LAST_INPUT_KEEP,actual_feature_keep_SHA256=hashlib.sha256((mask.detach().cpu().numpy()>0).astype('uint8').tobytes()).hexdigest());return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','campaign'],required=True);p.add_argument('--key');p.add_argument('--gpu');a=p.parse_args()
    if a.key:select(a.key)
    try:campaign() if a.phase=='campaign' else fit(a.key,a.gpu)
    except BaseException as e:
        f=OUT/a.key if a.key else OUT;mark=f/'failure.json' if f.exists() else f.with_name(f.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
