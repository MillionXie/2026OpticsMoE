"""Same independent view-weight/teacher/zoom/throughput CPU admission."""
import types
import caltech20_rich_feature_dropout46 as m
import check_caltech20_rich_augmented_supervision46 as previous
import audit_caltech20_rich_feature_dropout46 as audit
for name in ('independent_teacher','original_run','inherited','numpy_transform','prior','run'):
    f=getattr(previous,name);ns=dict(f.__globals__);ns.update(m=m,audit=audit)
    for k in ('independent_teacher','original_run','inherited','numpy_transform','prior'):
        if k in globals():ns[k]=globals()[k]
    globals()[name]=types.FunctionType(f.__code__,ns,name,f.__defaults__,f.__closure__)
inherited_admission=run
def run():
    assert m.config()['feature_dropout']==.2 and m.config()['microbatch_size']==4;inherited_admission();p=m.CPU/'checks.json';q=m.s.read(p);q.update(feature_dropout_TRAIN_augmented_only=.2,no_dropout_in_clean_eval=True,only_profile_difference_from_source501='feature_dropout .05 to .2');m.s.save(p,q)
if __name__=='__main__':
    try:run()
    except BaseException as e:
        p=m.CPU.with_name(m.CPU.name+'.failure.json')
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
