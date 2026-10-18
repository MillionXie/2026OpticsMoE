"""Same independent view-weight/teacher/zoom/throughput CPU admission."""
import types
import caltech20_rich_input_dropout46 as m
import check_caltech20_rich_augmented_supervision46 as previous
import audit_caltech20_rich_input_dropout46 as audit
for name in ('independent_teacher','original_run','inherited','numpy_transform','prior','run'):
    f=getattr(previous,name);ns=dict(f.__globals__);ns.update(m=m,audit=audit)
    for k in ('independent_teacher','original_run','inherited','numpy_transform','prior'):
        if k in globals():ns[k]=globals()[k]
    globals()[name]=types.FunctionType(f.__code__,ns,name,f.__defaults__,f.__closure__)
inherited_admission=run
def run():
    assert m.config()['input_dropout']==.03 and m.config()['microbatch_size']==4;inherited_admission();cfg=m.config();n=2;images=m.torch.arange(n*3*150*150).remainder(251).byte().reshape(n,3,150,150);labels=m.torch.tensor([0,1]);theta=m.np.tile(m.np.eye(2,3,dtype=m.np.float32),(n,1,1));_,_,mask,_=m.core.mixed((images,labels),m.np.arange(n),theta,1,0,cfg,'cpu');rg=m.np.random.default_rng(m.np.random.SeedSequence([17,1,0,20261018,40,1]));rg.uniform(cfg['crop_area_min'],1,n);rg.uniform(.9,1.1,n);rg.uniform(-1,1,n);rg.uniform(-1,1,n);rg.random(n);rg.uniform(.8,1.2,(n,3,1,1));rg.uniform(.8,1.2,(n,1,1,1));rg.random(n);keep=(rg.random((n,1,16,16))>=.03).astype('uint8');feature=rg.random((n,256))>=.05;assert m.LAST_INPUT_KEEP['sha256']==m.hashlib.sha256(keep.tobytes()).hexdigest() and m.np.array_equal(mask.numpy()>0,feature)
    p=m.CPU/'checks.json';q=m.s.read(p);q.update(input_dropout_TRAIN_augmented_only=.03,no_dropout_in_clean_eval=True,actual_mask_trace_synthetic_RGB_proof=True,synthetic_RGB_inputs=2,only_profile_difference_from_source501='input_dropout .0 to .03');m.s.save(p,q)
if __name__=='__main__':
    try:run()
    except BaseException as e:
        p=m.CPU.with_name(m.CPU.name+'.failure.json')
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
