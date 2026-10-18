"""Saved optical science and independent online-teacher linear/MixUp replay."""
import types,numpy as np,torch,hashlib
import caltech20_rich_augmented_teacher46 as m
import audit_caltech20_rich_view_target_consistency46 as previous
chain=('kernel','inherited','base_run','prototype_run','phase_run','retained_run','original_run','inherited_run','run')
for name in chain:
    f=getattr(previous,name);ns=dict(f.__globals__);ns['m']=m
    for k in chain:
        if k in globals():ns[k]=globals()[k]
    globals()[name]=types.FunctionType(f.__code__,ns,name,f.__defaults__,f.__closure__)
prior=run
def run(key):
    prior(key);out=m.OUT/key;fit=m.s.read(out/'fit_result.json');cfg=m.config();d=cfg['cases'][key]['depth'];h=torch.load(m.teacher_source.OUT/'selected_head.pt',map_location='cpu',weights_only=False);assert fit['teacher_backbone_before']==fit['teacher_backbone_after'] and fit['teacher_optimizer_steps']==0 and fit['extra_student_inference_parameters']==0;assert fit['augmented_teacher_trace_SHA256']==m.s.sha(out/'augmented_teacher_trace.npz') and fit['augmented_teacher_rows_SHA256']==m.s.sha(out/'augmented_teacher_rows.json');rows=m.s.read(out/'augmented_teacher_rows.json');trace=m.s.read(out/'method_trace.json');labels=m.incumbent(d)[2]['train']['labels'];assert len(rows)==len(trace)==197*5
    with np.load(out/'augmented_teacher_trace.npz',allow_pickle=False) as z:
        assert z['features'].shape==(3141*5,512) and z['q'].shape==(3141*5,20)
        for row,t in zip(rows,trace):
            epoch,batch,n,off=[row[k] for k in ('epoch','batch','n','offset')];assert (epoch,batch,n)==(t['epoch'],t['batch'],t['n']);order,_=m.engine.epoch_plan(epoch,3141,cfg);ix=order[batch*16:batch*16+n];y=labels[ix];assert np.array_equal(z['labels'][off:off+n],y);features=z['features'][off:off+n];x=((features-h['mean'])/h['std']).astype(np.float32).astype(float);logits=x@h['model']['weight'].double().numpy().T+h['model']['bias'].double().numpy();raw=np.exp(logits-logits.max(1,keepdims=True));raw/=raw.sum(1,keepdims=True);valid=((logits.argmax(1)==y)&(raw.max(1)>=.5)).astype(np.float32);temp=cfg['teacher_temperature_by_depth'][str(d)];p=np.exp(logits/temp-(logits/temp).max(1,keepdims=True));p=(p/p.sum(1,keepdims=True)).astype(np.float32);rng=np.random.default_rng(np.random.SeedSequence([17,epoch,batch,20261013,4]));lam=float(rng.beta(cfg['mixup_alpha'],cfg['mixup_alpha']));lam=max(lam,1-lam);perm=rng.permutation(n);mass=lam*valid+(1-lam)*valid[perm];q=(lam*valid[:,None]*p+(1-lam)*valid[perm,None]*p[perm])/mass[:,None].clip(1e-12);qa=z['q'][off:off+n];ma=z['mass'][off:off+n];assert np.max(abs(q-qa))<=3e-6 and np.max(abs(mass-ma))<=1e-7 and row['correct_fraction']==float((logits.argmax(1)==y).mean()) and row['gate_fraction']==float(valid.mean());v=t['view_info'];assert v['augmented_teacher_q_SHA256']==hashlib.sha256(qa.astype('<f4').tobytes()).hexdigest() and v['augmented_teacher_mass_SHA256']==hashlib.sha256(ma.astype('<f4').tobytes()).hexdigest() and v['augmented_teacher_features_SHA256']==hashlib.sha256(features.astype('<f4').tobytes()).hexdigest() and v['teacher_reference_augmented_RGB100']
    gate=m.s.read(out/'numeric_gate.json');assert fit['teacher_backbone_before']==m.s.read(m.teacher_source.OUT/'fit_result.json')['backbone_before'];assert fit['teacher_forwarded_TRAIN_images_including_gate']==fit['teacher_forwarded_TRAIN_views']+gate['teacher_gate_forwarded_images']
    viewpath=out/'science/view_and_derivative_checks.json';view=m.s.read(viewpath);view.update(teacher_reference_input_clean=False,clean_teacher_cached_augmented_teacher_online=True);m.s.save(viewpath,view)
    path=out/'science/audit.json';q=m.s.read(path);q.update(independent_augmented_teacher_head_and_MixUp=True,teacher_backbone_frozen_SHA256=fit['teacher_backbone_after'],teacher_optimizer_steps=0,teacher_deployed=False,new_student_parameters=0,view_checks_sha256=m.s.sha(viewpath));m.s.save(path,q)
if __name__=='__main__':
    p=m.argparse.ArgumentParser();p.add_argument('--key',required=True);a=p.parse_args()
    try:run(a.key)
    except BaseException as e:
        p=m.OUT/a.key/'audit_failure.json'
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
