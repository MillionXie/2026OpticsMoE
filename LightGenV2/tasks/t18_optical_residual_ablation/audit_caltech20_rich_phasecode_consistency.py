"""Full prior independent state math plus deterministic phase-code accounting."""
import types,numpy as np,math
import caltech20_rich_phasecode_consistency as m
import audit_caltech20_rich_prototype_consistency as previous
ks=dict(previous.kernel.__globals__);ks['m']=m;kernel=types.FunctionType(previous.kernel.__code__,ks,'same_saved_state_math',previous.kernel.__defaults__,previous.kernel.__closure__)
ns=dict(previous.inherited.__globals__);ns.update(m=m,kernel=kernel);inherited=types.FunctionType(previous.inherited.__code__,ns,'same_semantic_loss_math',previous.inherited.__defaults__,previous.inherited.__closure__)
ns=dict(previous.base_run.__globals__);ns.update(m=m,inherited=inherited);base_run=types.FunctionType(previous.base_run.__code__,ns,'same_loss_and_target_accounting',previous.base_run.__defaults__,previous.base_run.__closure__)
ps=dict(previous.run.__globals__);ps.update(m=m,base_run=base_run);prototype_run=types.FunctionType(previous.run.__code__,ps,'independent_TRAIN_prototype_reconstruction',previous.run.__defaults__,previous.run.__closure__)
def run(key):
    m.select(key);prototype_run(key);out=m.OUT/key;cfg=m.config();d=cfg['cases'][key]['depth'];trace=m.s.read(out/'method_trace.json');checks=[]
    for row in trace:
        noise=np.clip(np.random.default_rng(np.random.SeedSequence([int(row['seed']),20261018,421])).standard_normal((d+1,1,16,16)).astype(np.float32),-2,2)*cfg['phase_jitter_sigma_codes'];v=row['view_info'];expected=math.ceil(row['n']/cfg['microbatch_size'])*(d+1);assert v['phase_jitter_SHA256']==m.s.digest(noise) and v['phase_jitter_sigma_codes']==1 and v['phase_jitter_augmented_only'] and v['jittered_plane_calls']==v['clean_plane_calls']==expected
    gate=m.s.read(out/'numeric_gate.json');assert gate['phase_jitter_augmented_only'] and gate['parent_clean_probability_maxabs']<=3e-6 and gate['parent_capture_maxabs']<=3e-6;m.s.save(out/'science/phasecode_checks.json',dict(passed=True,macros=len(trace),independent_numpy_phase_noise=True,real_recorded_clean_and_augmented_plane_counts=True,clean_inference_unchanged=True,model_forward_calls=0,TEST_images_read=0),exclusive=True);proof=m.s.read(out/'science/audit.json');proof['phasecode_checks_SHA256']=m.s.sha(out/'science/phasecode_checks.json');m.s.save(out/'science/audit.json',proof)
if __name__=='__main__':
    p=m.argparse.ArgumentParser();p.add_argument('--key',required=True);a=p.parse_args()
    try:run(a.key)
    except BaseException as e:
        p=m.OUT/a.key/'audit_failure.json'
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
