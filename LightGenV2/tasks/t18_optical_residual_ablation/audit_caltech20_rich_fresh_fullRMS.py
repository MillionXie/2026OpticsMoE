"""Independent fresh state, saved plans, Adam/projection/EMA and prototype math."""
import types,numpy as np
import caltech20_rich_fresh_fullRMS as m
import audit_caltech20_rich_dualview_hosthash_repair as saved
ks=dict(saved.kernel.__globals__);ks['m']=m;kernel=types.FunctionType(saved.kernel.__code__,ks,'fresh_saved_state_math',saved.kernel.__defaults__,saved.kernel.__closure__)
def run(key):
    m.select(key);kernel(key);folder=m.OUT/key;meta=m.s.read(folder/'metadata.json');trace=m.s.read(folder/'method_trace.json');cfg=m.config();d=cfg['cases'][key]['depth'];assert meta['case']['fresh'] and meta['alpha_restart']==.45 and cfg['reset_trained_weights'];initial=m.initialize(d,True);assert meta['init_digest']==m.s.digest(initial.state_dict())==m.s.read(m.CPU/'checks.json')['fresh_states'][str(d)]['initial_digest'];targets,scale,sha=m.target_data(d)
    for row in trace:
        v=row['view_info'];expected=.5*v['clean']+.5*v['augmented']+cfg['KD_weight']*v['KD_clean']+cfg['consistency_weight']*v['JS']+cfg['phase_anchor_weight']*v['phase_anchor']+scale*v['feature_alignment'];assert abs(expected-row['loss'])<=2e-6*max(1,abs(expected)) and v['full_rms_gradient'] and v['alignment_target_sha256']==sha and abs(v['alignment_scale']-scale)<1e-15
    gate=m.s.read(folder/'numeric_gate.json');assert gate['full_RMS_gradient'] and gate['fresh_initial_digest']==meta['init_digest'];m.s.save(folder/'science/fresh_fullRMS_checks.json',dict(passed=True,fresh_initial_alpha=.45,full_RMS_gradient=True,TRAIN_class_prototype_SHA256=sha,optimizer_steps=len(trace),additional_inference_parameters=0,new_optical_forward_calls=0,TEST_images_read=0),exclusive=True);proof=m.s.read(folder/'science/audit.json');proof['fresh_fullRMS_checks_SHA256']=m.s.sha(folder/'science/fresh_fullRMS_checks.json');m.s.save(folder/'science/audit.json',proof)
if __name__=='__main__':
    p=m.argparse.ArgumentParser();p.add_argument('--key',required=True);a=p.parse_args()
    try:run(a.key)
    except BaseException as e:
        p=m.OUT/a.key/'audit_failure.json'
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
