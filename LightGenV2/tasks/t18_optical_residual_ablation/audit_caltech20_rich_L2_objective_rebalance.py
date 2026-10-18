"""Unchanged local-epoch Adam/EMA science and quarter-strength target proof."""
import types,numpy as np
import caltech20_rich_L2_objective_rebalance as m
import audit_caltech20_rich_L2_retained_soup_finetune as previous
ks=dict(previous.kernel.__globals__);ks['m']=m;kernel=types.FunctionType(previous.kernel.__code__,ks,'same_independent_local_epoch_state_math',previous.kernel.__defaults__,previous.kernel.__closure__)
ns=dict(previous.inherited.__globals__);ns.update(m=m,kernel=kernel);inherited=types.FunctionType(previous.inherited.__code__,ns,'same_loss_math',previous.inherited.__defaults__,previous.inherited.__closure__)
ns=dict(previous.base_run.__globals__);ns.update(m=m,inherited=inherited);base_run=types.FunctionType(previous.base_run.__code__,ns,'same_loss_accounting',previous.base_run.__defaults__,previous.base_run.__closure__)
ns=dict(previous.prototype_run.__globals__);ns.update(m=m,base_run=base_run);prototype_run=types.FunctionType(previous.prototype_run.__code__,ns,'same_independent_centers',previous.prototype_run.__defaults__,previous.prototype_run.__closure__)
ns=dict(previous.phase_run.__globals__);ns.update(m=m,prototype_run=prototype_run);phase_run=types.FunctionType(previous.phase_run.__code__,ns,'same_phase_noise_math',previous.phase_run.__defaults__,previous.phase_run.__closure__)
ns=dict(previous.retained_run.__globals__);ns.update(m=m,phase_run=phase_run);retained_run=types.FunctionType(previous.retained_run.__code__,ns,'same_gate_and_state_science',previous.retained_run.__defaults__,previous.retained_run.__closure__)

def run(key):
    retained_run(key);out=m.OUT/key;fit=m.s.read(out/'fit_result.json');assert fit['local_added_epochs']==3 and fit['previous_budget_epochs'] is None and fit['not_fresh_initialization'];q,pt,pred=m.incumbent(2);features=pred['train']['features'].astype(float);p=pred['train']['probabilities'];y=pred['train']['labels'];w=pt['model']['linear.weight'].numpy().astype(float);target,scale,_,_=m.prototype_data(2);soft=np.eye(20)[y]*(1-m.config()['label_smoothing'])+m.config()['label_smoothing']/20;expected=.25*min(.1,.2*np.linalg.norm((p-soft)@w)/max(np.linalg.norm(features-target),1e-12));assert abs(expected-scale)<1e-15;proof=m.s.read(out/'science/audit.json');proof.update(quarter_prototype_scale_independently_verified=True,local_added_epochs=3,not_from_scratch=True);m.s.save(out/'science/audit.json',proof)
if __name__=='__main__':
    p=m.argparse.ArgumentParser();p.add_argument('--key',required=True);a=p.parse_args()
    try:run(a.key)
    except BaseException as e:
        p=m.OUT/a.key/'audit_failure.json'
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
