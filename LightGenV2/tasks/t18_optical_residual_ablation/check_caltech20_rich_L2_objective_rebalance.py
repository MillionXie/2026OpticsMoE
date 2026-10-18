"""Same independent phase/state proof plus independent prototype coefficient."""
import types,numpy as np
import caltech20_rich_L2_objective_rebalance as m
import check_caltech20_rich_L2_retained_soup_finetune as previous
import audit_caltech20_rich_L2_objective_rebalance as audit
ns=dict(previous.same_phase_run.__globals__);ns.update(m=m,audit=audit);same_phase_run=types.FunctionType(previous.same_phase_run.__code__,ns,'same_independent_L2_state_phase_proof',previous.same_phase_run.__defaults__,previous.same_phase_run.__closure__)

def run():
    same_phase_run();q,pt,pred=m.incumbent(2);features=pred['train']['features'].astype(float);p=pred['train']['probabilities'];y=pred['train']['labels'];w=pt['model']['linear.weight'].numpy().astype(float);target,scale,sha,info=m.prototype_data(2);soft=np.eye(20)[y]*(1-m.config()['label_smoothing'])+m.config()['label_smoothing']/20;expected=.25*min(.1,.2*np.linalg.norm((p-soft)@w)/max(np.linalg.norm(features-target),1e-12));assert abs(expected-scale)<1e-15 and q['sha256']=='e7c140ecd11d52a91bef882a01bc947c3da7f9f72df63d7fcf23055ab503be69';proof=m.s.read(m.CPU/'checks.json');proof.update(quarter_scale_independently_verified=True,loaded_actual_8005_model_not_fresh=True,local_counter_not_historical_epoch=True,gate_producer_auditor_capture_key_contract_proved=True,numerical_thresholds_unchanged=True,expected_prototype_scale=float(expected));m.s.save(m.CPU/'checks.json',proof)
if __name__=='__main__':
    try:run()
    except BaseException as e:
        p=m.CPU.with_name(m.CPU.name+'.failure.json')
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
