"""Retain numerical checks and prove producer/consumer key compatibility."""
import types
import caltech20_rich_phasecode_gate_contract as m
import check_caltech20_rich_phasecode_consistency as previous
import audit_caltech20_rich_phasecode_gate_contract as audit
ns=dict(previous.run.__globals__);ns.update(m=m,audit=audit);original_run=types.FunctionType(previous.run.__code__,ns,'same_independent_phase_math',previous.run.__defaults__,previous.run.__closure__)
def run():
    assert not m.baseline.CPU.exists() and not m.baseline.OUT.exists() and not m.baseline.CPU.with_name(m.baseline.CPU.name+'.claim.json').exists();original_run();example=dict(parent_capture_maxabs=1e-7,parent_clean_probability_maxabs=2e-7,full_rms_gradient=True);result=m.complete_gate_contract(example);assert result['parent_clean_capture_maxabs']==result['parent_capture_maxabs']==example['parent_capture_maxabs'] and result['parent_clean_probability_maxabs']==example['parent_clean_probability_maxabs'];proof=m.s.read(m.CPU/'checks.json');proof.update(gate_producer_auditor_capture_key_contract_proved=True,source421_real_forward_and_optimizer_calls=0,numerical_thresholds_unchanged=True);m.s.save(m.CPU/'checks.json',proof)
if __name__=='__main__':
    try:run()
    except BaseException as e:
        p=m.CPU.with_name(m.CPU.name+'.failure.json')
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
