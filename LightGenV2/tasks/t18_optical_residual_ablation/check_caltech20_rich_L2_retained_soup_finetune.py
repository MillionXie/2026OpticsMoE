"""Actual L2 retained state, independent phase math and local-epoch contract."""
import inspect
import caltech20_rich_L2_retained_soup_finetune as m
import check_caltech20_rich_phasecode_consistency as previous
import audit_caltech20_rich_L2_retained_soup_finetune as audit
text=inspect.getsource(previous.run);assert text.count('for d in (4,6):')==1;text=text.replace('for d in (4,6):','for d in (2,):').replace('{4:21,6:30}','{2:12,4:21,6:30}');ns=dict(previous.run.__globals__);ns.update(m=m,audit=audit);exec(compile(text,__file__+':same_independent_phase_proof_for_L2','exec'),ns);same_phase_run=ns['run']
def run():
    same_phase_run();q,pt,pred=m.incumbent(2);assert q['sha256']=='e7c140ecd11d52a91bef882a01bc947c3da7f9f72df63d7fcf23055ab503be69' and q['joint_budget_epochs'] is None and q['selected_optical_epoch'] is None and q['parameter_average'];model=m.initialize(2,False);assert m.s.digest(model.state_dict())==m.s.digest(pt['model']) and m.engine.metrics(pred['test']['labels'],pred['test']['probabilities'])['accuracy']==329/411;cfg=m.config();assert cfg['local_adaptation_epoch_labels'] and not cfg['cases']['L2_soup_finetune']['fresh'];example=m.complete_gate_contract(dict(parent_capture_maxabs=1e-7,parent_clean_probability_maxabs=2e-7,full_rms_gradient=True));assert example['parent_clean_capture_maxabs']==example['parent_capture_maxabs'];p=m.s.read(m.CPU/'checks.json');p.update(loaded_actual_8005_model_not_fresh=True,local_counter_not_historical_epoch=True,gate_producer_auditor_capture_key_contract_proved=True,numerical_thresholds_unchanged=True);m.s.save(m.CPU/'checks.json',p)
if __name__=='__main__':
    try:run()
    except BaseException as e:
        p=m.CPU.with_name(m.CPU.name+'.failure.json')
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
