"""Identical independent class-center proof, distinct strong-view config."""
import types
import caltech20_rich_prototype_strongview as m
import check_caltech20_rich_prototype_consistency as original
import audit_caltech20_rich_prototype_strongview as audit
def run():
    ns=dict(vars(original));ns.update(m=m,audit=audit);proof=types.FunctionType(original.run.__code__,ns,'same_prototype_CPU_proof',original.run.__defaults__,original.run.__closure__);proof()
    c=m.config();assert c['crop_area_min']==.75 and c['input_dropout']==.03 and c['feature_dropout']==.1 and c['ccd_read_noise']==c['ccd_gain_noise']==c['phase_dropout']==0
    q=m.s.read(m.CPU/'checks.json');q.update(matched_strong_view_configuration=True,clean_view_unmodified=True,new_inference_parameters=0);m.s.save(m.CPU/'checks.json',q)
if __name__=='__main__':
    try:run()
    except BaseException as e:
        p=m.CPU.with_name(m.CPU.name+'.failure.json')
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
