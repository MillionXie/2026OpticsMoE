"""Same strict prototype proof against actual accepted E100 state."""
import types
import caltech20_rich_L4_prototype_continue as m
import check_caltech20_rich_prototype_consistency as previous
import audit_caltech20_rich_L4_prototype_continue as audit
def run():
    m.completed_L4();ns=dict(previous.run.__globals__);ns.update(m=m,audit=audit);proof=types.FunctionType(previous.run.__code__,ns,'same_prototype_CPU_actual_E100',previous.run.__defaults__,previous.run.__closure__);proof();q=m.s.read(m.CPU/'checks.json');q.update(parent_selected_epoch=m.incumbent(4)[0]['selected_optical_epoch'],parent_actual_budget=m.incumbent(4)[0]['joint_budget_epochs'],optimizer_reset_explicit=True,not_full_LAST_resume=True);m.s.save(m.CPU/'checks.json',q)
if __name__=='__main__':
    try:run()
    except BaseException as e:
        p=m.CPU.with_name(m.CPU.name+'.failure.json')
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
