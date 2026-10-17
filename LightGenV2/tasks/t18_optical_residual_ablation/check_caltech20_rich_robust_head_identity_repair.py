"""Regression for actual SHA-string dictionaries; unchanged objective math."""
import ast,inspect,os
import numpy as np
import torch
import caltech20_rich_robust_head_identity_repair as m
import check_caltech20_rich_robust_head as old
import audit_caltech20_rich_robust_head_identity_repair as audit
for function in (old.numpy_objective,old.run):exec(compile(ast.parse(inspect.getsource(function)),__file__+':unchanged_CPU_math','exec'),globals())
if __name__=='__main__':
    try:
        origin=m.reuse_preflight();actual=m.sources();assert audit.source_identity_matches(actual,dict(actual));changed=dict(actual);key=next(iter(changed));changed[key]='0'*64;assert not audit.source_identity_matches(actual,changed)
        assert not audit.source_identity_matches({'x':torch.ones(1)},actual) and not audit.source_identity_matches('wrongtype',actual)
        run();m.s.save(m.CPU/'source_identity_checks.json',dict(passed=True,real_source_string_dictionary=True,rejected_changed_SHA=True,rejected_tensor_and_string=True,original_failure=origin),exclusive=True)
    except BaseException as e:
        p=m.CPU.with_name(m.CPU.name+'.failure.json')
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
