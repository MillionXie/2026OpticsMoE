"""Actual anchor _apply migration proof plus original unchanged admission paths."""
import ast,inspect,os
import numpy as np
import torch
import caltech20_rich_alpha40_device_repair as m
import check_caltech20_rich_alpha40_rebalance as original
exec(compile(ast.parse(inspect.getsource(original.run)),__file__+':same_CPU_contract','exec'),globals())
if __name__=='__main__':
    try:
        assert os.environ['CUDA_VISIBLE_DEVICES']=='';evidence=m.failed_preflight()
        model=m.initialize(2,False);initial={n:v.clone() for n,v in model.phase_anchors.items()};assert initial and all(v.dtype==torch.float32 for v in initial.values())
        keys=set(model.state_dict());model.to(dtype=torch.float64)
        assert set(model.state_dict())==keys and all(v.dtype==dict(model.named_parameters())[n].dtype==torch.float64 for n,v in model.phase_anchors.items())
        assert all(torch.equal(v,initial[n].double()) for n,v in model.phase_anchors.items())
        # A deliberately stale anchor must be rejected before a noisy forward.
        name=next(iter(model.phase_anchors));model.phase_anchors[name]=model.phase_anchors[name].to('meta')
        try:model.context(1,1,m.config())
        except AssertionError:pass
        else:raise AssertionError('stale device anchor was accepted')
        del model
        run();p=m.s.read(m.CPU/'checks.json');p.update(original_failed_preflight=evidence,device_anchor_proof=dict(passed=True,actual_Module_apply_migrated_all_anchors=True,parameter_and_state_keys_unchanged=True,rejected_stale_meta_anchor=True,real_forward_calls=0));m.s.save(m.CPU/'checks.json',p)
    except BaseException as e:
        p=m.CPU.with_name(m.CPU.name+'.failure.json')
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
