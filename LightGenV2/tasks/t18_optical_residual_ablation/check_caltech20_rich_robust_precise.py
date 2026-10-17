"""Same actual-path CPU admission, with exact float32-effective alpha proof."""
import ast,inspect,os
import numpy as np
import torch
import caltech20_rich_robust_precise as m
import check_caltech20_rich_robust as old
exec(compile(ast.parse(inspect.getsource(old.run)),__file__+':same_CPU_paths','exec'),globals())
if __name__=='__main__':
    try:
        m.resource_overlap()
        for depth in (2,4,6):
            _,pt=m.parent(depth);new=m.remap(pt['model'])
            for n,v in pt['model'].items():
                if n.startswith('fusion_logits.'):
                    before=.7001+(.95-.7001)*v.sigmoid();after=(.6001+(.95-.6001)*new[n].sigmoid()).float()
                    assert torch.equal(before,after),(depth,n,float(before-after))
        run()
    except BaseException as e:
        p=m.CPU.with_name(m.CPU.name+'.failure.json')
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
