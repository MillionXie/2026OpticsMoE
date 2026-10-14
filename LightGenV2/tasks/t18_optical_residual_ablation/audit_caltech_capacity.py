"""Saved CPU proof for phase grids; no forward or test, exclusive failure retained."""
import argparse, copy, os
from pathlib import Path
import torch
import caltech_capacity as c
import audit_caltech_repair234 as original
from caltech_capacity_model import CapacityMoE, GRIDS, COUNTS
exited=original.exited

def audit(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not a.out.exists() and not (a.base/'cpu_report_failure.json').exists()
    meta=c.read(a.base/'metadata.json');state=c.read(a.base/'status.json')
    assert exited(meta['pid']) and not state['failures'] and len(state['completed'])==3 and state['active']==state['pending']==0
    assert sorted(j['grid'] for j in state['completed'])==sorted(GRIDS)
    c.prerequisites();c.engine.verify_sources(meta['sources'],meta['git_commit']);m=c.repaired.manifest(a.data);fits={};gates={}
    for job in state['completed']:
        assert exited(job['pid']) and job['exit_code']==0
        grid=job['grid'];c.bind(grid)
        # Reuse independent saved epoch/hash/Adam/prediction verification, but
        # construct only the correct free-parameter shape for this arm.
        original.CorrectedMoE=lambda depth,rho:CapacityMoE(depth,rho,grid)
        f=original.verify_arm(a.base,job['case'],a.data,m);assert f['sources']==meta['sources']
        assert f['config']['phase_grid']==grid and f['config']['mixup_alpha']==.6 and f['depth']==4 and f['rho']==.3
        hist=c.read(a.base/job['case']/'history.json');g=c.eligibility(f,hist)
        assert g==c.read(a.base/job['case']/'validation_gate.json') and not (a.base/job['case']/'failure.json').exists()
        cfg=f['config'];shapes=torch.load(a.base/job['case']/'best_checkpoint.pt',map_location='cpu',weights_only=False)['model']
        assert sum(v.numel() for v in shapes.values())==COUNTS[grid]
        assert tuple(shapes['raw_router'].shape)==(224,224)
        assert len([n for n in shapes if n.startswith('phase_sets.')])==10
        fits[job['case']]=f;gates[job['case']]=g
    normalized=[]
    for f in fits.values():
        v=copy.deepcopy(f['config'])
        for name in ('phase_grid','study_case','contract'):v.pop(name)
        normalized.append(v)
    assert all(v==normalized[0] for v in normalized)
    minimum=min(f['epochs_completed'] for f in fits.values())
    assert len({tuple(f['orders'][:minimum]) for f in fits.values()})==1
    assert len({tuple(f['transforms'][:minimum]) for f in fits.values()})==1
    eligible=[k for k in sorted(fits) if gates[k]['passed']]
    chosen=min(eligible,key=lambda k:fits[k]['metrics']['val']['balanced_nll']) if eligible else None
    a.out.mkdir(parents=True,exist_ok=False)
    c.save(a.out/'audit.json',dict(passed=True,fits=fits,gates=gates,eligible=eligible,chosen_profile_case=chosen,
           model_forward_calls=0,test_arrays_read=0,test_evaluations=0,optimizer_steps_per_epoch=78,tail_batch=14,
           initial_equality_proof=c.read(c.ROOT/c.read(c.PLAN)['gates']['cpu_native'])['initial_equality'],
           formal_authorized_by_this_receipt=False,architecture_adoption_authorized=False,
           next_step='Discuss generalization/data and selected trained native proof; no automatic formal launch' if chosen else 'Bounded failure; preserve all outputs, no automatic additional capacity fits'),exclusive=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('base','out','data'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    try:audit(a)
    except BaseException as e:
        marker=a.base/'cpu_report_failure.json'
        if not marker.exists():c.save(marker,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
