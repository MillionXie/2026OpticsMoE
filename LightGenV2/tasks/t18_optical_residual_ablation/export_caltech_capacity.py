"""Parameterization-aware offline export, using physical expanded planes."""
import argparse, os
from pathlib import Path
import torch
import caltech_capacity as c
import export_caltech_repair234 as repaired_export
from caltech_capacity_model import CapacityMoE, deployment_contract

def export_selected(model,config,out,checkpoint,checkpoint_sha256,classes,sources):
    assert isinstance(model,CapacityMoE) and config['phase_grid']==model.phase_grid
    value=repaired_export.export_selected(model,config,out,checkpoint,checkpoint_sha256,classes,sources)
    value.update(contract=deployment_contract(model.depth,model.rho,model.phase_grid),
                 parameterization_source_sha256=c.sha(Path(__file__).with_name('caltech_capacity_model.py')),
                 capacity_export_source_sha256=c.sha(__file__),
                 free_parameter_shapes={n:list(p.shape) for n,p in model.named_parameters()},
                 physical_main_shapes=[list(s) for s in model.physical_shapes],
                 amplitude_export_api='export_caltech_repair234.save_sample_bmps on CapacityMoE; expanded plane then joint absM/angleM/8bit')
    c.save(Path(out)/'manifest.json',value)
    return value

def selected(base,out):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not out.exists()
    from audit_caltech_capacity import exited
    proof=c.read(base/'cpu_report/audit.json');meta=c.read(base/'metadata.json');state=c.read(base/'status.json')
    assert proof['passed'] and proof['chosen_profile_case'] in proof['eligible'] and exited(meta['pid'])
    assert all(exited(j['pid']) for j in state['completed']) and not state['failures']
    key=proof['chosen_profile_case'];f=proof['fits'][key];ck=base/key/'best_checkpoint.pt'
    c.engine.verify_sources(f['sources'],meta['git_commit']);assert c.sha(ck)==f['checkpoint_sha256']
    state=torch.load(ck,map_location='cpu',weights_only=False);assert state['config']==f['config'] and state['epoch']==f['selected_epoch']
    model=CapacityMoE(4,.3,f['config']['phase_grid']).eval();model.load_state_dict(state['model'],strict=True)
    return export_selected(model,f['config'],out,ck,f['checkpoint_sha256'],f['config']['classes'],f['sources'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:selected(a.base,a.out)
    except BaseException as e:
        if a.out.exists():c.save(a.out/'failure.json',dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
