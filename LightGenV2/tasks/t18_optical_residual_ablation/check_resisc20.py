"""CPU full-resolution graph/slot admission; reuse existing native proof unchanged."""
import argparse,os,subprocess
from pathlib import Path
import torch
import resisc20 as s

def check(out):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not out.exists();torch.set_num_threads(4)
    src=s.sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=s.ROOT,text=True).strip();s.engine.verify_sources(src,commit)
    states=[]
    for rho in (0.,.3):
        torch.manual_seed(17);model=s.r.CorrectedMoE(4,rho);states.append(s.digest(model.state_dict()))
        assert sum(p.numel() for p in model.parameters())==908552 and len(list(model.parameters()))==11
        assert not any(isinstance(v,(torch.nn.Linear,torch.nn.Conv2d)) for v in model.modules())
        assert model.stages==['expert','global','expert','global'] and s.r.deployment_contract(4,rho)['total_captures']==5
    assert states[0]==states[1]
    old=s.HERE/'runs/smoke/caltech_repair234_matched_export_20261014/checks.json';proof=s.read(old)
    assert proof['passed'] and len(proof['checks'])==18
    assert proof['training_reference_sources']==s.r.sources() and proof['test_evaluations']==0 and proof['devices_opened']==0
    assert all(s.sha(s.HERE/name)==digest for name,digest in proof['source_sha256'].items())
    out.mkdir(parents=True,exist_ok=False)
    s.save(out/'checks.json',dict(passed=True,sources=src,git_commit=commit,paired_initial_parameter_sha256=states[0],parameters=908552,
           original_matched_native_receipt=str(old),original_matched_native_receipt_sha256=s.sha(old),original_native_not_rerun=True,
           source_unchanged=True,model_forward_calls=0,test_arrays_read=0,test_evaluations=0,devices_opened=0,gpu_used=False,
           scope='Same original full-resolution physical graph/source; existing actual native proof reused, newdataset numeric gate still required'),exclusive=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);check(p.parse_args().out)
