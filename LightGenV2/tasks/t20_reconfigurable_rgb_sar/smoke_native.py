"""CPU numerical/gradient checks of candidate graphs; not a hardware admission."""
import argparse
import json
from pathlib import Path
import torch

from .hardware_aware import NativeOEOOptics
from LightGenV2.tasks.t21_continual_learning_hardware.model import build_model


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    if args.out.exists():p.error('Do not overwrite prior evidence')
    torch.set_num_threads(2)
    torch.manual_seed(20261024)
    reports=[]
    amplitude=torch.rand(1,224,224)
    for task in (20,21):
        for architecture in ('moe','d2nn'):
            model=(NativeOEOOptics(architecture) if task==20 else build_model(architecture))
            if task==21:model.configure_stage(4)
            model.eval()
            out=model(amplitude)
            assert torch.isfinite(out['probabilities']).all()
            torch.testing.assert_close(out['probabilities'].sum(1),torch.ones(1))
            if architecture=='moe':
                torch.testing.assert_close(out['route_power'].sum(1),torch.ones(1))
            b=model.border
            measured={k:out[k][:,b:-b,b:-b].detach() for k in ('router','first','final') if out[k] is not None}
            with torch.inference_mode():
                replay=model(amplitude,measured=measured)
                error=float((out['logits']-replay['logits']).abs().max())
                torch.testing.assert_close(out['logits'],replay['logits'],atol=3e-6,rtol=3e-6)
            loss=-out['probabilities'][0,0].clamp_min(1e-20).log()
            loss.backward()
            gradients={name:float(v.grad.norm()) for name,v in model.named_parameters() if v.requires_grad and v.grad is not None}
            assert gradients and all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)
            assert gradients['global_phase']>0
            reports.append(dict(task=task,architecture=architecture,active=model.active,canvas=model.canvas,
                                replay_logit_maxerror=error,gradients=gradients,
                                hardware_admitted=False,performance_claim=False))
            del model,out,replay
    with args.out.open('x',encoding='utf-8') as f:json.dump(reports,f,indent=2)
    print(json.dumps(reports))


if __name__=='__main__':main()
