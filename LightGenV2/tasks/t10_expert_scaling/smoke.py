"""CUDA physics/gradient gates before starting training; no test data."""
import argparse
import copy
import json
import time
from pathlib import Path
import torch
from .model import ScalingOptics,ASM,encode_rgb
from .plan import load_protocol
from .train import save


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--experts',type=int,default=4)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(17);cfg=load_protocol();results=[]
    x=torch.rand(2,3,112,112,device='cuda');y=torch.tensor([0,1],device='cuda')
    for arch in ['moe_oeo','d2nn_total_parameter','d2nn_same_aperture']:
        model=ScalingOptics(cfg,a.experts,1,arch,layers=6).cuda();model.train()
        t=time.time();out=model(x)
        assert torch.allclose(out['probabilities'].sum(1),torch.ones(2,device='cuda'),atol=1e-5)
        loss=-out['probabilities'][torch.arange(2),y].log().mean();loss.backward()
        grad={n:float(p.grad.norm()) if p.grad is not None else None for n,p in model.named_parameters()}
        assert all(v is not None and v>0 and torch.isfinite(torch.tensor(v)) for v in grad.values()),grad
        if model.is_moe:
            assert torch.equal(out['route_mask'].sum(1),torch.ones(2,device='cuda'))
            amp,_,_=model.route(encode_rgb(x,model.router_side));assert torch.allclose(amp.square().sum(1),torch.ones(2,device='cuda'),atol=1e-6)
        model.eval()
        with torch.no_grad():
            first=model(x)['probabilities'];second=model(x)['probabilities']
            assert torch.equal(first,second)
        expected=model.geo['moe_phase_parameters'] if model.is_moe else (model.geo['d2nn_parameter_count'] if arch=='d2nn_total_parameter' else model.geo['d2nn_aperture_count'])
        assert sum(p.numel() for p in model.parameters())==expected
        results.append(dict(arch=arch,classification_gradients=grad,seconds=time.time()-t,
                            parameters=expected,peak_memory_bytes=torch.cuda.max_memory_allocated()))
        del model,out;torch.cuda.empty_cache()
    # Propagation convergence at the actual geometry, random and smooth fields.
    g=ScalingOptics(cfg,a.experts,1,layers=6).geo;c=g['canvas_side_px']
    p2=ASM(c,padding=2).cuda();p4=ASM(c,padding=4).cuda()
    field=torch.zeros(1,c,c,device='cuda',dtype=torch.complex64)
    source=encode_rgb(x[:1]);start=(c-224)//2;field[:,start:start+224,start:start+224]=source
    with torch.no_grad():
        u=p2(field);v=p4(field)
        relative=float((u-v).abs().norm()/v.abs().norm())
        intensity_error=float((u.abs().square()-v.abs().square()).norm()/v.abs().square().norm())
    summary=dict(experts=a.experts,gradient_checks=results,asm_relative_field_error=relative,
                 asm_relative_intensity_error=intensity_error,passed=relative<.01 and intensity_error<.01)
    save(a.out/'result.json',summary);print(json.dumps(summary),flush=True)
    assert summary['passed'],summary


if __name__=='__main__':main()
