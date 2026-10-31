"""CPU audit of the unchanged router's sampled propagator and RGB support.

Power iteration gives an observed Rayleigh value, not a certified upper bound.
The Frobenius bound applies to the discrete stored transfer function only.
No checkpoint, detector, input encoding, or training state is modified.
"""
import argparse
import json
from pathlib import Path
import torch
from .model_four import FourRouterClassification, CONTRACT
from .train import write


@torch.no_grad()
def audit(iterations=80):
    torch.set_num_threads(4);torch.manual_seed(17)
    model=FourRouterClassification('optical','ccd_grid',8).cpu()
    h=model.propagator.transfer_function.to(torch.complex128)
    support=torch.zeros((1,518,518),dtype=torch.float64)
    support[:,147:371,147:371]=1;support[:,259:371,259:371]=0
    kernel=torch.fft.ifft2(h)
    row_energy=torch.fft.ifft2(torch.fft.fft2(support)*torch.fft.fft2(kernel.abs().square())).real
    def forward(x):return torch.fft.ifft2(torch.fft.fft2(x)*h)
    def adjoint(x):return torch.fft.ifft2(torch.fft.fft2(x)*h.conj())
    x=torch.randn((1,518,518),dtype=torch.complex128);y=torch.randn_like(x)
    lhs=(forward(x).conj()*y).sum();rhs=(x.conj()*adjoint(y)).sum()
    adjoint_error=float((lhs-rhs).abs()/lhs.abs().clamp_min(1e-20))
    assert adjoint_error<1e-10
    actual=model.propagator(x.to(torch.complex64)).to(torch.complex128)
    kernel_replay_error=float((forward(x.to(torch.complex64).to(torch.complex128))-actual).norm()/actual.norm())
    assert kernel_replay_error<2e-6
    rows=[]
    for index,(cy,cx) in enumerate(model.router_centers):
        target=torch.zeros_like(support);target[:,cy-30:cy+30,cx-30:cx+30]=1
        frobenius=float((row_energy*target).sum())
        trials=[]
        for seed in (17,29):
            torch.manual_seed(seed);v=torch.randn_like(x)*support;v/=v.norm()
            for _ in range(iterations):
                w=adjoint(forward(v)*target)*support;v=w/w.norm().clamp_min(1e-30)
            w=adjoint(forward(v)*target)*support
            value=float((v.conj()*w).sum().real)
            residual=float((w-value*v).norm())
            assert value>=-1e-10 and value<=min(float(h.abs().square().max()),frobenius)+1e-8
            trials.append(dict(seed=seed,rayleigh_energy=value,eigen_residual=residual))
        rows.append(dict(port=index,detector_center=[cy,cx],frobenius_norm_squared=frobenius,
                         discrete_operator_energy_upper_bound=min(float(h.abs().square().max()),frobenius),
                         observed_Rayleigh_energy=max(t['rayleigh_energy'] for t in trials),trials=trials))
    return dict(role='CPU_DISCRETE_ROUTER_GEOMETRY_AUDIT_NOT_CLASSIFICATION',contract=CONTRACT,
                input_support='Three nonzero 112-square RGB tiles in central 224-square, fourth tile zero',
                relaxation='Any complex amplitude on RGB support; less restrictive than fixed-amplitude phase-only inputs',
                iterations_per_start=iterations,rows=rows,adjoint_relative_error=adjoint_error,
                canonical_float32_replay_relative_error=kernel_replay_error,
                interpretation='Rayleigh values are achievable relaxed-input estimates, not certified maxima. Upper bounds use the discrete stored kernel, not a hardware model. A small absolute capture does not preclude a large fraction after normalizing four detector energies. Do not change geometry or claim port unreachable from these diagnostics.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--iterations',type=int,default=80);args=parser.parse_args()
    if args.iterations<1:raise ValueError('positive iterations')
    args.out.mkdir(parents=True,exist_ok=False);result=audit(args.iterations)
    write(args.out/'result.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
