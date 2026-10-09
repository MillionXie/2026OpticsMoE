"""Fixed coherent phase leakage, local to T18; no edits to historical optics."""
import torch


def coherent_modulation(field, phase, rho):
    if not 0 <= rho <= 1:
        raise ValueError('rho must be an amplitude mixture in [0,1]')
    return field.to(torch.complex64) * (
        (1-rho)*torch.exp(1j*phase.to(device=field.device,dtype=torch.float32))+rho)


def install_residual(model, phase_type, rho):
    class ResidualPhase(phase_type):
        def forward(self, field):
            if self._dropout_enabled():
                raise RuntimeError('T18 disallows simultaneous phase dropout')
            return coherent_modulation(field,self.get_phase(),self.residual_rho)

    targets=[]
    for name,module in model.named_modules():
        if isinstance(module,phase_type) and (
            '.expert_bank.' in name or '.global_fcs.' in name):
            module.__class__=ResidualPhase
            module.residual_rho=float(rho)
            targets.append(name)
    cycles=model.net.num_cycles
    assert cycles in (2,3)
    assert len(targets)==10*cycles,targets
    model.net.expert_bank.vectorize_homogeneous_d2nn=False
    assert sum(p.numel() for p in model.parameters())==10000+439848*cycles
    return targets
