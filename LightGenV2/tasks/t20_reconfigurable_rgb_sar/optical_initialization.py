"""Finite training-only restart of EXISTING optical/head parameters.

Uses all native phase codes, without changing geometry, frozen frontend, gains,
or inference operators. Published parent checkpoints are never modified.
"""
import torch


@torch.no_grad()
def restart_existing_parameters(model,config):
    if not config.get('restart_existing_optics',False):return None
    generator=torch.Generator(device='cpu').manual_seed(config['seed']+1907)
    changed={}
    for name,p in model.named_parameters():
        if 'phase' not in name:continue
        codes=torch.randint(0,256,p.shape,generator=generator)
        raw=torch.logit((codes.float()+.5)/256)
        p.copy_(raw.to(p.device))
        changed[name]=dict(parameters=p.numel(),minimum_code=int(codes.min()),maximum_code=int(codes.max()))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config['seed']+1908)
        weight=torch.empty(model.shared_head.weight.shape)
        torch.nn.init.xavier_uniform_(weight,gain=.1)
        model.shared_head.weight.copy_(weight.to(model.shared_head.weight.device))
    return dict(training_initialization_only=True,phase_blocks=changed,original_readout_reset=True,
                frontend_changed=False,display_gains_changed=False,new_parameters=0)
