"""Task-local execution batching; the optical graph and residual formula stay fixed."""
import types
import butterfly_ablation as optical

torch = optical.torch


def expert_stage_batched(bank, stage_index, local_fields):
    experts = list(bank.experts)
    reference = experts[0]
    fields = torch.stack(local_fields, dim=1).to(torch.complex64)
    phases = torch.stack([e.phase_layers[stage_index].get_phase() for e in experts])
    rho = torch.tensor([e.phase_layers[stage_index].residual_rho for e in experts],
                       device=fields.device, dtype=torch.float32).reshape(1, -1, 1, 1)
    assert all(not e.phase_layers[stage_index]._dropout_enabled() for e in experts)
    fields = fields * ((1-rho) * torch.exp(1j * phases.to(fields.device)).unsqueeze(0) + rho)
    padding = int(reference.propagation_padding)
    if padding:
        fields = torch.nn.functional.pad(fields, (padding,) * 4)
    batch, count, height, width = fields.shape
    fields = reference.propagators[stage_index].propagator(fields.reshape(batch*count, height, width))
    fields = fields.reshape(batch, count, height, width)
    if padding:
        size = int(reference.size)
        fields = fields[:, :, padding:padding+size, padding:padding+size]
    return list(fields.unbind(1))


def enable_expert_batching(model):
    bank = model.net.expert_bank
    assert len(bank.experts) == 9
    assert all(hasattr(p, 'residual_rho') for e in bank.experts for p in e.phase_layers)
    bank._forward_homogeneous_d2nn_stage_batched = types.MethodType(expert_stage_batched, bank)
    bank.vectorize_homogeneous_d2nn = True


def detector(classes):
    assert 9 <= classes <= 60
    cols = min(8, classes)
    rows = (classes+cols-1)//cols
    counts = [cols] * (rows-1) + [classes-cols*(rows-1)]
    width, height = cols*32+(cols-1)*28, rows*32+(rows-1)*28
    return dict(detector_size=32, layout='fixed_2x2', normalize_detector_energy=True,
                start_pos_x=(500-width)//2, start_pos_y=(500-height)//2,
                N_det_sets=counts, det_steps_x=[28]*rows, det_steps_y=28,
                start_pos_x_per_row=[(500-(n*32+(n-1)*28))//2 for n in counts])


def build(depth, profile, cfg, batched=True, device='cuda'):
    graph = optical.graph_config(depth, profile)
    model = optical.MultiClassMoE(graph, len(cfg['classes']))
    with torch.no_grad():
        for name, p in model.named_parameters():
            assert name.endswith('raw_phase')
            p.uniform_(-cfg['phase_init_raw_uniform'], cfg['phase_init_raw_uniform'])
    optical.install_residual(model, cfg['residual_rho'])
    if batched:
        enable_expert_batching(model)
    assert sum(p.numel() for p in model.parameters()) == 10000+439848*(depth//2)
    assert not any(isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)) for m in model.modules())
    assert tuple(model.masks.shape) == (len(cfg['classes']), 500, 500)
    return model.to(device)
