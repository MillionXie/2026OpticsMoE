"""Small real-optics forward/backward check; no full Qwen model is loaded."""
import argparse
import json
from pathlib import Path
import torch
from .settings import load_settings
from .modeling import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', default='embedding_alpha40')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(4)
    cfg = load_settings(Path(__file__).parent / 'configs' / (args.profile + '.yaml'))
    device = torch.device(args.device)
    model = build_model(cfg, device)
    model.train()
    image = torch.rand(2, 3, 224, 224, device=device)
    tokens = [torch.randn(12, 2048, device=device), torch.randn(17, 2048, device=device)]
    output = model(image, tokens)
    loss = output['category_logits'].square().mean() + output['edit_logits'].square().mean()
    loss.backward()
    phases = {n: {'gradient_present': p.grad is not None,
                  'gradient_rms': None if p.grad is None else float(p.grad.float().square().mean().sqrt())}
              for n, p in model.named_parameters() if 'raw_phase' in n or 'raw_router_phase' in n}
    assert all(r['gradient_present'] for r in phases.values()), phases
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert output['category_logits'].shape == (2, 17, 6, 6)
    model.assert_contract()
    for core in (model.language_core, model.vision_core):
        core.set_fusion_ablation('remove_optical')
    with torch.no_grad():
        assert torch.isfinite(model(image, tokens)['category_logits']).all()
    print(json.dumps({'profile': args.profile, 'loss': float(loss.detach()), 'architecture': model.architecture_report(),
                      'phases': phases}, indent=2))


if __name__ == '__main__':
    main()
