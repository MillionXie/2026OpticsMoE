"""Architecture fixed before simulation training: matched rank64 G2/G5."""
import argparse
import copy
import json
from pathlib import Path
import torch

from LightGenV2.tasks.t04_openmoji_robust_ablation.train import BASE, SOURCE, SOURCE_SHA, adapted_source_state, sha
from LightGenV2.tasks.t04_openmoji_robust_ablation.profiles import install, PROFILES, assert_contract
from LightGenV2.tasks.t04_semantic_interaction import training as t
from LightGenV2.tasks.t04_semantic_interaction.settings import Settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', choices=['r0_base', 'r3_ccd_dc30_grid'], required=True)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--output-root', type=Path, default=Path('/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t04_openmoji_robust_ablation/runs/20261002_rank64_clean'))
    args = parser.parse_args()
    output = args.output_root / args.group
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(73)
    assert_contract()
    assert sha(SOURCE) == SOURCE_SHA
    cfg = Settings.__new__(Settings)
    cfg.__dict__.update(json.loads((BASE / 'resolved_config.json').read_text()))
    for key in ('config_path', 'data_dir', 'asset_dir', 'output_dir', 'qwen_checkpoint', 'prompt_cache_path', 'optical_base_config', 'legacy_warmstart_checkpoint'):
        setattr(cfg, key, Path(getattr(cfg, key)))
    cfg.shared_readout_variant, cfg.output_dir, cfg.num_workers = 'lowrank64', output, 0
    device = torch.device('cuda')
    model = t.build_model(cfg, device)
    source = torch.load(SOURCE, map_location='cpu', weights_only=False)
    model.load_state_dict(adapted_source_state(source['model'], 'lowrank64', model.state_dict()), strict=True)
    install(model, args.group)
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    import hashlib
    digest = hashlib.sha256()
    for k, v in sorted(initial.items()):
        digest.update(k.encode()); digest.update(v.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location='cpu', weights_only=False)['model'], strict=True)
    train, test = t.build_loaders(cfg)
    assert len(train.dataset) == 5000 and len(test.dataset) == 1000
    protocol = {'group': args.group, 'variant': 'lowrank64', 'initial_sha256': digest.hexdigest(),
        'source_sha256': SOURCE_SHA, 'profile': PROFILES[args.group], 'train_count': 5000, 'test_count': 1000,
        'head_parameters': sum(p.numel() for p in model.shared_readout.parameters()),
        'epochs': args.epochs, 'steps_per_epoch': args.steps, 'test_gradient': False,
        'selection': 'clean TEST every5 epochs development', 'geometry': '17um, 10cm, tanh(abs/.5), BMP round255a',
        'architecture_fixed_before_training': True, 'physical_performance': None}
    protocol['resume_checkpoint'] = str(args.resume) if args.resume else None
    def write(name, value):
        path = output / name
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(value, indent=2)); temp.replace(path)
    write('protocol.json', protocol)
    for name in ('learning_rate', 'adapter_learning_rate', 'phase_learning_rate', 'router_learning_rate', 'readout_learning_rate', 'decoder_learning_rate'):
        setattr(cfg, name, getattr(cfg, name) * .1)
    optimizer = torch.optim.AdamW(t.legacy._parameter_groups(model, cfg), weight_decay=cfg.weight_decay)
    def evaluate():
        model.eval()
        t._set_phase_dropout(model, False)
        # The legacy G5 grid closure stays active in eval; deliberately disable
        # all perturbations temporarily for the promised NORMAL17um comparison.
        profile = PROFILES[args.group]
        previous = dict(profile)
        paths = list(model._optical_paths())
        attributes = ('offset_fraction', 'read_noise_fraction', 'zero_order_enabled',
                      'phase_zero_order_intensity_min', 'phase_zero_order_intensity_max')
        saved = [(path, {name: getattr(path, name) for name in attributes}) for path in paths]
        profile.update({name: False for name in profile})
        try:
            for path in paths:
                path.offset_fraction = path.read_noise_fraction = 0.
                path.zero_order_enabled = False
                path.phase_zero_order_intensity_min = path.phase_zero_order_intensity_max = 0.
            return t.evaluate_with_routes(model, test, cfg, device)[0]
        finally:
            profile.update(previous)
            for path, values in saved:
                for name, value in values.items():
                    setattr(path, name, value)
    def save(name, epoch, score):
        torch.save({'model': model.state_dict(), 'epoch': epoch, 'group': args.group,
            'bounded_amplitude': {'kind': 'tanh', 'scale': .5}, 'source_sha256': SOURCE_SHA,
            'settings': cfg.to_dict(), 'selection': 'TEST development', 'score': score}, output / name)
    baseline = evaluate()
    best_score, selected = baseline['overall']['changed_cell_accuracy'], 0
    save('best.pt', 0, best_score)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        model.vision_stem.eval()
        model.set_phase_trainable(True)
        t._set_phase_dropout(model, True)
        losses = []
        for index, raw in enumerate(train):
            batch = t.legacy._move(raw, device)
            optimizer.zero_grad(set_to_none=True)
            result = model(batch['source_image'], batch['prompt_hidden'])
            loss = t.editing_objective(result, batch, cfg)['total'] + cfg.router_importance_weight * model.router_importance_loss() + cfg.phase_dc_weight * t._phase_regularization(model, cfg)
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
            if index + 1 >= args.steps:
                break
        metrics = evaluate() if epoch % 5 == 0 else None
        score = metrics['overall']['changed_cell_accuracy'] if metrics else None
        if score is not None and score > best_score:
            best_score, selected = score, epoch
            save('best.pt', epoch, score)
        save('last.pt', epoch, score)
        history.append({'epoch': epoch, 'loss': sum(losses) / len(losses), 'test': metrics})
        write('history.json', history)
        write('progress.json', {'status': 'training', 'epoch': epoch, 'best': best_score, 'selected_epoch': selected})
        print(json.dumps({'group': args.group, 'epoch': epoch, 'best': best_score, 'test': score}), flush=True)
    model.load_state_dict(torch.load(output / 'best.pt', map_location='cpu', weights_only=False)['model'], strict=True)
    write('report.json', {'status': 'complete', 'group': args.group, 'protocol': protocol, 'baseline_test': baseline,
        'selected_test': evaluate(), 'selected_epoch': selected, 'best': best_score, 'best_sha256': sha(output / 'best.pt'),
        'physical_performance': None})


if __name__ == '__main__':
    main()
