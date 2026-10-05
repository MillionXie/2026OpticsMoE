"""Continue a disturbed OpenMoji group, selecting only on clean TRAIN holdout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from LightGenV2.tasks.t04_semantic_interaction import training as t
from LightGenV2.tasks.t04_semantic_interaction.settings import Settings
from .profiles import PROFILES, assert_contract, install
from .train import BASE, OUTPUTS, SOURCE_SHA, metric, save_json, sha


def run(group: str, epochs: int, steps: int, run_name: str) -> None:
    if group not in PROFILES or group == 'r0_base':
        raise ValueError('Recovery applies only to disturbed training groups')
    assert_contract()
    source_dir = OUTPUTS / f'compact_lowrank16_{group}'
    source = source_dir / 'best.pt'
    original_report = json.loads((source_dir / 'report.json').read_text())
    assert original_report['status'] == 'complete'
    assert original_report['source_sha256'] == SOURCE_SHA
    assert sha(source) == original_report['best_sha256']
    out = OUTPUTS / run_name
    out.mkdir(parents=True, exist_ok=False)

    cfg = Settings.__new__(Settings)
    cfg.__dict__.update(json.loads((BASE / 'resolved_config.json').read_text()))
    for key in ('config_path', 'data_dir', 'asset_dir', 'output_dir', 'qwen_checkpoint',
                'prompt_cache_path', 'optical_base_config', 'legacy_warmstart_checkpoint'):
        setattr(cfg, key, Path(getattr(cfg, key)))
    cfg.output_dir = out
    cfg.num_workers = 0
    cfg.shared_readout_variant = 'lowrank16'
    torch.manual_seed(73)
    device = torch.device('cuda')
    model = t.build_model(cfg, device)
    payload = torch.load(source, map_location='cpu', weights_only=False)
    assert payload['group'] == group
    model.load_state_dict(payload['model'], strict=True)
    install(model, group)
    clean_model = t.build_model(cfg, device).eval()
    install(clean_model, 'r0_base')

    train, test = t.build_loaders(cfg)
    assert len(test.dataset) == 1000
    split = json.loads((source_dir / 'split.json').read_text())
    fit_ids, val_ids = split['fit'], split['validation']
    assert len(fit_ids) == 4000 and len(val_ids) == 1000
    assert not set(fit_ids).intersection(val_ids)

    def loader(ids, shuffle=False):
        return DataLoader(Subset(train.dataset, ids), batch_size=32, shuffle=shuffle,
                          collate_fn=train.collate_fn, num_workers=0)

    fit, val = loader(fit_ids, True), loader(val_ids)
    save_json(out / 'split.json', split)
    save_json(out / 'protocol.json', {
        'group': group, 'training_profile': PROFILES[group],
        'selection_profile': PROFILES['r0_base'], 'variant': 'lowrank16',
        'source_checkpoint': str(source), 'source_sha256': sha(source),
        'epochs': epochs, 'steps_per_epoch': steps,
        'fit_samples': len(fit_ids), 'validation_samples': len(val_ids),
        'test_evaluated': False,
        'note': 'Train with the group perturbation; choose best on clean TRAIN holdout only.'})

    for name in ('learning_rate', 'adapter_learning_rate', 'phase_learning_rate',
                 'router_learning_rate', 'readout_learning_rate', 'decoder_learning_rate'):
        setattr(cfg, name, getattr(cfg, name) * 0.1)
    optimizer = torch.optim.AdamW(t.legacy._parameter_groups(model, cfg),
                                  weight_decay=cfg.weight_decay)

    def clean_validation():
        clean_model.load_state_dict(model.state_dict(), strict=True)
        clean_model.eval()
        t._set_phase_dropout(clean_model, False)
        return t.evaluate_with_routes(clean_model, val, cfg, device)[0]

    def save(name: str, epoch: int):
        torch.save({'model': model.state_dict(), 'epoch': epoch, 'group': group,
                    'source_sha256': sha(source), 'selection_profile': 'r0_base',
                    'settings': cfg.to_dict()}, out / name)

    initial = clean_validation()
    best_score, best_epoch = metric(initial), 0
    history = [{'epoch': 0, 'clean_validation_changed_cell_accuracy': best_score}]
    save('best.pt', 0)
    for epoch in range(1, epochs + 1):
        model.train()
        model.vision_stem.eval()
        model.set_phase_trainable(True)
        t._set_phase_dropout(model, True)
        loss_sum = 0.0
        for n, raw in enumerate(fit, 1):
            batch = t.legacy._move(raw, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch['source_image'], batch['prompt_hidden'])
            loss = t.editing_objective(output, batch, cfg)['total']
            loss = loss + cfg.router_importance_weight * model.router_importance_loss()
            loss = loss + cfg.phase_dc_weight * t._phase_regularization(model, cfg)
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach())
            if n >= steps:
                break
        score = metric(clean_validation())
        history.append({'epoch': epoch, 'loss': loss_sum / n,
                        'clean_validation_changed_cell_accuracy': score})
        save_json(out / 'history.json', history)
        if score > best_score:
            best_score, best_epoch = score, epoch
            save('best.pt', epoch)
        save('last.pt', epoch)
        print(json.dumps({'group': group, 'epoch': epoch, 'clean_val': score,
                          'best_epoch': best_epoch, 'best_clean_val': best_score}), flush=True)
    save_json(out / 'report.json', {'status': 'complete', 'group': group,
              'best_epoch': best_epoch, 'best_clean_validation': best_score,
              'initial_clean_validation': metric(initial),
              'best_sha256': sha(out / 'best.pt'), 'test_evaluated': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', required=True, choices=tuple(PROFILES))
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--run-name', required=True)
    args = parser.parse_args()
    run(args.group, args.epochs, args.steps, args.run_name)
