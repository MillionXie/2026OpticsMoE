"""Run one pinned OpenMoji ablation group; TRAIN holdout selects, TEST once."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from LightGenV2.tasks.t04_semantic_interaction import training as t
from LightGenV2.tasks.t04_semantic_interaction.settings import Settings
from .profiles import PROFILES, assert_contract, install

ROOT = Path('/DATA/DATA1/guest3/2026OpticsMoE')
BASE = ROOT / 'LightGenV2/tasks/t04_semantic_interaction/runs/simulation/routerfill_shared_s73'
SOURCE = BASE / 'best_checkpoint.pt'
SOURCE_SHA = 'a69ddcee827749fb9202f9aef11ea45011e433d8b2f0151be2eec3db7dbff9eb'
OUTPUTS = ROOT / 'LightGenV2/tasks/t04_openmoji_robust_ablation/runs/20260928'


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def metric(m) -> float:
    return float(m['overall']['changed_cell_accuracy'])


def run(group: str, epochs: int, steps: int, quick: bool) -> None:
    assert group in PROFILES
    assert_contract()
    out = OUTPUTS / (group + ('_quick' if quick else ''))
    out.mkdir(parents=True, exist_ok=False)
    assert sha(SOURCE) == SOURCE_SHA
    cfg = Settings.__new__(Settings)
    cfg.__dict__.update(json.loads((BASE / 'resolved_config.json').read_text()))
    for key in ('config_path', 'data_dir', 'asset_dir', 'output_dir', 'qwen_checkpoint',
                'prompt_cache_path', 'optical_base_config', 'legacy_warmstart_checkpoint'):
        setattr(cfg, key, Path(getattr(cfg, key)))
    cfg.output_dir = out
    cfg.num_workers = 0
    cfg.shared_readout_variant = 'standard'
    device = torch.device('cuda')
    torch.manual_seed(73)
    model = t.build_model(cfg, device)
    payload = torch.load(SOURCE, map_location='cpu', weights_only=False)
    model.load_state_dict(payload['model'], strict=True)
    install(model, group)
    source_protected = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    train, test = t.build_loaders(cfg)
    order = torch.randperm(len(train.dataset), generator=torch.Generator().manual_seed(73)).tolist()
    cutoff = int(0.8 * len(order))

    def loader(ids, shuffle=False):
        return DataLoader(Subset(train.dataset, ids), batch_size=32, shuffle=shuffle,
                          collate_fn=train.collate_fn, num_workers=0)

    fit, val = loader(order[:cutoff], True), loader(order[cutoff:])
    save_json(out / 'split.json', {'fit': order[:cutoff], 'validation': order[cutoff:],
              'test_used_for_selection': False, 'warm_start_saw_original_train': True})
    save_json(out / 'protocol.json', {'group': group, 'profile': PROFILES[group],
              'initial_sha256': SOURCE_SHA, 'amplitude': 'tanh(abs/.5), zero/phase preserving; BMP round255a only',
              'grid': '17→8→17 differentiable raster proxy, not true 8um propagation',
              'selection': 'TRAIN-only holdout; best epoch including 0', 'test': 'once per selected group',
              'epochs': epochs, 'steps_per_epoch': steps})

    cfg.learning_rate *= 0.1
    cfg.adapter_learning_rate *= 0.1
    cfg.phase_learning_rate *= 0.1
    cfg.router_learning_rate *= 0.1
    cfg.readout_learning_rate *= 0.1
    cfg.decoder_learning_rate *= 0.1
    optimizer = torch.optim.AdamW(t.legacy._parameter_groups(model, cfg), weight_decay=cfg.weight_decay)

    def evaluate(dl):
        model.eval()
        t._set_phase_dropout(model, False)
        return t.evaluate_with_routes(model, dl, cfg, device)[0]

    def save(name, epoch):
        torch.save({'model': model.state_dict(), 'epoch': epoch, 'group': group,
                    'bounded_amplitude': {'kind': 'tanh', 'scale': 0.5},
                    'source_sha256': SOURCE_SHA, 'settings': cfg.to_dict()}, out / name)

    initial = evaluate(val)
    best_score, best_epoch = metric(initial), 0
    history = [{'epoch': 0, 'validation_changed_cell_accuracy': best_score}]
    save('best.pt', 0)
    for epoch in range(1, epochs + 1):
        model.train()
        model.vision_stem.eval()
        model.set_phase_trainable(True)
        t._set_phase_dropout(model, True)
        loss_sum, n = 0.0, 0
        for raw in fit:
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
            n += 1
            if n >= steps:
                break
        val_metrics = evaluate(val)
        score = metric(val_metrics)
        row = {'epoch': epoch, 'loss': loss_sum / n, 'validation_changed_cell_accuracy': score,
               'router_audit': val_metrics.get('router_audit')}
        history.append(row)
        save_json(out / 'history.json', history)
        if score > best_score:
            best_score, best_epoch = score, epoch
            save('best.pt', epoch)
        save('last.pt', epoch)
        print(json.dumps({'group': group, 'epoch': epoch, 'loss': row['loss'], 'val': score,
                          'best_epoch': best_epoch, 'best_val': best_score}), flush=True)

    if quick:
        save_json(out / 'report.json', {'status': 'quick_complete', 'best_epoch': best_epoch,
                  'best_validation': best_score, 'test_evaluated': False})
        return
    selected = torch.load(out / 'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(selected['model'], strict=True)
    final = evaluate(test)
    assert len(test.dataset) == 1000
    save_json(out / 'report.json', {'status': 'complete', 'group': group,
              'source_sha256': SOURCE_SHA, 'best_sha256': sha(out / 'best.pt'),
              'best_epoch': best_epoch, 'best_validation': best_score,
              'test': final, 'test_evaluated_once_after_selection': True,
              'source_train_seen_caveat': True, 'profile': PROFILES[group]})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--group', choices=tuple(PROFILES), required=True)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--quick', action='store_true')
    a = p.parse_args()
    run(a.group, a.epochs, a.steps, a.quick)


if __name__ == '__main__':
    main()
