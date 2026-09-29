"""Full-head continuation with TRAIN-only clean and matched-stress selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from LightGenV2.tasks.t04_semantic_interaction import training as t
from LightGenV2.tasks.t04_semantic_interaction.settings import Settings
from .evaluate_matched_stress import _activate_stress_only
from .profiles import PROFILES, assert_contract, install
from .train import BASE, OUTPUTS, SOURCE, SOURCE_SHA, metric, save_json, sha


SPLIT_SOURCE = OUTPUTS / 'fullhead_probe1_r1_ccd_20260929/split.json'
# The user-selected full-head source scores about 0.889 on the common clean
# TEST, and 0.962 on its TRAIN-derived holdout. Never substitute the older
# 0.95-TEST ablation baseline.
MIN_CLEAN_VALIDATION = 0.942
SEEDS = (1042, 1043)


def run(group: str, out: Path, epochs: int, steps: int,
        noise_scale: float, selection_profile: str | None = None,
        selection_noise_scale: float | None = None,
        paired_consistency_weight: float = 0.0) -> None:
    if group not in PROFILES:
        raise ValueError(group)
    selection_profile = selection_profile or group
    selection_noise_scale = selection_noise_scale or noise_scale
    if selection_profile not in PROFILES or selection_profile == 'r0_base':
        raise ValueError('Selection profile must apply a stated stress')
    if paired_consistency_weight < 0 or (paired_consistency_weight and group != 'r1_ccd'):
        raise ValueError('Paired consistency is currently defined only for CCD-only G3')
    assert_contract()
    if sha(SOURCE) != SOURCE_SHA:
        raise RuntimeError('User-selected full-head checkpoint SHA mismatch')
    out.mkdir(parents=True, exist_ok=False)
    cfg = Settings.__new__(Settings)
    cfg.__dict__.update(json.loads((BASE / 'resolved_config.json').read_text()))
    for key in ('config_path', 'data_dir', 'asset_dir', 'output_dir',
                'qwen_checkpoint', 'prompt_cache_path', 'optical_base_config',
                'legacy_warmstart_checkpoint'):
        setattr(cfg, key, Path(getattr(cfg, key)))
    cfg.output_dir = out
    cfg.num_workers = 0
    cfg.shared_readout_variant = 'standard'
    torch.manual_seed(73)
    device = torch.device('cuda')
    source_state = torch.load(SOURCE, map_location='cpu', weights_only=False)['model']

    def build(profile: str, *, training_noise_scale: float = 1.0,
              pixel_shift: int = 0):
        model = t.build_model(cfg, device)
        model.load_state_dict(source_state, strict=True)
        if sum(p.numel() for p in model.shared_readout.parameters()) != 381976:
            raise RuntimeError('Full-head architecture changed')
        install(model, profile, noise_scale=training_noise_scale,
                pixel_shift=pixel_shift)
        return model

    model = build(group, training_noise_scale=noise_scale)
    clean_model = build('r0_base').eval()
    stress_model = build(selection_profile,
                         training_noise_scale=selection_noise_scale,
                         pixel_shift=1).eval()
    t._set_phase_dropout(clean_model, False)
    t._set_phase_dropout(stress_model, False)
    _activate_stress_only(stress_model, selection_profile)
    train, test = t.build_loaders(cfg)
    assert len(test.dataset) == 1000
    split = json.loads(SPLIT_SOURCE.read_text())
    fit_ids, val_ids = split['fit'], split['validation']
    assert len(fit_ids) == 4000 and len(val_ids) == 1000
    assert not set(fit_ids).intersection(val_ids)
    fit = DataLoader(Subset(train.dataset, fit_ids), batch_size=32,
                     shuffle=True, collate_fn=train.collate_fn, num_workers=0)
    val = DataLoader(Subset(train.dataset, val_ids), batch_size=32,
                     collate_fn=train.collate_fn, num_workers=0)
    save_json(out / 'split.json', split)
    save_json(out / 'protocol.json', {
        'group': group, 'source_checkpoint': str(SOURCE),
        'source_sha256': sha(SOURCE), 'variant': 'standard',
        'source_clean_test_reference': 0.888,
        'shared_readout_parameters': 381976,
        'training_profile': PROFILES[group], 'training_noise_scale': noise_scale,
        'training_pixel_shift': 0, 'selection_pixel_shift': 1,
        'selection_profile': selection_profile,
        'paired_consistency_weight': paired_consistency_weight,
        'selection_noise_scale': selection_noise_scale, 'selection_seeds': SEEDS,
        'min_clean_validation': MIN_CLEAN_VALIDATION,
        'selection': 'max mean stressed TRAIN holdout VAL subject to clean VAL guard',
        'test_used_for_selection': False, 'epochs': epochs,
        'steps_per_epoch': steps,
        'stress_proxy_is_not_calibrated_hardware': True})
    for name in ('learning_rate', 'adapter_learning_rate',
                 'phase_learning_rate', 'router_learning_rate',
                 'readout_learning_rate', 'decoder_learning_rate'):
        setattr(cfg, name, getattr(cfg, name) * 0.1)
    optimizer = torch.optim.AdamW(t.legacy._parameter_groups(model, cfg),
                                  weight_decay=cfg.weight_decay)

    def evaluate():
        clean_model.load_state_dict(model.state_dict(), strict=True)
        stress_model.load_state_dict(model.state_dict(), strict=True)
        with torch.inference_mode():
            torch.manual_seed(SEEDS[0])
            clean = metric(t.evaluate_with_routes(clean_model, val, cfg, device)[0])
            stressed = []
            for seed in SEEDS:
                torch.manual_seed(seed)
                stressed.append(metric(t.evaluate_with_routes(
                    stress_model, val, cfg, device)[0]))
        return clean, stressed

    def save(name: str, epoch: int):
        torch.save({'model': model.state_dict(), 'epoch': epoch,
                    'group': group, 'source_sha256': sha(SOURCE),
                    'bounded_amplitude': {'kind': 'tanh', 'scale': 0.5},
                    'settings': cfg.to_dict()}, out / name)

    initial_clean, initial_stress = evaluate()
    initial_stress_mean = sum(initial_stress) / len(initial_stress)
    best_epoch, best_stress = 0, initial_stress_mean
    history = [{'epoch': 0, 'clean_validation': initial_clean,
                'stress_validation_by_seed': initial_stress,
                'stress_validation_mean': initial_stress_mean,
                'eligible': initial_clean >= MIN_CLEAN_VALIDATION}]
    save('best.pt', 0)
    save_json(out / 'history.json', history)
    for epoch in range(1, epochs + 1):
        model.train()
        model.vision_stem.eval()
        model.set_phase_trainable(True)
        t._set_phase_dropout(model, True)
        loss_sum = 0.0
        for n, raw in enumerate(fit, 1):
            batch = t.legacy._move(raw, device)
            optimizer.zero_grad(set_to_none=True)
            if paired_consistency_weight:
                # Keep all learned modules and phase-dropout mode fixed. Only
                # disable detector noise for this clean companion forward.
                model._profile_noise_enabled = False
                for path in model._optical_paths():
                    path.offset_fraction = 0.0
                    path.read_noise_fraction = 0.0
                clean_output = model(batch['source_image'], batch['prompt_hidden'])
                model._profile_noise_enabled = True
                for path in model._optical_paths():
                    path.offset_fraction = 0.03 * noise_scale
                    path.read_noise_fraction = 0.01 * noise_scale
            output = model(batch['source_image'], batch['prompt_hidden'])
            loss = t.editing_objective(output, batch, cfg)['total']
            if paired_consistency_weight:
                clean_loss = t.editing_objective(clean_output, batch, cfg)['total']
                loss = 0.5 * (loss + clean_loss)
                consistency = F.kl_div(
                    F.log_softmax(output['category_logits'].float(), dim=1),
                    F.softmax(clean_output['category_logits'].float().detach(), dim=1),
                    reduction='none').sum(dim=1).mean()
                consistency = consistency + F.mse_loss(
                    torch.sigmoid(output['edit_logits'].float()),
                    torch.sigmoid(clean_output['edit_logits'].float().detach()))
                loss = loss + paired_consistency_weight * consistency
            loss = loss + cfg.router_importance_weight * model.router_importance_loss()
            loss = loss + cfg.phase_dc_weight * t._phase_regularization(model, cfg)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach())
            if n >= steps:
                break
        clean, stressed = evaluate()
        stressed_mean = sum(stressed) / len(stressed)
        eligible = clean >= MIN_CLEAN_VALIDATION
        row = {'epoch': epoch, 'train_loss': loss_sum / n,
               'clean_validation': clean, 'stress_validation_by_seed': stressed,
               'stress_validation_mean': stressed_mean, 'eligible': eligible}
        history.append(row)
        save_json(out / 'history.json', history)
        if eligible and stressed_mean > best_stress:
            best_epoch, best_stress = epoch, stressed_mean
            save('best.pt', epoch)
        save('last.pt', epoch)
        print(json.dumps({'group': group, **row, 'best_epoch': best_epoch,
                          'best_stress': best_stress}), flush=True)
    save_json(out / 'report.json', {
        'status': 'complete', 'group': group,
        'source_sha256': sha(SOURCE), 'best_epoch': best_epoch,
        'best_sha256': sha(out / 'best.pt'),
        'initial_clean_validation': initial_clean,
        'initial_stress_validation_mean': initial_stress_mean,
        'best_stress_validation_mean': best_stress,
        'best_clean_validation': history[best_epoch]['clean_validation'],
        'test_evaluated': False,
        'stress_proxy_is_not_calibrated_hardware': True})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', choices=tuple(PROFILES), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--noise-scale', type=float, default=10.0)
    parser.add_argument('--selection-profile', choices=tuple(PROFILES))
    parser.add_argument('--selection-noise-scale', type=float)
    parser.add_argument('--paired-consistency-weight', type=float, default=0.0)
    args = parser.parse_args()
    run(args.group, args.output, args.epochs, args.steps, args.noise_scale,
        args.selection_profile, args.selection_noise_scale,
        args.paired_consistency_weight)


if __name__ == '__main__':
    main()
