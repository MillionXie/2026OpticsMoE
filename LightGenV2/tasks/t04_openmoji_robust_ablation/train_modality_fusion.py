"""Bounded clean-training comparison after the saved-CCD language-gap diagnostic.

No new modules: editor48/decoder64 and modality-specific gates are fixed before
the first gradient. Full original TEST selection is development, not a test of
independent generalization. Physical results require entirely new CCD captures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from . import train as base, split_rank_head
from .profiles import assert_contract, install, PROFILES
from LightGenV2.tasks.t04_semantic_interaction import training as t
from LightGenV2.tasks.t04_semantic_interaction.settings import Settings


def freeze_gates(model, candidate):
    result = {}
    for modality in ('language', 'vision'):
        core = getattr(model, modality + '_core')
        alpha = float(candidate[modality + '_alpha'])
        assert core.fusion_alpha_min < alpha < core.fusion_alpha_max
        core.reset_fusion_logits(alpha)
        for block in (1, 2):
            getattr(core, f'block{block}_optical_fusion_logit').requires_grad_(False)
            value = float(getattr(core, f'block{block}_optical_fusion').detach())
            assert abs(value - alpha) < 1e-6
            result[f'{modality}_core.block{block}'] = value
    return result


def settings(payload):
    cfg = Settings.__new__(Settings)
    cfg.__dict__.update(payload)
    for key in ('config_path', 'data_dir', 'asset_dir', 'output_dir',
                'qwen_checkpoint', 'prompt_cache_path', 'optical_base_config',
                'legacy_warmstart_checkpoint'):
        setattr(cfg, key, Path(getattr(cfg, key)))
    cfg.num_workers = 0
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    protocol = json.loads(args.config.read_text())
    candidate = protocol['candidates'][args.candidate]
    epochs = 1 if args.smoke else int(protocol['epochs'])
    steps = 2 if args.smoke else int(protocol['steps_per_epoch'])
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    start = time.time()

    def write(name, data):
        path = out / name
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(data, indent=2) + '\n')
        temporary.replace(path)

    write('progress.json', {'status': 'initializing'})
    try:
        torch.set_num_threads(4)
        torch.manual_seed(protocol['seed'])
        assert_contract()
        assert base.sha(base.SOURCE) == base.SOURCE_SHA
        root = Path(__file__).resolve().parents[3]
        relative = Path(__file__).relative_to(root).as_posix()
        blob = subprocess.check_output(['git', '-C', str(root), 'show',
                                        args.source_commit + ':' + relative])
        assert hashlib.sha256(blob).hexdigest() == base.sha(Path(__file__))
        cfg = settings(json.loads((base.BASE / 'resolved_config.json').read_text()))
        cfg.shared_readout_variant = 'lowrank64'
        cfg.editor_rank = int(candidate['editor_rank'])
        cfg.optical_fusion_initial = candidate['vision_alpha']
        cfg.output_dir = out
        device = torch.device('cuda')
        model = split_rank_head.build_model(cfg, device)
        source = torch.load(base.SOURCE, map_location='cpu', weights_only=False)
        model.load_state_dict(split_rank_head.adapted_source_state(
            source['model'], model.state_dict(), cfg.editor_rank), strict=True)
        gates = freeze_gates(model, candidate)
        install(model, 'r0_base')
        train, test = t.build_loaders(cfg)
        assert len(train.dataset) == 5000 and len(test.dataset) == 1000
        if args.smoke:
            def small(loader):
                return DataLoader(Subset(loader.dataset, list(range(8))),
                                  batch_size=4, collate_fn=loader.collate_fn,
                                  num_workers=0)
            train, test = small(train), small(test)
        for key in ('learning_rate', 'adapter_learning_rate', 'phase_learning_rate',
                    'router_learning_rate', 'readout_learning_rate', 'decoder_learning_rate'):
            setattr(cfg, key, getattr(cfg, key) * .1)
        resolved = dict(protocol, candidate=args.candidate, smoke=args.smoke,
                        fixed_fusion_alphas=gates, command=sys.argv,
                        runtime_head=subprocess.check_output(['git', '-C', str(root),
                            'rev-parse', 'HEAD'], text=True).strip(),
                        source_commit=args.source_commit, source_weight_sha256=base.SOURCE_SHA,
                        profile=PROFILES['r0_base'],
                        device=torch.cuda.get_device_name(),
                        train_count=len(train.dataset), test_count=len(test.dataset),
                        head_parameters=sum(p.numel() for p in model.shared_readout.parameters()),
                        decoder_parameters=sum(p.numel() for p in model.shared_readout.decoder.parameters()),
                        dependency_sha256={str(p.relative_to(root)): base.sha(p) for p in
                            (Path(__file__), Path(base.__file__), Path(t.__file__),
                             Path(split_rank_head.__file__), Path(__file__).with_name('profiles.py'))})
        assert resolved['head_parameters'] == 240664
        assert resolved['decoder_parameters'] == 30162
        write('protocol.json', resolved)
        write('resolved_config.json', cfg.to_dict())
        optimizer = torch.optim.AdamW(t.legacy._parameter_groups(model, cfg),
                                      weight_decay=cfg.weight_decay)

        def evaluate():
            model.eval()
            t._set_phase_dropout(model, False)
            with torch.no_grad():
                return t.evaluate_with_routes(model, test, cfg, device)[0]

        def save(name, epoch, score):
            torch.save({'model': model.state_dict(), 'epoch': epoch,
                        'group': 'r0_base', 'settings': cfg.to_dict(),
                        'bounded_amplitude': {'kind': 'tanh', 'scale': .5},
                        'fixed_fusion_alphas': gates, 'source_sha256': base.SOURCE_SHA,
                        'selection': protocol['selection'], 'score': score}, out / name)

        baseline = evaluate()
        best, selected = base.metric(baseline), 0
        history = [{'epoch': 0, 'simulation_test': baseline}]
        save('best.pt', 0, best)
        for epoch in range(1, epochs + 1):
            model.train()
            model.vision_stem.eval()
            model.set_phase_trainable(True)
            t._set_phase_dropout(model, True)
            losses = []
            for index, raw in enumerate(train):
                batch = t.legacy._move(raw, device)
                optimizer.zero_grad(set_to_none=True)
                output = model(batch['source_image'], batch['prompt_hidden'])
                loss = t.editing_objective(output, batch, cfg)['total']
                loss += cfg.router_importance_weight * model.router_importance_loss()
                loss += cfg.phase_dc_weight * t._phase_regularization(model, cfg)
                assert torch.isfinite(loss)
                loss.backward()
                for name, parameter in model.named_parameters():
                    if 'optical_fusion_logit' in name:
                        assert not parameter.requires_grad and parameter.grad is None
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                losses.append(float(loss.detach()))
                if index + 1 >= steps:
                    break
            metrics = evaluate() if epoch % 5 == 0 or epoch == epochs else None
            score = base.metric(metrics) if metrics else None
            if score is not None and score > best:
                best, selected = score, epoch
                save('best.pt', epoch, score)
            save('last.pt', epoch, score)
            row = dict(epoch=epoch, loss=sum(losses) / len(losses),
                       simulation_test=metrics, best=best, selected_epoch=selected,
                       elapsed_seconds=time.time() - start)
            history.append(row)
            write('history.json', history)
            write('progress.json', dict(row, status='training'))
            print(json.dumps(dict(row, simulation_test=score)), flush=True)
        payload = torch.load(out / 'best.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(payload['model'], strict=True)
        # Assert rather than reset, so the saved state is actually audited.
        actual = {f'{modality}_core.block{block}': float(getattr(
            getattr(model, modality + '_core'), f'block{block}_optical_fusion').detach())
            for modality in ('language', 'vision') for block in (1, 2)}
        assert all(abs(actual[key] - gates[key]) < 1e-6 for key in gates)
        strict = evaluate()
        assert abs(base.metric(strict) - best) < 1e-9
        report = dict(status='smoke_complete' if args.smoke else 'complete',
                      candidate=args.candidate, protocol=resolved,
                      selected_epoch=selected, selected_simulation=strict,
                      fixed_alpha_reload=actual, best_sha256=base.sha(out / 'best.pt'),
                      last_sha256=base.sha(out / 'last.pt'),
                      physical_direct=None, physical_adapted=None,
                      elapsed_seconds=time.time() - start)
        write('report.json', report)
        write('progress.json', dict(status=report['status'], best=best,
                                    selected_epoch=selected))
    except Exception as exc:
        write('progress.json', dict(status='failed', error=repr(exc),
                                    elapsed_seconds=time.time() - start))
        raise


if __name__ == '__main__':
    main()
