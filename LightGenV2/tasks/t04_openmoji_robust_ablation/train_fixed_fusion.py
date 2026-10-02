"""Predeclared rank64 no-trick candidates; never select an artificially bad PT."""
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

from . import train as base
from .profiles import PROFILES, assert_contract, install
from LightGenV2.tasks.t04_semantic_interaction import training as t
from LightGenV2.tasks.t04_semantic_interaction.settings import Settings


def freeze_fusion(model, alpha):
    values = {}
    for name in ('language_core', 'vision_core'):
        core = getattr(model, name)
        if not core.fusion_alpha_min < alpha < core.fusion_alpha_max:
            raise ValueError('Fixed alpha outside the model fusion interval')
        core.reset_fusion_logits(alpha)
        for block in (1, 2):
            getattr(core, f'block{block}_optical_fusion_logit').requires_grad_(False)
            value = float(getattr(core, f'block{block}_optical_fusion').detach())
            assert abs(value - alpha) < 1e-6
            values[f'{name}.block{block}'] = value
    return values


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--candidate', choices=('alpha65', 'alpha80'), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--resume', type=Path)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    protocol = json.loads(args.config.read_text())
    alpha = protocol['candidates'][args.candidate]['alpha']
    epochs = 1 if args.smoke else protocol['epochs']
    steps = 2 if args.smoke else protocol['steps_per_epoch']
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()

    def write(name, value):
        target = out / name
        temp = target.with_suffix('.tmp')
        temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
        temp.replace(target)

    write('progress.json', {'status': 'initializing', 'candidate': args.candidate})
    try:
        torch.set_num_threads(4)
        torch.manual_seed(protocol['seed'])
        assert_contract()
        assert base.sha(base.SOURCE) == base.SOURCE_SHA
        cfg = Settings.__new__(Settings)
        cfg.__dict__.update(json.loads((base.BASE / 'resolved_config.json').read_text()))
        for key in ('config_path', 'data_dir', 'asset_dir', 'output_dir', 'qwen_checkpoint',
                    'prompt_cache_path', 'optical_base_config', 'legacy_warmstart_checkpoint'):
            setattr(cfg, key, Path(getattr(cfg, key)))
        cfg.shared_readout_variant = 'lowrank64'
        cfg.optical_fusion_initial = alpha
        cfg.output_dir, cfg.num_workers = out, 0
        device = torch.device('cuda')
        model = t.build_model(cfg, device)
        source = torch.load(base.SOURCE, map_location='cpu', weights_only=False)
        model.load_state_dict(base.adapted_source_state(source['model'], 'lowrank64', model.state_dict()), strict=True)
        resume_sha = None
        if args.resume:
            resumed = torch.load(args.resume, map_location='cpu', weights_only=False)
            assert abs(resumed['fixed_fusion_alpha'] - alpha) < 1e-9
            assert resumed['group'] == 'r0_base'
            assert resumed['settings']['shared_readout_variant'] == 'lowrank64'
            model.load_state_dict(resumed['model'], strict=True)
            for name in ('language_core', 'vision_core'):
                for block in (1, 2):
                    assert abs(float(getattr(getattr(model, name), f'block{block}_optical_fusion').detach()) - alpha) < 1e-6
            resume_sha = base.sha(args.resume)
        fusion = freeze_fusion(model, alpha)
        install(model, 'r0_base')
        train, test = t.build_loaders(cfg)
        assert len(train.dataset) == 5000 and len(test.dataset) == 1000
        if args.smoke:
            def small(loader):
                return DataLoader(Subset(loader.dataset, list(range(8))), batch_size=4,
                                  collate_fn=loader.collate_fn, num_workers=0)
            train, test = small(train), small(test)
        root = Path(__file__).resolve().parents[3]
        commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
        entry_blob = subprocess.check_output(['git', '-C', str(root), 'show',
                                             args.source_commit + ':' + Path(__file__).relative_to(root).as_posix()])
        assert hashlib.sha256(entry_blob).hexdigest() == base.sha(Path(__file__)), 'Published entry mismatch'
        dependency_shas = {str(path.relative_to(root)): base.sha(path) for path in
                           (Path(base.__file__), Path(__file__), Path(t.__file__),
                            Path(__file__).with_name('profiles.py'))}
        resolved = dict(protocol, candidate=args.candidate, fixed_alpha=fusion,
                        smoke=args.smoke, actual_epochs=epochs, actual_steps=steps,
                        runtime_git_head=commit, source_commit=args.source_commit,
                        source_entry_sha256=base.sha(Path(__file__)),
                        dependency_sha256=dependency_shas, command=sys.argv,
                        source_weight_sha256=base.SOURCE_SHA, profile=PROFILES['r0_base'],
                        head_parameters=sum(v.numel() for v in model.shared_readout.parameters()),
                        decoder_parameters=sum(v.numel() for v in model.shared_readout.decoder.parameters()),
                        device=torch.cuda.get_device_name(), torch_version=torch.__version__,
                        train_count=len(train.dataset), test_count=len(test.dataset))
        resolved.update(resume_checkpoint=str(args.resume) if args.resume else None,
                        resume_sha256=resume_sha, optimizer_restarted=bool(args.resume))
        write('protocol.json', resolved)
        write('resolved_config.json', cfg.to_dict())
        for key in ('learning_rate', 'adapter_learning_rate', 'phase_learning_rate',
                    'router_learning_rate', 'readout_learning_rate', 'decoder_learning_rate'):
            setattr(cfg, key, getattr(cfg, key) * .1)
        write('resolved_config.json', cfg.to_dict())
        optimizer = torch.optim.AdamW(t.legacy._parameter_groups(model, cfg), weight_decay=cfg.weight_decay)

        def evaluate():
            model.eval()
            t._set_phase_dropout(model, False)
            with torch.no_grad():
                return t.evaluate_with_routes(model, test, cfg, device)[0]

        def save(name, epoch, score):
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'group': 'r0_base',
                        'settings': cfg.to_dict(), 'bounded_amplitude': {'kind': 'tanh', 'scale': .5},
                        'source_sha256': base.SOURCE_SHA, 'fixed_fusion_alpha': alpha,
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
                result = model(batch['source_image'], batch['prompt_hidden'])
                loss = t.editing_objective(result, batch, cfg)['total']
                loss = loss + cfg.router_importance_weight * model.router_importance_loss()
                loss = loss + cfg.phase_dc_weight * t._phase_regularization(model, cfg)
                assert torch.isfinite(loss)
                loss.backward()
                for name, param in model.named_parameters():
                    if 'optical_fusion_logit' in name:
                        assert not param.requires_grad and param.grad is None
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
            row = {'epoch': epoch, 'loss': sum(losses) / len(losses),
                   'simulation_test': metrics, 'best': best, 'selected_epoch': selected,
                   'elapsed_seconds': time.time() - started}
            history.append(row)
            write('history.json', history)
            write('progress.json', dict(row, status='training'))
            print(json.dumps(dict(row, simulation_test=score)), flush=True)
        selected_payload = torch.load(out / 'best.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(selected_payload['model'], strict=True)
        actual_fusion = {f'{name}.block{block}': float(getattr(getattr(model, name), f'block{block}_optical_fusion').detach())
                         for name in ('language_core', 'vision_core') for block in (1, 2)}
        assert all(abs(value - alpha) < 1e-6 for value in actual_fusion.values())
        strict_metrics = evaluate()
        assert abs(base.metric(strict_metrics) - best) < 1e-9
        report = {'status': 'smoke_complete' if args.smoke else 'complete',
                  'candidate': args.candidate, 'protocol': resolved, 'selected_epoch': selected,
                  'selected_simulation': strict_metrics, 'fixed_alpha_reload': actual_fusion,
                  'best_sha256': base.sha(out / 'best.pt'), 'last_sha256': base.sha(out / 'last.pt'),
                  'physical_direct': None, 'physical_adapted': None,
                  'simulation_below_90': best < .9 if not args.smoke else None,
                  'elapsed_seconds': time.time() - started}
        write('report.json', report)
        write('progress.json', {'status': report['status'], 'selected_epoch': selected, 'best': best})
    except Exception as exc:
        write('progress.json', {'status': 'failed', 'error': repr(exc), 'elapsed_seconds': time.time() - started})
        raise


if __name__ == '__main__':
    main()
