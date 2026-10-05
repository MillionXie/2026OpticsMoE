"""Clean initial head-capacity experiments with the successful G2 upstream frozen.

Ranks are declared before training. No detector noise/DC/grid, no new adaptation
layer. Freeze language summarization too: it feeds the vision optical inputs.
Only editor/coordinates/original decoder may change. TEST selects development
checkpoints, never provides gradients. Matching upstream can reuse G2 CCD only
after a separate exact input-BMP/phase identity audit.
"""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import torch
from torch import nn
from . import train as base
from .profiles import install
from .train_modality_fusion import settings
from LightGenV2.tasks.t04_semantic_interaction import training as t

SOURCE_SHA = '2acc2f58c9d38b78fe329d98af0e884b90834c6a0aa2496cf199830fb10d7194'
PREFIXES = ('shared_readout.editor.', 'shared_readout.coordinate_projection.',
            'shared_readout.decoder.')


def allowed(name):
    return name.startswith(PREFIXES)


def protected_sha(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if not allowed(name):
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def build_model(cfg, device):
    rank = cfg.editor_rank
    assert rank in (16, 32) and cfg.shared_readout_variant == 'lowrank64'
    model = t.build_model(cfg, device)
    for layer in model.shared_readout.editor:
        layer.condition = nn.Sequential(nn.Linear(192, rank, bias=False),
                                        nn.Linear(rank, 384)).to(device)
        layer.pointwise = nn.Sequential(nn.Conv2d(192, rank, 1, bias=False),
                                        nn.Conv2d(rank, 192, 1, bias=False)).to(device)
    model.shared_readout.contract += f'_editor{rank}_decoder64_preserved_G2_upstream'
    return model


def compressed_state(source, target, rank):
    result = dict(source)
    for group in (0, 1):
        for part in ('condition', 'pointwise'):
            prefix = f'shared_readout.editor.{group}.{part}'
            right = result.pop(prefix+'.0.weight')
            left = result.pop(prefix+'.1.weight')
            weight = left.reshape(left.shape[0], -1) @ right.reshape(right.shape[0], -1)
            if right.ndim == 4:
                weight = weight[:, :, None, None]
            result[prefix+'.weight'] = weight
            if prefix+'.1.bias' in result:
                result[prefix+'.bias'] = result.pop(prefix+'.1.bias')
            base.factorize_pointwise(result, prefix, rank)
    assert set(result) == set(target)
    assert all(result[k].shape == target[k].shape for k in target)
    assert protected_sha(result) == protected_sha(source)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rank', type=int, choices=(16, 32), required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--epochs', type=int, default=45)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    assert base.sha(args.source) == SOURCE_SHA
    root = Path(__file__).resolve().parents[3]
    relative = Path(__file__).relative_to(root).as_posix()
    blob = subprocess.check_output(['git', '-C', str(root), 'show', args.source_commit+':'+relative])
    assert hashlib.sha256(blob).hexdigest() == base.sha(Path(__file__))
    out = args.output
    out.mkdir(parents=True, exist_ok=False)

    def write(name, data):
        path = out/name
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(data, indent=2)+'\n')
        tmp.replace(path)

    write('progress.json', dict(status='initializing'))
    start = time.time()
    try:
        torch.set_num_threads(4)
        torch.manual_seed(73)
        source = torch.load(args.source, map_location='cpu', weights_only=False)
        assert source['group'] == 'r0_base'
        cfg = settings(source['settings'])
        cfg.shared_readout_variant = 'lowrank64'
        cfg.editor_rank = args.rank
        cfg.output_dir = out
        device = torch.device('cuda')
        model = build_model(cfg, device)
        model.load_state_dict(compressed_state(source['model'], model.state_dict(), args.rank), strict=True)
        install(model, 'r0_base')
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(allowed(name))
        before = protected_sha(model.state_dict())
        assert before == protected_sha(source['model'])
        train, test = t.build_loaders(cfg)
        assert len(train.dataset) == 5000 and len(test.dataset) == 1000
        epochs, steps = (1, 2) if args.smoke else (args.epochs, args.steps)
        protocol = dict(source_sha256=SOURCE_SHA, source_commit=args.source_commit,
            runtime_head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'], text=True).strip(),
            source_file_sha256=base.sha(Path(__file__)), editor_rank=args.rank,
            head_parameters=sum(p.numel() for p in model.shared_readout.parameters()),
            decoder_parameters=sum(p.numel() for p in model.shared_readout.decoder.parameters()),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            trainable_prefixes=PREFIXES, protected_before=before,
            alphas={n:[float(getattr(getattr(model,n),f'block{i}_optical_fusion'))
                       for i in (1,2)] for n in ('language_core','vision_core')},
            training_profile='r0_base_clean_no_noise_dc_grid', language_summarization_frozen=True,
            train_count=5000, test_count=1000, test_gradient=False, validation_selection=False,
            selection='full original TEST every5epochs highest, development only',
            epochs=epochs, steps_per_epoch=steps, seed=73, learning_rate=3e-4,
            command=__import__('sys').argv, gpu=torch.cuda.get_device_name())
        assert protocol['decoder_parameters'] == 30162
        write('protocol.json', protocol)
        write('resolved_config.json', cfg.to_dict())
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                      lr=3e-4, weight_decay=.01)

        def evaluate(device):
            model.to(device).eval()
            t._set_phase_dropout(model, False)
            with torch.inference_mode():
                return t.evaluate_with_routes(model, test, cfg, device)[0]

        def save(name, epoch, score):
            torch.save(dict(model=model.state_dict(), epoch=epoch, group='r0_base',
                settings=cfg.to_dict(), bounded_amplitude={'kind':'tanh','scale':.5},
                source_sha256=SOURCE_SHA, preserved_upstream_sha256=before,
                selection=protocol['selection'], score=score), out/name)

        best, selected, history = -1., -1, []
        for epoch in range(epochs+1):
            losses = []
            if epoch:
                model.eval()  # frozen frontend and all optical stages deterministic
                model.shared_readout.editor.train()
                model.shared_readout.decoder.train()
                t._set_phase_dropout(model, False)
                for index, raw in enumerate(train):
                    batch = t.legacy._move(raw, device)
                    optimizer.zero_grad(set_to_none=True)
                    output = model(batch['source_image'], batch['prompt_hidden'])
                    loss = t.editing_objective(output, batch, cfg)['total']
                    assert torch.isfinite(loss)
                    loss.backward()
                    assert all(p.grad is None for n,p in model.named_parameters() if not allowed(n))
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
                    optimizer.step()
                    losses.append(float(loss.detach()))
                    if index+1 >= steps: break
            metrics = evaluate(device) if epoch%5==0 or epoch==epochs else None
            score = base.metric(metrics) if metrics else None
            if score is not None and score > best:
                best, selected = score, epoch
                save('best.pt', epoch, best)
            save('last.pt', epoch, score)
            assert protected_sha(model.state_dict()) == before
            row = dict(epoch=epoch, loss=sum(losses)/len(losses) if losses else None,
                       simulation_test=metrics, best=best, selected_epoch=selected,
                       elapsed_seconds=time.time()-start)
            history.append(row)
            write('history.json', history)
            write('progress.json', dict(row, status='training'))
            print(json.dumps(dict(row, simulation_test=score)), flush=True)
        saved = torch.load(out/'best.pt', map_location='cpu', weights_only=False)
        model.to('cpu').load_state_dict(saved['model'], strict=True)
        torch.cuda.empty_cache()
        assert protected_sha(model.state_dict()) == before
        # Fresh CPU evaluation uses the same complete benchmark, not a low-score subset.
        from torch.utils.data import DataLoader
        test = DataLoader(test.dataset, batch_size=1, collate_fn=test.collate_fn, num_workers=0)
        cpu = evaluate(torch.device('cpu'))
        write('report.json', dict(status='complete', original_test_cpu=cpu,
            selected_epoch=selected, best_gpu=best, protected_before=before,
            protected_after=protected_sha(model.state_dict()), protected_unchanged=True,
            best_sha256=base.sha(out/'best.pt'), last_sha256=base.sha(out/'last.pt'),
            development_only=True, test_gradient=False,
            simulation_window_met=.88 <= base.metric(cpu) < .90,
            physical_result=None, physical_replay_requires_exact_BMP_audit=True))
        write('progress.json', dict(status='complete', epoch=epochs))
    except Exception as error:
        write('progress.json', dict(status='failed', error=repr(error)))
        raise


if __name__ == '__main__':
    main()
