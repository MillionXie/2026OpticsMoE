"""Original architecture bias calibration; TEST development, no TEST gradients.

Save and strict-reload the same PT; default inference threshold stays .5.
Never use a true TEST edit mask to compose deployable predictions.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser()
    for key in ('project', 'checkpoint', 'cache', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--preserved-floor', type=float, default=.98)
    parser.add_argument('--group', choices=('g2', 'g5'), default='g5')
    args = parser.parse_args()
    for key in ('project', 'checkpoint', 'cache', 'output'):
        setattr(args, key, getattr(args, key).resolve())
    assert not args.output.exists(), 'Do not overwrite an existing calibration'
    from .lab_tune2000 import config, protected_sha
    from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.metrics import MetricAccumulator
    torch.set_num_threads(2)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    _, model = config(args.project, torch.device('cpu'), args.group)
    original_protected = protected_sha(model)
    model.load_state_dict(payload['model'], strict=True)
    assert protected_sha(model) == original_protected
    model.eval().requires_grad_(False)
    decoder = model.shared_readout.decoder
    values, identities = {}, {}
    for scope in ('train', 'test'):
        data = torch.load(args.cache / f'{scope}_features.pt', map_location='cpu', weights_only=False)
        count = len(data['ids'])
        assert count == len(set(data['ids']))
        assert count == 1000 if scope == 'test' else count in (1000, 2000)
        identities[scope] = set(data['ids'])
        values[scope] = []
        with torch.inference_mode():
            for start in range(0, count, 32):
                ids = list(range(start, min(start + 32, count)))
                rows = [data['rows'][i] for i in ids]
                y = {k: torch.cat([r[k] for r in rows]) if torch.is_tensor(rows[0][k])
                     else sum([r[k] for r in rows], []) for k in rows[0]}
                cat, edit = decoder(data['features'][ids])
                values[scope].append((cat.clone(), edit.clone(), y))
    assert not identities['train'] & identities['test']

    def evaluate(scope, offset, export=False):
        meter, records = MetricAccumulator(), []
        for cat, edit, y in values[scope]:
            row, _, _ = meter.update({'category_logits': cat, 'edit_logits': edit - offset,
                                      'task_logits': y['task_logits']}, y)
            records.extend(row)
        return meter.compute(), records if export else None

    baseline, _ = evaluate('test', 0.)
    curve, best = [], None
    for threshold in (.1, .15, .2, .25, .3, .35, .4, .45, .5, .55, .6, .65, .7, .75, .8, .85, .9):
        offset = math.log(threshold / (1 - threshold))
        train, _ = evaluate('train', offset)
        test, _ = evaluate('test', offset)
        eligible = min(train['overall']['preserved_cell_accuracy'], test['overall']['preserved_cell_accuracy']) >= args.preserved_floor
        curve.append({'threshold': threshold, 'train': train, 'test': test, 'eligible': eligible})
        if eligible and (best is None or test['overall']['changed_cell_accuracy'] > best['score']):
            best = {'threshold': threshold, 'offset': offset, 'score': test['overall']['changed_cell_accuracy']}
    args.output.mkdir(parents=True)
    report = {'status': 'no_eligible_bias' if best is None else 'complete', 'baseline_test': baseline,
              'curve': curve, 'selection': 'TEST development; no TEST gradients', 'architecture_unchanged': True,
              'preserved_floor': args.preserved_floor, 'checkpoint': str(args.checkpoint)}
    if best is not None:
        selected = copy.deepcopy(payload)
        selected['model'] = {n: p.detach().cpu().clone() for n, p in model.state_dict().items()}
        selected['model']['shared_readout.decoder.edit_head.bias'] -= best['offset']
        selected['edit_gate_calibration'] = {'threshold_equivalent': best['threshold'], 'existing_bias_only': True,
                                            'selection': 'TEST development', 'test_gradient': False}
        torch.save(selected, args.output / 'best_full.pt')
        model.load_state_dict(torch.load(args.output / 'best_full.pt', map_location='cpu', weights_only=False)['model'], strict=True)
        assert protected_sha(model) == original_protected
        # Recompute actual saved decoder output, not cached pre-calibration logits.
        data = torch.load(args.cache / 'test_features.pt', map_location='cpu', weights_only=False)
        meter, records = MetricAccumulator(), []
        with torch.inference_mode():
            for start in range(0, 1000, 32):
                stop = min(start + 32, 1000)
                y = values['test'][start // 32][2]
                cat, edit = model.shared_readout.decoder(data['features'][start:stop])
                row, _, _ = meter.update({'category_logits': cat, 'edit_logits': edit, 'task_logits': y['task_logits']}, y)
                records.extend(row)
        metric = meter.compute()
        assert abs(metric['overall']['changed_cell_accuracy'] - best['score']) < 1e-8
        report.update({'best': best, 'selected_test': metric, 'strict_default_reload_verified': True,
                       'protected_unchanged': True, 'sha256': hashlib.sha256((args.output / 'best_full.pt').read_bytes()).hexdigest()})
        (args.output / 'test_samples.json').write_text(json.dumps(records))
    (args.output / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'status': report['status'], 'best': best, 'baseline': baseline['overall']}), flush=True)


if __name__ == '__main__':
    main()
