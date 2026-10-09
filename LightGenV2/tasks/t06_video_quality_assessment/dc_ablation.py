"""Pinned Temporal 0.8044 coherent unmodulated-field ablation.

Evaluate the exact archived test fields before attempting matched fine-tuning.
All six propagations remain enabled; rho=0 removes only the unmodulated field.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import io
import json
import os
import random
import subprocess
import sys
import zipfile
from pathlib import Path

import torch

from .lab_runtime import forward, load_model, sha, write
from .models.multivideo9x4 import build_model
from .multivideo_settings import resolved_dict

TASK = Path(__file__).resolve().parent


def ablation_settings(settings, output, arm):
    if arm not in ('dc20', 'dc0'):
        raise ValueError('Unknown ablation arm')
    values = dict(output_dir=Path(output), unmodulated_ablation=True,
                  phase_snapshot_interval_epochs=0)
    if arm == 'dc0':
        values.update(unmodulated_power_fraction_min=0.0,
                      unmodulated_power_fraction_max=0.0,
                      unmodulated_power_fraction_eval=0.0)
    result = dataclasses.replace(settings, **values)
    result.validate()
    return result


def _run_root(value):
    root = Path(value).resolve()
    allowed = (TASK / 'runs' / 'simulation').resolve()
    if allowed not in root.parents or 'dc_ablation' not in root.name:
        raise ValueError('Output must be a named dc_ablation run under T06/runs/simulation')
    if root.exists():
        raise FileExistsError(f'Refusing to overwrite an existing run: {root}')
    return root


def _manifest(root, checkpoint, settings, phase):
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    tracked = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True)
    if tracked.strip():
        raise RuntimeError('Publish tracked source edits before running a formal ablation')
    root.mkdir(parents=True, exist_ok=False)
    write(root / 'run_manifest.json', dict(
        git_commit=head, command=[sys.executable, *sys.argv], phase=phase,
        checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
        torch=torch.__version__, cuda=torch.version.cuda,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        test_used_for_selection=phase == 'train', validation_used=False,
        interpretation='development metrics; historical checkpoint selected on TEST',
        optical_propagations=6, optical_router='top2', videos_per_field=16,
        frames_per_video=4, intervention='only nominal unmodulated power coefficient',
    ))
    write(root / 'resolved_config.json', resolved_dict(settings))


def audit(args):
    root = _run_root(args.output)
    checkpoint = Path(args.checkpoint).resolve()
    model, settings = load_model('temporal', checkpoint, 'cpu')
    _manifest(root, checkpoint, settings, 'audit')
    from experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.metrics import regression_metrics
    package = Path(args.package).resolve()
    rows = []
    with zipfile.ZipFile(package) as archive:
        release_bytes = archive.read('release.json')
        release = json.loads(release_bytes)
        manifest = json.loads(archive.read('SHA256.json'))
        import hashlib
        if hashlib.sha256(release_bytes).hexdigest() != manifest['release.json']:
            raise ValueError('Release SHA mismatch')
        if release['target'] != 'temporal' or release['checkpoint_sha256'] != sha(checkpoint):
            raise ValueError('Archived fields and checkpoint have different identities')
        if not release['full_test'] or release['test_videos_in_package'] != 558:
            raise ValueError('Requires the complete pinned 558-video release')
        batches = []
        for item in release['fields']:
            data = archive.read(item['file'])
            if hashlib.sha256(data).hexdigest() != item['sha256'] or item['sha256'] != manifest[item['file']]:
                raise ValueError('Field identity mismatch: ' + item['key'])
            batches.append((item, torch.load(io.BytesIO(data), map_location='cpu', weights_only=False)))
    write(root / 'input_identity.json', dict(package=str(package), package_sha256=sha(package),
        field_count=len(batches), full_test=True, checkpoint_sha256=sha(checkpoint)))
    if args.vision_cache_path or args.language_cache_path:
        if not (args.vision_cache_path and args.language_cache_path):
            raise ValueError('Cache verification requires both vision and language paths')
        vision_path = Path(args.vision_cache_path).resolve()
        language_path = Path(args.language_cache_path).resolve()
        vision = torch.load(vision_path, map_location='cpu', weights_only=False)
        language = torch.load(language_path, map_location='cpu', weights_only=False)
        index = {sample: i for i, sample in enumerate(vision['sample_ids'])}
        checked = set()
        for item, batch in batches:
            for slot, valid in enumerate(item['valid']):
                if not valid:
                    continue
                sample = item['sample_ids'][slot]
                source = index[sample]
                if vision['splits'][source] != 'test':
                    raise ValueError('Recovered test sample has different split')
                for key in ('vision_tokens', 'quality_tokens'):
                    if not torch.equal(vision[key][source], batch[key][0, slot]):
                        raise ValueError(f'Recovered cache differs: {sample}, {key}')
                checked.add(sample)
            for key, recovered in (('language_tokens', language['language_tokens']),
                                   ('language_mask', language['attention_mask'])):
                if not torch.equal(recovered, batch[key]):
                    raise ValueError('Recovered language input differs from archived fields')
        if len(checked) != 558:
            raise ValueError('Incomplete recovered cache verification')
        write(root/'cache_recovery.json', dict(
            vision_path=str(vision_path), vision_sha256=sha(vision_path),
            language_path=str(language_path), language_sha256=sha(language_path),
            test_samples_bitwise_equal=558,
            train_provenance='reconstructed cache, not recovered original bytes; front identity and split checked by loader',
        ))
        del vision, language
    results = {}
    for arm in ('dc20', 'dc0'):
        chosen = ablation_settings(settings, root, arm)
        candidate = build_model(chosen)
        candidate.load_state_dict(model.state_dict(), strict=True)
        candidate.eval().to(args.device)
        predictions, targets, reference = [], [], []
        with torch.inference_mode():
            for item, batch in batches:
                output = forward(candidate, batch)
                scores = output['prediction'].detach().cpu().flatten().tolist()
                for slot, valid in enumerate(item['valid']):
                    if valid:
                        target = item['targets'][slot]
                        predictions.append(scores[slot]); targets.append(target)
                        reference.append(item['simulation_prediction'][slot])
                        rows.append(dict(arm=arm, field=item['key'], slot=slot,
                            sample_id=item['sample_ids'][slot], target=target, prediction=scores[slot]))
        if len(set(row['sample_id'] for row in rows if row['arm'] == arm)) != 558:
            raise ValueError('Test identities are not 558 unique videos')
        metrics = regression_metrics(torch.tensor(predictions), torch.tensor(targets), 'temporal')
        if arm == 'dc20':
            delta = max(abs(x-y) for x,y in zip(predictions,reference))
            if delta > .01 or abs(metrics['srcc']-release['simulation_metrics']['srcc']) > .00015:
                raise RuntimeError('Canonical 0.8044 baseline did not reproduce')
        results[arm] = dict(metrics=metrics,
            nominal_training_power=[chosen.unmodulated_power_fraction_min, chosen.unmodulated_power_fraction_max],
            nominal_eval_power=chosen.unmodulated_power_fraction_eval,
            alpha=[float(layer.alpha.detach()) for layer in candidate.fusions],
            router=candidate.settings.top_k)
        print(arm, json.dumps(results[arm]), flush=True)
        del candidate
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    with (root/'test_predictions.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    results['srcc_delta_dc20_minus_dc0'] = results['dc20']['metrics']['srcc']-results['dc0']['metrics']['srcc']
    results['interpretation'] = 'same checkpoint intervention, not separately optimized DC-free model'
    results['status'] = 'complete'
    write(root/'comparison.json',results)
    write(root/'status.json',dict(status='complete',gpu_process_exits_after_return=True))


def train_arm(args):
    root = _run_root(args.output)
    checkpoint = Path(args.checkpoint).resolve()
    model, settings = load_model('temporal', checkpoint, 'cpu')
    settings = ablation_settings(settings, root, args.arm)
    overrides = dict(epochs=args.epochs, batch_size=args.batch_size, random_seed=args.seed,
        learning_rate=args.learning_rate, phase_learning_rate=args.phase_learning_rate,
        router_phase_learning_rate=args.router_phase_learning_rate, num_workers=0,
        initialization_checkpoint=checkpoint)
    for key in ('manifest_path','vision_cache_path','language_cache_path'):
        value=getattr(args,key)
        if value: overrides[key]=Path(value).resolve()
    settings=dataclasses.replace(settings,**overrides);settings.validate()
    if not args.audit_output:
        raise ValueError('Training requires --audit-output from the completed canonical audit')
    audited = Path(args.audit_output)
    comparison = json.loads((audited/'comparison.json').read_text())
    recovery = json.loads((audited/'cache_recovery.json').read_text())
    if comparison.get('status') != 'complete':
        raise ValueError('Canonical baseline audit is incomplete')
    for key in ('vision', 'language'):
        if sha(getattr(settings, key+'_cache_path')) != recovery[key+'_sha256']:
            raise ValueError('Training cache differs from audited recovery')
    from experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.data import load_single_metric_cache
    missing=[str(getattr(settings,k)) for k in ('manifest_path','vision_cache_path','language_cache_path','training_soft_targets_path')
             if getattr(settings,k) is not None and not Path(getattr(settings,k)).is_file()]
    if missing:
        raise FileNotFoundError('Original matched training assets unavailable: '+json.dumps(missing))
    payload=load_single_metric_cache(settings)
    if payload['splits'].count('train')!=2250 or payload['splits'].count('test')!=558:
        raise ValueError('Matched ablation requires original TRAIN2250/TEST558')
    _manifest(root,checkpoint,settings,'train')
    write(root/'cache_recovery.json',recovery)
    write(root/'audit_identity.json',dict(path=str(audited.resolve()),
        comparison_sha256=sha(audited/'comparison.json')))
    # Rebuild and STRICT reload all parameters; no name/shape warm-start filtering.
    candidate=build_model(settings);candidate.load_state_dict(model.state_dict(),strict=True)
    random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    # Match the RNG draw budget of the original random-rho arm. A zero-rho
    # condition otherwise omits one draw per optical pass, shifting every later
    # dropout/shift random sample. The hook is scoped to this one worker only.
    from .models import multivideo9x4
    original=multivideo9x4._phase_modulation
    def matched_modulation(raw, *, settings, training):
        result=original(raw,settings=settings,training=training)
        if training and args.arm=='dc0':
            torch.rand((),device=raw.device)
        return result
    multivideo9x4._phase_modulation=matched_modulation
    from .multivideo_training import train
    try:
        result=train(candidate,payload,settings,torch.device(args.device))
        write(root/'status.json',dict(status='complete',summary=result))
    finally:
        multivideo9x4._phase_modulation=original


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('audit','train'),required=True)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--package',help='Pinned original full-test ZIP, required for audit')
    parser.add_argument('--arm',choices=('dc20','dc0'))
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--seed',type=int,default=163)
    parser.add_argument('--learning-rate',type=float,default=3e-5)
    parser.add_argument('--phase-learning-rate',type=float,default=2e-3)
    parser.add_argument('--router-phase-learning-rate',type=float,default=3.2e-3)
    parser.add_argument('--manifest-path')
    parser.add_argument('--vision-cache-path')
    parser.add_argument('--language-cache-path')
    parser.add_argument('--audit-output',help='Completed audit with bitwise recovered-cache verification')
    args=parser.parse_args()
    if args.phase=='audit' and not args.package:parser.error('audit requires --package')
    if args.phase=='train' and not args.arm:parser.error('train requires --arm')
    (audit if args.phase=='audit' else train_arm)(args)


if __name__=='__main__':
    main()
