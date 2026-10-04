"""Explicit, fixed-weight T08 reverse retrieval entry; no hardware operation.

The exact historical reverse implementation is reverse_runtime.py. This entry
only resolves external assets and prevents direction/initialization mistakes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

BODY_SHA256 = 'cc977b83286a8e90398ebd30064428886bc3c06f1eb0557e470060f7ff5c1cae'
PROFILE_SHA256 = '59435a148c02a60d84792408436e08e3d5dbef7716cfd8eefa729918dbc22c1a'
PROFILE = Path(__file__).resolve().parent/'configs/text_to_image_10cm_adopted_eval.yaml'
PATH_KEYS = (('dataset', 'dataset_root'), ('qwen', 'cache_dir'),
             ('hybrid', 'initial_electronic_checkpoint'), ('router_experiment', 'source_checkpoint'),
             ('abo_image_text', 'resume_checkpoint'), ('abo_image_text', 'cache_file'))


def configuration(profile, *, model, data_root, checkpoint, teacher_cache, run_dir):
    """Resolve locations only; never change geometry, fusion or token processing."""
    profile = Path(profile).resolve()
    if hashlib.sha256(profile.read_bytes().replace(b'\r\n', b'\n')).hexdigest() != PROFILE_SHA256:
        raise ValueError('Profile differs from the audited fixed-evaluation contract')
    raw = yaml.safe_load(profile.read_text(encoding='utf-8'))
    if not isinstance(raw, dict) or 'base_config' in raw:
        raise ValueError('Use the flattened audited fixed-evaluation profile')
    abo = raw.get('abo_image_text', {})
    if abo.get('retrieval_direction') != 'text_to_image' or abo.get('selection_direction') != 'text_to_image':
        raise ValueError('This entry accepts text-to-image, not image-to-text')
    if raw.get('language_optical', {}).get('distance_m') != .10:
        raise ValueError('Only the adopted 10cm profile is verified here')
    if any(abo.get(key, False) for key in ('allow_alpha_range_transition', 'allow_propagation_distance_transition', 'allow_electronic_compaction_transition')):
        raise ValueError('Fixed inference cannot reset or transplant trained tensors')
    if abo.get('resume_checkpoint_sha256') != BODY_SHA256:
        raise ValueError('Profile is not bound to the adopted checkpoint')
    for section, key in PATH_KEYS:
        value = raw.get(section, {}).get(key)
        if value is not None:
            location = Path(str(value)).expanduser()
            raw[section][key] = str((profile.parent/location).resolve() if not location.is_absolute() else location.resolve())
    # These inherited initialization references are not used in fixed resume,
    # but retain their original absolute meaning in the saved run config.
    warmstart = raw.get('warmstart', {})
    for key in ('electronic_checkpoint', 'optical_checkpoint', 'stage_a_checkpoint'):
        value = warmstart.get(key)
        if value is not None:
            location = Path(str(value)).expanduser()
            warmstart[key] = str((profile.parent/location).resolve() if not location.is_absolute() else location.resolve())
    raw['qwen']['model_id'] = str(Path(model).expanduser().resolve())
    raw['qwen']['local_files_only'] = True
    raw['dataset']['dataset_root'] = str(Path(data_root).expanduser().resolve())
    raw['abo_image_text']['resume_checkpoint'] = str(Path(checkpoint).expanduser().resolve())
    raw['abo_image_text']['cache_file'] = str(Path(teacher_cache).expanduser().resolve())
    raw['output_dir'] = str(Path(run_dir).expanduser().resolve())
    return raw


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--profile', type=Path, default=PROFILE)
    result.add_argument('--model', type=Path, required=True, help='Existing local frozen Qwen directory')
    result.add_argument('--data-root', type=Path, required=True)
    result.add_argument('--checkpoint', type=Path, required=True)
    result.add_argument('--teacher-cache', type=Path, required=True)
    result.add_argument('--run-dir', type=Path, required=True, help='Must not already exist')
    result.add_argument('--device', default='cpu')
    result.add_argument('--seed', type=int, default=42)
    result.add_argument('--inspect', action='store_true', help='Show resolved profile without loading models or writing a run')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    raw = configuration(args.profile, model=args.model, data_root=args.data_root,
                        checkpoint=args.checkpoint, teacher_cache=args.teacher_cache, run_dir=args.run_dir)
    if args.inspect:
        print(json.dumps(raw, ensure_ascii=False, indent=2))
        return 0
    if not args.model.is_dir():
        raise FileNotFoundError('Local Qwen directory is required; no download is started')
    for name in ('train.csv', 'test.csv', 'titles.csv', 'manifest.csv'):
        if not (args.data_root/name).is_file():
            raise FileNotFoundError(args.data_root/name)
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if digest != BODY_SHA256:
        raise RuntimeError('Adopted body SHA mismatch; no run started')
    # Parse/import before creating a destination. Model construction is left to
    # the exact runtime and its existing strict metadata checks.
    from . import reverse_runtime
    run_dir = args.run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    config = run_dir/'input_config.yaml'
    config.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding='utf-8')
    options = argparse.Namespace(config=str(config), run_dir=str(run_dir), device=args.device, seed=args.seed,
                                 epochs=None, steps_per_epoch=None, eval_every=None, force_teacher_cache=False,
                                 evaluate_only=True, resume_checkpoint=str(args.checkpoint.resolve()),
                                 expected_resume_sha256=BODY_SHA256, data_root=str(args.data_root.resolve()),
                                 teacher_cache=str(args.teacher_cache.resolve()))
    report = reverse_runtime.run(options)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
