"""Flatten pinned T08 YAML inheritance and emit a reviewed apply_patch payload.

Only profile inheritance/location and fixed-evaluation controls change. No
architecture, optical, preprocessing or trained-tensor value is rewritten.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import posixpath
import subprocess

import yaml

ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'd0662a7d240340817948a3496c2cd43f4240e76d'
TASK = 'LightGenV2/tasks/t08_abo_image_text_retrieval/'
PROFILE = TASK+'configs/text_to_image_10cm_adopted_eval.yaml'
PREFIXES = ('/DATA/DATA1/guest3/2026OpticsMoE/.worktrees/t08_text_to_image_20260920/',
            '/DATA/DATA1/guest3/2026OpticsMoE/')


def merge(base, override):
    output = copy.deepcopy(base)
    for key, value in override.items():
        output[key] = merge(output[key], value) if isinstance(value, dict) and isinstance(output.get(key), dict) else copy.deepcopy(value)
    return output


def read_chain(path, seen, rows):
    if path in seen:
        raise RuntimeError('Cyclic configuration')
    seen.add(path)
    raw = subprocess.check_output(['git', '-C', str(ROOT), 'show', SOURCE+':'+path])
    rows.append(dict(path=path, sha256=hashlib.sha256(raw).hexdigest()))
    values = yaml.safe_load(raw)
    parent = values.pop('base_config', None)
    if parent is None:
        return values
    if parent.startswith('/'):
        prefix = next((p for p in PREFIXES if parent.startswith(p)), None)
        if prefix is None:
            raise RuntimeError('Unexpected external configuration parent')
        resolved = parent[len(prefix):]
    else:
        resolved = posixpath.normpath(posixpath.join(posixpath.dirname(path), parent))
    if resolved.startswith('../'):
        raise RuntimeError('Configuration escapes original repository')
    return merge(read_chain(resolved, seen, rows), values)


def build():
    rows = []
    original = read_chain(TASK+'configs/optical_text_to_image_64_10cm_compact_e0p5.yaml', set(), rows)
    profile = copy.deepcopy(original)
    changes = []
    def assign(parts, value, reason):
        node = profile
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        before = node.get(parts[-1])
        node[parts[-1]] = value
        if before != value:
            changes.append(dict(key='.'.join(parts), original=before, value=value, reason=reason))
    def relocate(node, path=()):
        for key, value in list(node.items()):
            if isinstance(value, dict):
                relocate(value, path+(key,))
            elif isinstance(value, str):
                prefix = next((p for p in PREFIXES if value.startswith(p)), None)
                if prefix:
                    assign(path+(key,), posixpath.relpath(value[len(prefix):], posixpath.dirname(PROFILE)),
                           'Relocate original repository asset reference; no asset is copied.')
    relocate(profile)
    assign(('output_dir',), '../runs/simulation/text_to_image_10cm_fixed_eval', 'CLI requires a fresh run directory, never the original run.')
    assign(('abo_image_text', 'resume_checkpoint'), '../runs/simulation/optical_text_to_image_64_10cm_compact_e0p5_seed42_20260925/best_checkpoint.pt', 'Fixed adopted body, not 15cm training initialization.')
    assign(('abo_image_text', 'resume_checkpoint_sha256'), 'cc977b83286a8e90398ebd30064428886bc3c06f1eb0557e470060f7ff5c1cae', 'Exact adopted body identity.')
    for name in ('allow_propagation_distance_transition', 'allow_electronic_compaction_transition', 'allow_alpha_range_transition'):
        assign(('abo_image_text', name), False, 'Fixed inference must not reset phase/gates or transplant training initialization.')
    if profile['abo_image_text']['retrieval_direction'] != 'text_to_image' or profile['language_optical']['distance_m'] != .10:
        raise RuntimeError('Wrong selected direction/geometry')
    payload = '# Fixed adopted T08 text-to-image body. Generated from the pinned source configuration chain.\n# Use text_to_image.py with explicit assets and a fresh run directory; not a new training profile.\n'+yaml.safe_dump(profile, sort_keys=False, allow_unicode=True)
    record = dict(source_commit=SOURCE, original_profile=TASK+'configs/optical_text_to_image_64_10cm_compact_e0p5.yaml',
                  profile=PROFILE, profile_sha256=hashlib.sha256(payload.encode()).hexdigest(),
                  inherited_configs=rows, reviewed_changes=changes, architecture_parameters_changed=False,
                  metric_reproduction_verified=False, windows_deployment_verified=False)
    return payload, record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--patch', action='store_true')
    args = parser.parse_args()
    profile, record = build()
    if args.patch:
        paths = {PROFILE: profile, 'maintenance/storage/T08_EVAL_PROFILE_IDENTITY_20261004.json': json.dumps(record, ensure_ascii=False, indent=2)+'\n'}
        for path in paths:
            if (ROOT/path).exists():
                raise FileExistsError(path)
        print('*** Begin Patch')
        for path, content in paths.items():
            print('*** Add File: '+path)
            print('\n'.join('+'+line for line in content.splitlines()))
        print('*** End Patch')
    else:
        print(json.dumps(record, ensure_ascii=False, indent=2))
