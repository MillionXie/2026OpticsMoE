"""Reviewed configuration and an explicit launch hold for the pending Vision-only."""
import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = Path(__file__).with_name('routing_policy.json')


def load_policy():
    policy = json.loads(POLICY_PATH.read_text(encoding='utf-8'))
    from .layout import A_EXPERTS,B_EXPERTS,POSITIONS
    if (tuple(policy['expert_groups']['A']),tuple(policy['expert_groups']['B']))!=(A_EXPERTS,B_EXPERTS):
        raise ValueError('Training policy differs from physical expert ownership')
    if policy['modalities']!=['vision'] or tuple(policy['spatial_order'])!=POSITIONS:
        raise ValueError('This candidate requires the reviewed vision-only spatial order')
    if not .5 < policy['loss']['target_group_probability'] < 1:
        raise ValueError('Target group probability must be between .5 and 1')
    if policy['loss']['clean_route_kd_weight'] != 0:
        raise ValueError('Vision-only removes source-router KD; a change requires a new review')
    return policy


def policy_sha256(policy=None):
    return hashlib.sha256(json.dumps(policy or load_policy(), sort_keys=True, separators=(',',':')).encode()).hexdigest()


def output_root(smoke=False):
    name = load_policy()['output_relative_directory'] + ('_smoke' if smoke else '')
    return PROJECT_ROOT / name


def require_authorization(root=PROJECT_ROOT):
    root = Path(root)
    path = root / 'RUN_AUTHORIZATION.json'
    authorization = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    if authorization.get('approved') is not True:
        raise RuntimeError('Vision-only is awaiting explicit user confirmation. Training, GPU smoke runs and final evaluation are not authorized.')
    manifest_path = root / 'REVIEW_MANIFEST.json'
    if not manifest_path.exists():
        raise RuntimeError('Reviewed code manifest is missing')
    raw = manifest_path.read_bytes()
    if authorization.get('review_manifest_sha256') != hashlib.sha256(raw).hexdigest():
        raise RuntimeError('Authorization does not match the reviewed code manifest')
    for row in json.loads(raw)['files']:
        path = (root / row['path']).resolve()
        if not path.is_relative_to(root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise RuntimeError('Reviewed source has changed: ' + row['path'])


def require_gpu_preflight(root=PROJECT_ROOT):
    root=Path(root);smoke=root/(load_policy()['output_relative_directory']+'_smoke')
    marker=smoke/'contract_tests.json'
    result=json.loads(marker.read_text()) if marker.exists() else {}
    manifest=root/'REVIEW_MANIFEST.json'
    reviewed=hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.exists() else None
    if reviewed is None or result.get('status')!='passed' or result.get('policy_sha256')!=policy_sha256() or result.get('review_manifest_sha256')!=reviewed:
        raise RuntimeError('The reviewed Vision-only needs approved GPU contract checks and smoke training before a formal run')
    for stage in ('moe_A','moe_reserved_B','moe_all_B','d2nn_A','d2nn_d2nn_B'):
        path=smoke/stage/'final.json'
        result=json.loads(path.read_text()) if path.exists() else {}
        if result.get('status')!='smoke_complete' or result.get('policy_sha256')!=policy_sha256():
            raise RuntimeError('Vision-only smoke stage is incomplete: '+stage)


class RoutingProtocolFailure(RuntimeError):
    """A recorded experimental failure, distinct from an implementation error."""
