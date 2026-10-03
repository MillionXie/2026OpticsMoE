"""CPU-only candidate state audit against the accepted, SHA-pinned architecture.

This checks stored tensors/configuration, NOT inference execution or accuracy.
Full-split recheck and source/runtime audits remain required for a release.
Only load trusted locally produced checkpoints; torch pickle is not sandboxed.
"""
import argparse
import hashlib
import io
import json
import subprocess
from pathlib import Path

import torch
from experiments.vision2_hybrid_dense.modeling import SaliencyDensityDecoder
from .modeling import architecture_label
from .settings import load_settings
from .training import _write_json

REFERENCE_SHA256 = '87ad4db51e3f58f9a41d6df09092439e5a09e81e93f88a3bfe2d3d008fafb29a'
PHASE_PREFIX = 'hybrid.optical_branch.core.'
PHASE_SHAPES = {
    PHASE_PREFIX+'router.raw_router_phase': (224, 224),
    **{PHASE_PREFIX+f'expert_layers.0.experts.{i}.raw_phase': (224, 224) for i in range(4)},
    PHASE_PREFIX+'global_phase.phase.raw_phase': (478, 478),
}


def read_checkpoint(path):
    data = Path(path).read_bytes()
    return torch.load(io.BytesIO(data), map_location='cpu', weights_only=False), hashlib.sha256(data).hexdigest()


def _state_spec(state):
    if not isinstance(state, dict) or any(not torch.is_tensor(v) for v in state.values()):
        raise ValueError('Expected a tensor state dictionary')
    if any(not torch.isfinite(v).all() for v in state.values()):
        raise ValueError('Nonfinite checkpoint tensor')
    return {k: (tuple(v.shape), str(v.dtype)) for k, v in state.items()}


def audit_payload(candidate, reference, settings, qwen=None):
    exact = {
        'lightgen_model_variant': 'optical_router_scale_matched_moe',
        'router_backend': 'optical', 'top_k': 2, 'active_size': 478,
        'expert_size': 224, 'electronic_width': 192, 'ccd_normalization': 'mean_only',
        # Config 0 means the unmodified depthwise path (384 runtime groups).
        'electronic_ffn_hidden_width': 384, 'electronic_ffn_groups': 0,
        'electronic_ffn_spatial_dilation': 1, 'electronic_spatial_kernel_size': 3,
        'electronic_grn': False, 'electronic_global_rank': 0,
        'router_phase_coordinates': 'sigmoid', 'phase_parameterization': 'sigmoid',
        'fusion_alpha_min': .4, 'fusion_alpha_max': 1.,
        'pixel_pitch_um': 17., 'global_to_detector_distance_m': .1,
        'language_optical_zero_order_enabled': True,
        'language_optical_phase_zero_order_intensity_min': .2,
        'language_optical_phase_zero_order_intensity_max': .3,
    }
    for name, value in exact.items():
        if getattr(settings, name) != value:
            raise ValueError(f'Candidate config changes accepted contract: {name}')
    label = architecture_label(settings)
    if candidate.get('architecture') != label or reference.get('architecture') != label:
        raise ValueError('Architecture label differs from accepted model/config')
    for name in ('core', 'saliency_head'):
        if _state_spec(candidate[name]) != _state_spec(reference[name]):
            raise ValueError(f'Candidate changes accepted state keys/shapes/dtypes: {name}')
    expected_head = SaliencyDensityDecoder(192, 224)
    if _state_spec(candidate['saliency_head']) != _state_spec(expected_head.state_dict()):
        raise ValueError('Not the current 85412-parameter decoder')
    core = candidate['core']
    actual_phases = {k: tuple(v.shape) for k, v in core.items()
                     if k.endswith(('raw_phase', 'raw_router_phase'))}
    if actual_phases != PHASE_SHAPES:
        raise ValueError('Optical phase layout changed')
    contract = {'minimum': .4, 'maximum': 1.}
    if candidate.get('fusion_contract') != contract:
        raise ValueError('Missing or changed checkpoint alpha contract')
    alphas = []
    for index in (1, 2):
        raw = core[f'hybrid.block{index}_optical_fusion_logit']
        if raw.numel() != 1:
            raise ValueError('Alpha must be scalar')
        alpha = float(.4+.6*torch.sigmoid(raw.double()))
        if not .4 <= alpha <= 1:
            raise ValueError('Alpha out of bounds')
        alphas.append(alpha)
    phase_change = {}
    for key in PHASE_SHAPES:
        phase = 2*torch.pi*core[key].double().sigmoid()
        previous = 2*torch.pi*reference['core'][key].double().sigmoid()
        delta = torch.atan2(torch.sin(phase-previous), torch.cos(phase-previous))
        phase_change[key] = float(delta.square().mean().sqrt())
    same_qwen_head = None
    if qwen is not None:
        if qwen.get('architecture') != 'frozen_qwen24_adapter192_identical_progressive_decoder_v1':
            raise ValueError('Expected explicit identical-head frozen Qwen baseline')
        decoder = {k.removeprefix('decoder.'): v for k,v in qwen['head'].items() if k.startswith('decoder.')}
        if _state_spec(decoder) != _state_spec(candidate['saliency_head']):
            raise ValueError('Qwen/student decoder specifications differ')
        same_qwen_head = True
    return {
        'state_checks_passed': True, 'architecture': label, 'alpha': alphas,
        'decoder_parameters': sum(p.numel() for p in expected_head.parameters()),
        'effective_ffn_groups': 384,
        'same_qwen_decoder_specification': same_qwen_head,
        'optical_parameters_including_router': sum(core[k].numel() for k in PHASE_SHAPES),
        'phase_rms_change_rad_vs_reference': phase_change,
        'weight_kind': candidate.get('weight_kind', 'unspecified'),
        'checkpoint_epoch': candidate.get('epoch'), 'config_contract': exact,
        'runtime_and_full_split_recheck_still_required': True,
        'limits': 'Does not prove prediction graph, GPU usage, optimizer scope, generalization, hardware performance or CC>=0.87. Same tensor shapes alone do not prove identical computation.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--qwen-checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new audit output; do not overwrite previous evidence')
    reference, digest = read_checkpoint(args.reference)
    if digest != REFERENCE_SHA256:
        raise ValueError('Accepted reference SHA256 mismatch')
    candidate, candidate_digest = read_checkpoint(args.checkpoint)
    qwen, qwen_digest = read_checkpoint(args.qwen_checkpoint) if args.qwen_checkpoint else (None, None)
    report = audit_payload(candidate, reference, load_settings(args.config), qwen)
    report.update(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=candidate_digest,
                  reference_sha256=digest, same_file_content_as_reference=candidate_digest == digest,
                  qwen_checkpoint_sha256=qwen_digest, config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest())
    report.update(audit_git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  torch_version=torch.__version__)
    _write_json(args.output, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
