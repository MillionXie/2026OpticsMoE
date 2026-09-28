"""Matched TRAIN-only stress screen for one ABO inference checkpoint.

The 800 original TEST queries are never evaluated as a selection set. A separate
metadata checkpoint fixes the identical optical noise proxy across candidates.
"""
import argparse
import copy
from pathlib import Path

import torch
from transformers import AutoProcessor

from LightGenV2.tasks.t07_abo_image_retrieval.standalone.io import sha256, verify_assets, write_json
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.model import OpticalRetrieval, alpha_value
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_adapt import assessment, fitting_groups
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_screen import load_screen
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.robust_holdout import split_train


def main():
    parser = argparse.ArgumentParser()
    for name in ('data', 'manifest', 'assets', 'checkpoint', 'noise_metadata_checkpoint', 'output'):
        parser.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--noise-metadata-sha256', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Screen output exists; refusing duplicate')
    if sha256(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError('Candidate checkpoint SHA mismatch')
    if sha256(args.noise_metadata_checkpoint) != args.noise_metadata_sha256:
        raise ValueError('Noise metadata SHA mismatch')
    verify_assets(args.assets)
    protocol, groups = load_screen(args.manifest, args.data)
    reduced, holdout, audit = split_train(groups)
    fit = fitting_groups(protocol, reduced, multi_view=True)
    args._selection_groups = holdout
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    noise = torch.load(args.noise_metadata_checkpoint, map_location='cpu', weights_only=True)
    device = torch.device(args.device)
    torch.set_num_threads(4)
    model = OpticalRetrieval(copy.deepcopy(noise['metadata'])).to(device)
    model.load_state_dict(payload['state_dict'], strict=True)
    model.eval().requires_grad_(False)
    processor = AutoProcessor.from_pretrained(str(args.assets / 'processor'), local_files_only=True)
    measured = assessment(model, processor, groups, args, device, fit=fit, selection=True)
    alphas = {}
    for name, parameter in model.named_parameters():
        if name.endswith(('block1_optical_fusion_logit', 'block2_optical_fusion_logit')):
            modality = model.vision if name.startswith('vision.') else model.language
            alphas[name] = float(alpha_value(parameter, modality.alpha_bounds))
    report = dict(status='complete', checkpoint_sha256=args.checkpoint_sha256,
                  noise_metadata_sha256=args.noise_metadata_sha256,
                  clean_train_holdout_r1=float(measured['test']['r_at_1']),
                  noisy_train_holdout_r1=float(measured['validation_noisy']['r_at_1']),
                  alpha=alphas, router=measured['router'],
                  holdout_audit=audit,
                  caveat='400 TRAIN continuation holdout identities were seen by the original checkpoint; not independent')
    args.output.mkdir(parents=True)
    write_json(args.output / 'report.json', report)
    print(report, flush=True)


if __name__ == '__main__':
    main()
