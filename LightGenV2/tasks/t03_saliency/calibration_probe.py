"""Train-only two-scalar readout calibration; fixed optical model diagnostic.

No image-dependent branch, teacher, attention, or per-test-image fitting.
This is a separate candidate, never a silent change to existing evaluations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn


class SpatialLogitCalibration(nn.Module):
    def __init__(self, size: int = 224):
        super().__init__()
        self.raw_scale = nn.Parameter(torch.zeros(()))
        self.raw_center = nn.Parameter(torch.zeros(()))
        axis = (torch.arange(size, dtype=torch.float32) + .5) / size * 2 - 1
        radial = -(axis[:, None].square() + axis[None, :].square())
        self.register_buffer('radial', (radial - radial.mean())[None, None])

    def coefficients(self):
        return torch.exp(math.log(2) * self.raw_scale.tanh()), 2 * self.raw_center.tanh()

    def forward(self, logits):
        if logits.ndim != 4 or logits.shape[1:] != self.radial.shape[1:]:
            raise ValueError('Calibration requires [B,1,H,W] matching the fixed readout grid')
        scale, center = self.coefficients()
        return scale * logits.float() + center * self.radial


def require_train_ids(ids):
    if not ids or len(set(ids)) != len(ids) or any(not x.startswith('train/') for x in ids):
        raise ValueError('Calibration fitting requires unique training identities only')


def run(args):
    from .run import _seed, _git_value
    from .settings import load_settings, save_resolved_config
    from .modeling import load_vision_backbone, build_student, initialize_student, sha256_file
    from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency import training as legacy
    from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency.datasets import prepare_salicon
    from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency.objectives import (
        correlation_coefficient, density_from_logits, SaliencyAccumulator)
    from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency.io_utils import environment_report

    if not 1 <= args.epochs <= 100 or args.batch_size < 1:
        raise ValueError('Expected 1..100 epochs and a positive inference batch size')
    out = Path(args.output).resolve()
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite probe: {out}')
    out.mkdir(parents=True)
    s = load_settings(Path(args.config))
    s.output_dir = out
    s.augmentation_enabled = False
    s.inference_batch_size = args.batch_size
    s.num_workers = 2
    _seed(42)
    def write(name, value):
        (out/name).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    write('environment.json', environment_report())
    bundle = prepare_salicon(s, persist=True)
    ids = [x.sample_id for x in bundle.train_records]
    require_train_ids(ids)
    if len(ids) != 10000 or len(bundle.validation_records) != 5000:
        raise ValueError('Formal probe requires all 10000 train and 5000 public-test images')
    loaded = load_vision_backbone(s, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model = build_student(loaded, s)
    initialization = initialize_student(model, s)
    model.requires_grad_(False).eval()
    model.core.set_phase_dropout_active(False)
    save_resolved_config(s)
    write('initialization_report.json', initialization)
    write('protocol.json', {
        'git_commit': _git_value('rev-parse', 'HEAD'), 'command': vars(args),
        'source_sha256': sha256_file(s.initialization_checkpoint),
        'train_ids_sha256': hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
        'inference_parameters_added': 2, 'optical_model_frozen': True,
        'train_cache': 'float32 CPU RAM only; discarded after probe',
        'selection': 'best training CC for two scalars; one final paired full public-test evaluation',
        'public_test_already_used_by_base_model_and_project': True,
        'test_optics': 'unchanged standard clean eval mode',
        'bounds': {'logit_scale': [.5,2.], 'radial_bias': [-2.,2.]}})
    train_loader, test_loader = legacy.build_loaders(bundle, s, training=False)
    cached_logits, cached_targets, cache_ids = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(train_loader, 1):
            inp = legacy.preprocess_vision(loaded.processor, batch['images'], loaded.device)
            with legacy._autocast(s, loaded.device):
                logits, _, _ = model(inp['pixel_values'], inp['image_grid_thw'])
            cached_logits.append(logits.float().cpu())
            cached_targets.append(batch['density'].float().cpu())
            cache_ids.extend(batch['sample_ids'])
            if i % 100 == 0:
                print(f'CACHE train batches={i}/{len(train_loader)}', flush=True)
    if cache_ids != ids:
        raise ValueError('Training loader ordering differs from the source identity manifest')
    logits = torch.cat(cached_logits); targets = torch.cat(cached_targets)
    del cached_logits, cached_targets
    if not torch.isfinite(logits).all() or not torch.isfinite(targets).all():
        raise ValueError('Nonfinite training cache')
    calibration = SpatialLogitCalibration(s.image_size).to(loaded.device)
    opt = torch.optim.Adam(calibration.parameters(), lr=.03)
    rng = torch.Generator().manual_seed(42)
    @torch.no_grad()
    def train_cc():
        total = 0.
        for a in range(0, len(ids), 64):
            x = logits[a:a+64].to(loaded.device); y = targets[a:a+64].to(loaded.device)
            total += len(x)*float(correlation_coefficient(density_from_logits(calibration(x)), y))
        return total/len(ids)
    best = train_cc(); history = [{'epoch': 0, 'train_cc': best, 'scale': 1., 'center': 0.}]
    best_state = {k:v.detach().cpu().clone() for k,v in calibration.state_dict().items()}
    for epoch in range(1, args.epochs+1):
        order = torch.randperm(len(ids), generator=rng)
        for index in order.split(64):
            x=logits[index].to(loaded.device); y=targets[index].to(loaded.device)
            opt.zero_grad(set_to_none=True)
            scale, center = calibration.coefficients()
            loss = (1-correlation_coefficient(density_from_logits(calibration(x)), y)
                    + .001*(scale.log().square()+center.square()))
            loss.backward(); opt.step()
        score = train_cc(); scale, center = calibration.coefficients()
        row = {'epoch':epoch, 'train_cc':score, 'scale':float(scale.detach()), 'center':float(center.detach())}
        history.append(row); print('CALIBRATION',json.dumps(row),flush=True)
        if score > best:
            best=score; best_state={k:v.detach().cpu().clone() for k,v in calibration.state_dict().items()}
        write('training_history.json',history)
    torch.save({'calibration':calibration.state_dict(), 'base':initialization,
                'format':'t03_two_scalar_calibration_v1'},out/'last_checkpoint.pt')
    calibration.load_state_dict(best_state)
    torch.save({'calibration':best_state, 'base':initialization,
                'format':'t03_two_scalar_calibration_v1'},out/'best_checkpoint.pt')
    del logits,targets
    original, corrected = SaliencyAccumulator(), SaliencyAccumulator()
    paired=[]
    with torch.no_grad():
        for i,batch in enumerate(test_loader,1):
            inp=legacy.preprocess_vision(loaded.processor,batch['images'],loaded.device)
            with legacy._autocast(s,loaded.device):
                logits,_,_=model(inp['pixel_values'],inp['image_grid_thw'])
            y=batch['density'].to(loaded.device); f=batch['fixation'].to(loaded.device)
            refined=calibration(logits)
            original.update(logits,y,f); corrected.update(refined,y,f)
            for n,key in enumerate(batch['sample_ids']):
                paired.append({'sample_id':key,
                    'base_cc':float(correlation_coefficient(density_from_logits(logits[n:n+1]),y[n:n+1])),
                    'calibrated_cc':float(correlation_coefficient(density_from_logits(refined[n:n+1]),y[n:n+1]))})
            if i%100==0: print(f'TEST {i}/{len(test_loader)}',flush=True)
    with (out/'per_image_cc.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(paired)
    scale,center=calibration.coefficients()
    report={'base':original.compute(),'calibrated':corrected.compute(),
            'scale':float(scale.detach()),'radial_bias':float(center.detach()),
            'calibration_parameters':2,'base_model_retrained':False,
            'best_checkpoint_sha256':sha256_file(out/'best_checkpoint.pt'),
            'deployment_status':'diagnostic wrapper; base checkpoint also required; not a replacement core checkpoint'}
    write('calibration_report.json',report);print(json.dumps(report),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--epochs',type=int,default=20);p.add_argument('--batch-size',type=int,default=16)
    run(p.parse_args())
