"""Evaluate a separate candidate using fixed TRAIN input-amplitude scales."""
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch
import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess

import numpy as np
import torch
from torch.nn import functional as F

from . import _paths
from .data import batch, groups, load_inputs, sha256
from .run import evaluate, save_checkpoint, write
from .spatial_locality import STAGES, audit_spatial_locality
from .evaluate_detector_noise import (detector_noise_metrics, reference_intensities)
from .analyze_phase_effect import phase_information, constant_modulation
from ..model import GlobalStage, LightGenTemporal, ModelConfig, load_lightgen


ARCHITECTURE = 't22_fixed_train_input_scale_candidate_v1'


def encoded_amplitude(stage, tokens):
    amplitude = F.relu(stage.width_to_field(tokens))
    if stage.frame_level:
        amplitude = F.relu(stage.tokens_to_field(amplitude.transpose(-2, -1))).transpose(-2, -1)
    else:
        amplitude = F.pad(amplitude, (0, 0, 0, 56 - amplitude.shape[-2]))
    return amplitude


class FixedInputScaleGlobal(GlobalStage):
    def __init__(self, config, frame_level):
        super().__init__(config, frame_level)
        self.register_buffer('input_scale', torch.ones(()))

    def fields(self, tokens):
        return encoded_amplitude(self, tokens) / self.input_scale


class FixedInputScaleModel(LightGenTemporal):
    def __init__(self, config):
        super().__init__(config)
        self.frame_global = FixedInputScaleGlobal(config, True)
        self.video_global = FixedInputScaleGlobal(config, False)

    def checkpoint(self, **metadata):
        return dict(schema=1, architecture=ARCHITECTURE, config=asdict(self.cfg),
                    model=self.state_dict(), metadata=metadata)


def make_candidate(parent):
    model = FixedInputScaleModel(parent.cfg)
    missing, unexpected = model.load_state_dict(parent.state_dict(), strict=False)
    if set(missing) != {'frame_global.input_scale', 'video_global.input_scale'} or unexpected:
        raise ValueError('Candidate and parent tensor names differ')
    return model.eval()


def load_candidate(path, device='cpu'):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload['schema'] != 1 or payload['architecture'] != ARCHITECTURE:
        raise ValueError('Expected a fixed TRAIN input-scale candidate checkpoint')
    model = FixedInputScaleModel(ModelConfig(**payload['config']))
    model.load_state_dict(payload['model'], strict=True)
    for attribute in STAGES.values():
        value = float(getattr(model, attribute).input_scale)
        if not math.isfinite(value) or value <= 0:
            raise ValueError('Input scale must be finite and positive')
    return model.to(device).eval(), payload['metadata']


@torch.inference_mode()
def calibrate_input_scales(model, payload, device, batch_fields):
    indices, valid = groups(payload, 'train')
    record = {}
    for branch, attribute in STAGES.items():
        stage = getattr(model, attribute)
        totals = []
        current_valid = None

        def collect(module, arguments):
            intensity = encoded_amplitude(module, arguments[0]).square().mean((-2, -1))
            totals.append(intensity[current_valid].double().cpu().flatten())

        handle = stage.register_forward_pre_hook(collect)
        try:
            for start in range(0, len(indices), batch_fields):
                item = batch(payload, indices[start:start+batch_fields], valid[start:start+batch_fields], device)
                current_valid = item['valid']
                model(item['vision'])
        finally:
            handle.remove()
        intensity = torch.cat(totals)
        expected = 2250 * (4 if branch == 'vision' else 1)
        if len(intensity) != expected or not bool(torch.isfinite(intensity).all()):
            raise ValueError('TRAIN input-scale calibration has invalid identities or values')
        scale = float(intensity.mean().sqrt())
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError('TRAIN input-scale calibration is dark')
        stage.input_scale.fill_(scale)
        record[branch] = dict(scale=scale, train_regions=expected,
                              raw_mean_intensity=float(intensity.mean()))
    return record


@torch.inference_mode()
def input_field_audit(model, payload, device, directory):
    indices, valid = groups(payload, 'test')
    item = batch(payload, indices[:1], valid[:1], device)
    captured = {}
    handles = []
    for branch, attribute in STAGES.items():
        def capture(module, arguments, branch=branch):
            captured[branch] = module.fields(arguments[0]).detach().cpu()
        handles.append(getattr(model, attribute).register_forward_pre_hook(capture))
    try:
        result = model(item['vision'])
    finally:
        for handle in handles:
            handle.remove()
    records = []
    summaries = {}
    arrays = {}
    for branch, route_name in (('vision','frame'), ('language','video')):
        amplitude = captured[branch]
        intensity = amplitude.square().mean((-2,-1))
        weights = result['routing'][route_name]['weights'].detach().cpu()
        if not bool(((weights != 0).sum(-1) == 2).all()):
            raise ValueError('Candidate violates Top2 routing')
        for slot in range(16):
            count = 4 if branch == 'vision' else 1
            for frame in range(count):
                value = intensity[0,slot,frame] if branch == 'vision' else intensity[0,slot]
                gates = weights[0,slot,frame] if branch == 'vision' else weights[0,slot]
                row = dict(branch=branch, sample_id=payload['ids'][int(indices[0,slot])],
                           frame=frame+1 if branch == 'vision' else '', mean_intensity=float(value))
                row.update({f'weight_E{k+1}':float(gates[k]) for k in range(4)})
                records.append(row)
        summaries[branch] = dict(minimum_mean_intensity=float(intensity.min()),
                                 maximum_mean_intensity=float(intensity.max()),
                                 average_mean_intensity=float(intensity.mean()))
        arrays[branch+'_amplitude'] = amplitude.numpy()
        arrays[branch+'_weights'] = weights.numpy()
    np.savez_compressed(directory/'input_regions.npz', **arrays)
    with (directory/'input_regions.csv').open('w',newline='',encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return summaries


@torch.inference_mode()
def phase_controls(model, payload, device, batch_fields):
    values = {}
    for branch, attribute in STAGES.items():
        stage = getattr(model, attribute)
        _, _, centers, _ = phase_information(stage)
        branch_values = {}
        for kind in ('circular_mean','zero','pi'):
            center = torch.full_like(centers, math.pi) if kind == 'pi' else centers
            modulation = constant_modulation(stage, center, zero=kind == 'zero')
            with patch.object(stage, 'modulation', lambda value=modulation: value):
                branch_values[kind] = evaluate(model,payload,device,batch_fields)['srcc']
        values[branch] = branch_values
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--paths',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--batch-fields',type=int,default=4)
    parser.add_argument('--noise-trials',type=int,default=3)
    parser.add_argument('--source-commit')
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(20261117)
    np.random.seed(20261117)
    status = args.out/'status.json'
    write(status,dict(state='loading_inputs'))
    payload = load_inputs(json.loads(args.paths.read_text('utf-8')))
    parent, metadata = load_lightgen(args.weights,args.device)
    if metadata['dataset']['manifest']['sha256'] != payload['identity']['manifest']['sha256']:
        raise ValueError('Parent checkpoint and candidate dataset identities differ')
    write(args.out/'configuration.json',dict(weights=str(args.weights),paths=str(args.paths),
          device=args.device,batch_fields=args.batch_fields,noise_trials=args.noise_trials,
          model_config=asdict(parent.cfg),candidate_architecture=ARCHITECTURE))
    candidate = make_candidate(parent.cpu()).to(args.device).eval()
    parent = parent.to(args.device).eval()
    candidate.requires_grad_(False)
    parent.requires_grad_(False)
    write(status,dict(state='calibrating_TRAIN_scales'))
    calibration = calibrate_input_scales(candidate,payload,args.device,args.batch_fields)
    for name,value in parent.state_dict().items():
        if not torch.equal(value,candidate.state_dict()[name]):
            raise ValueError('Calibration changed a parent tensor: '+name)
    source_commit = args.source_commit or subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=_paths.ROOT,text=True).strip()
    relative_source = Path(__file__).resolve().relative_to(_paths.ROOT.parents[2]).as_posix()
    tracked_source = subprocess.check_output(['git','show',source_commit+':'+relative_source],cwd=_paths.ROOT)
    if hashlib.sha256(tracked_source).hexdigest() != sha256(Path(__file__)):
        raise ValueError('Candidate source differs from its recorded commit')
    record = dict(role='separate_development_candidate', source_commit=source_commit,
                  source_sha256=sha256(Path(__file__)), parent_checkpoint_sha256=sha256(args.weights),
                  dataset=payload['identity'], calibration=calibration,
                  input_scale_definition='One fixed RMS per branch, calibrated only on TRAIN',
                  optimization_steps=0, torch_version=str(torch.__version__), python_version=platform.python_version())
    save_checkpoint(args.out/'candidate.pt',candidate.checkpoint(**record))
    reloaded,_ = load_candidate(args.out/'candidate.pt',args.device)
    for name,value in candidate.state_dict().items():
        if not torch.equal(value,reloaded.state_dict()[name]):
            raise ValueError('Candidate checkpoint reload differs: '+name)
    del reloaded
    write(status,dict(state='clean_evaluation'))
    for name,model in (('parent',parent),('candidate',candidate)):
        directory = args.out/name
        directory.mkdir()
        record[name] = evaluate(model,payload,args.device,args.batch_fields,directory,0)
    record['candidate_input_fields'] = input_field_audit(candidate,payload,args.device,args.out)
    write(args.out/'comparison.json',record)
    write(status,dict(state='phase_controls',candidate_srcc=record['candidate']['srcc']))
    record['candidate_constant_phase_srcc'] = phase_controls(candidate,payload,args.device,args.batch_fields)
    record['candidate_phase_gain'] = {branch:record['candidate']['srcc']-max(values.values())
                                    for branch,values in record['candidate_constant_phase_srcc'].items()}
    write(status,dict(state='full_leakage_audit',candidate_srcc=record['candidate']['srcc']))
    locality,arrays = audit_spatial_locality(candidate,payload,args.device,split='test',field_count=35)
    record['candidate_locality'] = locality
    np.savez_compressed(args.out/'single_video_leakage.npz',**arrays)
    write(args.out/'comparison.json',record)
    write(status,dict(state='noise_evaluation',candidate_srcc=record['candidate']['srcc']))
    parent_reference = reference_intensities(parent,payload,args.device,args.batch_fields)
    candidate_reference = reference_intensities(candidate,payload,args.device,args.batch_fields)
    record['noise_reference_intensity'] = dict(parent_TRAIN=parent_reference,candidate_TRAIN=candidate_reference)
    record['candidate_noise'] = {}
    for label,reference in (('fixed_parent_TRAIN',parent_reference),('candidate_TRAIN',candidate_reference)):
        config = dict(reference_intensity=reference,evaluation_kinds=['gaussian','poisson','combined'],
                      evaluation_fractions=[.05,.1],selection_trials=args.noise_trials,selection_seed_base=29301)
        values = detector_noise_metrics(candidate,payload,args.device,args.batch_fields,config)
        record['candidate_noise'][label] = values
        write(args.out/'comparison.json',record)
    record['checks'] = dict(
        clean_srcc_in_requested_interval=.790<=record['candidate']['srcc']<.800,
        both_global_phase_gains_positive=all(value>0 for value in record['candidate_phase_gain'].values()),
        every_audited_video_leakage_below_one_percent=all(locality[name]['maximum_video_foreign_fraction']<.01 for name in STAGES),
        detector_noise_targets=all(row['srcc']>=(.79 if row['fraction']==.05 else .78)
                                   for values in record['candidate_noise'].values() for row in values))
    write(args.out/'comparison.json',record)
    write(status,dict(state='completed',candidate_srcc=record['candidate']['srcc'],checks=record['checks']))
    print(json.dumps(dict(out=str(args.out),parent_srcc=record['parent']['srcc'],
                         candidate_srcc=record['candidate']['srcc'],checks=record['checks']),indent=2),flush=True)


if __name__ == '__main__':
    main()
