"""Refine the isolated fixed-input-scale candidate on TRAIN only."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess

import numpy as np
import torch
from torch.nn import functional as F

from . import _paths
from .router_power_candidate import load_candidate, phase_controls, input_field_audit
from .data import batch, groups, load_inputs, sha256
from .run import correlation, evaluate, save_checkpoint, write
from .spatial_locality import (audit_spatial_locality, capture_modulated_fields,
                                spatial_locality_loss)
from .phase_regularization import model_phase_penalties, smooth_phase_gradients
from .evaluate_detector_noise import (detector_noise, noise_level, detector_noise_metrics,
                                      reference_intensities)
from ..model import load_lightgen


@torch.inference_mode()
def validate_candidate(model,payload,device,review,epoch,reference):
    review = Path(review)
    review.mkdir(parents=True,exist_ok=False)
    metrics = evaluate(model,payload,device,4,review,epoch)
    locality,arrays = audit_spatial_locality(model,payload,device,split='test',field_count=35)
    np.savez_compressed(review/'single_video_leakage.npz',**arrays)
    controls = phase_controls(model,payload,device,4)
    gains = {branch:metrics['srcc']-max(values.values()) for branch,values in controls.items()}
    final = dict(metrics=metrics,phase_gain=gains,constant_phase_srcc=controls,locality=locality,
                 input_fields=input_field_audit(model,payload,device,review),
                 noise_reference_intensity={'fixed_parent_TRAIN':reference},noise={})
    final['noise_reference_intensity']['candidate_TRAIN'] = reference_intensities(model,payload,device,4)
    for label,intensity in final['noise_reference_intensity'].items():
        configuration = dict(reference_intensity=intensity,evaluation_kinds=['gaussian','poisson','combined'],
                             evaluation_fractions=[.05,.1],selection_trials=10,selection_seed_base=29301)
        final['noise'][label] = detector_noise_metrics(model,payload,device,4,configuration)
        write(review/'final_review.json',final)
    final['checks'] = dict(clean_srcc_in_requested_interval=.790<=metrics['srcc']<.800,
                           both_phase_gains_positive=min(gains.values())>0,
                           every_video_leakage_below_one_percent=all(locality[branch]['maximum_video_foreign_fraction']<.01
                                                                     for branch in ('vision','language')),
                           noise_targets=all(row['srcc']>=(.79 if row['fraction']==.05 else .78)
                                             for values in final['noise'].values() for row in values))
    write(review/'final_review.json',final)
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--teacher',type=Path,required=True)
    parser.add_argument('--paths',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--device',default='cuda')
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(913)
    np.random.seed(913)
    relative_source = Path(__file__).resolve().relative_to(_paths.ROOT.parents[2]).as_posix()
    committed = subprocess.check_output(['git','show',args.source_commit+':'+relative_source],cwd=_paths.ROOT)
    if hashlib.sha256(committed).hexdigest()!=sha256(Path(__file__)):
        raise ValueError('Refinement source differs from its recorded commit')
    payload = load_inputs(json.loads(args.paths.read_text('utf-8')))
    model,metadata = load_candidate(args.weights,args.device)
    teacher,_ = load_lightgen(args.teacher,args.device)
    model.eval().requires_grad_(False)
    teacher.eval().requires_grad_(False)
    phase,encoder,readout = [],[],[]
    for name,value in model.named_parameters():
        if name.endswith('raw_global_phase'):
            phase.append(value)
        elif name.startswith(('frame_global.width_to_field.','frame_global.tokens_to_field.',
                              'video_global.width_to_field.')):
            encoder.append(value)
        elif name.startswith('readout.'):
            readout.append(value)
        else:
            continue
        value.requires_grad_(True)
    optimizer = torch.optim.AdamW([dict(params=phase,lr=3e-4),dict(params=encoder,lr=3e-6),
                                   dict(params=readout,lr=1e-5)],weight_decay=1e-5)
    train_targets = payload['target'][torch.tensor([split=='train' for split in payload['splits']])]
    target_scale = train_targets.std(unbiased=False).to(args.device)
    comparison = json.loads((args.weights.parent/'comparison.json').read_text('utf-8'))
    reference = comparison['noise_reference_intensity']['parent_TRAIN']
    base_phases = [stage.phase().detach().clone() for stage in (model.frame_global,model.video_global)]
    scales = [stage.input_scale.clone() for stage in (model.frame_global,model.video_global)]
    protocol = dict(source_commit=args.source_commit,source_sha256=sha256(Path(__file__)),
                    candidate_parent_sha256=sha256(args.weights),teacher_sha256=sha256(args.teacher),
                    dataset=payload['identity'],epochs=args.epochs,train_videos=2250,
                    supervised_split='train',development_selection_split='original_TEST',
                    phase_learning_rate=3e-4,encoder_learning_rate=3e-6,readout_learning_rate=1e-5,
                    input_scales_frozen=True,noise_reference=reference)
    write(args.out/'protocol.json',protocol)
    write(args.out/'status.json',dict(state='training',epoch=0))
    best_score = -float('inf')
    best_epoch = None
    history = []
    step = 0
    for epoch in range(1,args.epochs+1):
        indices,valid = groups(payload,'train',seed=913+epoch)
        losses = []
        for start in range(0,len(indices),2):
            item = batch(payload,indices[start:start+2],valid[start:start+2],args.device)
            keep = item['valid']
            with torch.no_grad():
                desired = teacher(item['vision'])['prediction'][keep]
            optimizer.zero_grad(set_to_none=True)
            with capture_modulated_fields(model) as captured:
                result = model(item['vision'])
                slots = torch.randperm(16,device=args.device)[:8]
                locality = spatial_locality_loss(model,captured,keep,slots,foreign_threshold=.003)
            prediction = result['prediction'][keep]
            target = item['target'][keep]
            clean = F.mse_loss(prediction/target_scale,target/target_scale)
            distill = F.mse_loss(prediction/target_scale,desired/target_scale)
            kind = ('combined','gaussian','combined','poisson')[step%4]
            fraction = (.05,.1,.1,.1)[step%4]
            with detector_noise(model,kind,noise_level(kind,fraction),reference,51000+step,straight_through=True):
                observed = model(item['vision'])['prediction'][keep]
            noisy = F.mse_loss(observed/target_scale,target/target_scale)
            consistency = F.mse_loss(observed/target_scale,prediction.detach()/target_scale)
            roughness = model_phase_penalties(model)
            anchor = torch.stack([(1-(stage.phase()-baseline).cos()).mean()
                                  for stage,baseline in zip((model.frame_global,model.video_global),base_phases)]).mean()
            loss = (clean+.2*(1-correlation(prediction,target))+10*distill+3*noisy+consistency
                    +60*locality+.3*anchor+.01*roughness['curvature']+.002*roughness['slope'])
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite candidate refinement loss')
            loss.backward()
            smooth_phase_gradients(model,1.)
            torch.nn.utils.clip_grad_norm_(phase+encoder+readout,1.)
            optimizer.step()
            losses.append(float(loss.detach()))
            step += 1
        metadata_out = dict(protocol=protocol,parent_candidate_metadata=metadata,epoch=epoch,
                            optimization_steps=step,dataset=payload['identity'])
        save_checkpoint(args.out/'last_checkpoint.pt',model.checkpoint(**metadata_out))
        row = dict(epoch=epoch,loss=float(np.mean(losses)))
        if epoch==1 or epoch%5==0 or epoch==args.epochs:
            metrics = evaluate(model,payload,args.device,4)
            locality,arrays = audit_spatial_locality(model,payload,args.device,split='test',field_count=35)
            controls = phase_controls(model,payload,args.device,4)
            gains = {branch:metrics['srcc']-max(values.values()) for branch,values in controls.items()}
            worst = max(locality[branch]['maximum_video_foreign_fraction'] for branch in ('vision','language'))
            eligible = .790<=metrics['srcc']<.800 and worst<.01 and min(gains.values())>0
            score = metrics['srcc'] if eligible else -float('inf')
            row.update(srcc=metrics['srcc'],maximum_video_foreign_fraction=worst,
                       phase_gain=gains,eligible=eligible)
            if score>best_score:
                best_score = score
                best_epoch = epoch
                save_checkpoint(args.out/'best_checkpoint.pt',model.checkpoint(**metadata_out))
                write(args.out/'best_metrics.json',dict(metrics=metrics,locality=locality,
                      phase_controls=controls,phase_gain=gains,epoch=epoch))
                np.savez_compressed(args.out/'best_single_video_leakage.npz',**arrays)
        for stage,value in zip((model.frame_global,model.video_global),scales):
            if not torch.equal(stage.input_scale,value):
                raise ValueError('Fixed input scale changed during refinement')
        history.append(row)
        write(args.out/'history.json',history)
        write(args.out/'status.json',dict(state='training',epoch=epoch,best_epoch=best_epoch,
                                          best_srcc=best_score if best_epoch is not None else None))
        print(json.dumps(row),flush=True)
    chosen_epoch = best_epoch if best_epoch is not None else args.epochs
    chosen_path = args.out/('best_checkpoint.pt' if best_epoch is not None else 'last_checkpoint.pt')
    write(args.out/'status.json',dict(state='final_validation',best_epoch=best_epoch,chosen_epoch=chosen_epoch))
    selected,_ = load_candidate(chosen_path,args.device)
    selected.requires_grad_(False)
    final = validate_candidate(selected,payload,args.device,args.out/'review',chosen_epoch,reference)
    write(args.out/'status.json',dict(state='completed',best_epoch=best_epoch,
                                      best_srcc=best_score if best_epoch is not None else None,
                                      evaluated_epoch=chosen_epoch,checks=final['checks']))


if __name__=='__main__':
    main()
