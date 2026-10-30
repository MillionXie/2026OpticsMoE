"""Optional TRAIN-only decoded-image supervision of a six-step compact rollout.

The same frozen original VAE decoder converts saved electronic30 terminal
latents into teacher images. It also decodes the student during training, so
multiscale RGB and edge losses reach the existing readout/UNet through all steps.
No VAE encoder, reference-image bypass, new network or deployed parameter is
introduced. Existing latent-only runs and samplers are left unchanged.
"""
import argparse
import json
from pathlib import Path
import sys


def image_agreement(actual,wanted):
    import torch
    from torch.nn import functional as F
    if actual.shape!=wanted.shape or actual.ndim!=4 or actual.shape[1]!=3:
        raise ValueError('Matched decoded B3HW images required')
    wanted=wanted.detach().float();actual=actual.float()
    pixel=torch.stack([(F.avg_pool2d(actual,k)-F.avg_pool2d(wanted,k)).abs().mean()for k in (1,2,4)]).mean()
    edge=torch.stack([((actual[...,1:,:]-actual[...,:-1,:])-(wanted[...,1:,:]-wanted[...,:-1,:])).abs().mean(),
                      ((actual[...,:,1:]-actual[...,:,:-1])-(wanted[...,:,1:]-wanted[...,:,:-1])).abs().mean()]).mean()
    return pixel,edge


def decoded_agreement(decoder,student,teacher,scaling_factor,gradient=False):
    import torch
    from torch.utils.checkpoint import checkpoint
    if not scaling_factor>0 or any(p.requires_grad for p in decoder.parameters()):
        raise ValueError('Frozen original decoder and positive original latent scale required')
    def decode(x):return decoder.decode((x/scaling_factor).to(decoder.dtype),return_dict=False)[0].float()/2+.5
    with torch.no_grad():wanted=decode(teacher.detach())
    actual=checkpoint(decode,student,use_reentrant=False)if gradient else decode(student)
    return image_agreement(actual,wanted)


def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--decoded-pixel-weight',type=float,default=.5);p.add_argument('--decoded-edge-weight',type=float,default=.2)
    custom,remaining=p.parse_known_args()
    if not(0<custom.decoded_pixel_weight<=1 and 0<custom.decoded_edge_weight<=1):raise ValueError('Explicit weak positive TRAIN image losses required')
    from . import scene_joint_cached_rollout as base
    from .white_reference_scene_probe import digest,write
    old_state,old_load,old_objective,old_run=base.complete_state,base.load_generator,base.objective,base.run
    holder={}
    contract=dict(schema='scene_TRAIN_decoded_rollout_v1',pixel_weight=custom.decoded_pixel_weight,edge_weight=custom.decoded_edge_weight,
        decoder_is_original_frozen_VAE=True,decoder_used_for_TRAIN_loss_only=True,VAE_encoder_used=False,
        target='Original frozen E30 terminal latents from the same192 TRAIN paths',extra_deployment_parameters=0)
    def state(folder):
        value,report=old_state(folder);return dict(value,decoded_rollout_TRAIN_contract=contract),report
    def load(a,value,trainable):
        pipe,head=old_load(a,value,trainable);holder['decoder']=pipe.vae.cuda().eval().requires_grad_(False)
        holder['scale']=float(pipe.vae.config.scaling_factor);return pipe,head
    def objective(student,teacher,own,wanted,edges):
        loss,terminal,anchor=old_objective(student,teacher,own,wanted,edges)
        pixel,edge=decoded_agreement(holder['decoder'],student,teacher,holder['scale'],student.requires_grad)
        image=custom.decoded_pixel_weight*pixel+custom.decoded_edge_weight*edge
        return loss+image,terminal+image,anchor
    def run(a):
        write(a.output/'TRAIN_decoded_loss_contract.json',dict(contract,source_sha256=digest(Path(__file__)),
            parent_trainer_source_sha256=digest(Path(base.__file__)),sampling_grid_changed=False,DEV_or_TEST_supervision=False))
        old_run(a)
        report=json.loads((a.output/'report.json').read_text());report.update(decoded_image_TRAIN_supervision=True,
            decoded_TRAIN_contract=contract,wrapper_source_sha256=digest(Path(__file__)),
            best_selection='Fixed reused TRAIN composite latent plus decoded-image score; not DEV/TEST or independent quality')
        for prefix in ['initial_fixed_TRAIN','best_TRAIN']:
            key=prefix+'_terminal_loss'
            if key in report:report[prefix+'_composite_score']=report.pop(key)
        write(a.output/'report.json',report)
        path=a.output/'history.json'
        if path.exists():
            rows=json.loads(path.read_text())
            for row in rows:row['fixed_TRAIN_composite_score']=row.pop('fixed_TRAIN_terminal_loss')
            write(path,rows)
    base.complete_state=state;base.load_generator=load;base.objective=objective;base.run=run;sys.argv=[sys.argv[0],*remaining]
    try:base.main()
    finally:base.complete_state=old_state;base.load_generator=old_load;base.objective=old_objective;base.run=old_run


if __name__=='__main__':main()
