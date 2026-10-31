"""Optional TRAIN-only perceptual feature loss for the existing compact rollout.

The frozen AlexNet LPIPS network is used only after the original frozen VAE
decoder during training. Inference retains the same Qwen, head, UNet and VAE;
no reference VAE encoder, feedback, new sampling step or deployed module is added.
"""
import argparse
import json
from pathlib import Path
import sys


def perceptual_agreement(metric,actual,wanted):
    from torch.nn import functional as F
    if actual.shape!=wanted.shape or actual.ndim!=4 or actual.shape[1]!=3:
        raise ValueError('Matched decoded B3HW teacher and student images required')
    if any(p.requires_grad for p in metric.parameters()):
        raise ValueError('TRAIN-only perceptual network must be frozen')
    # Fixed downsampling limits TRAIN memory. No inference image is resized here.
    actual=F.interpolate(actual.float(),size=(256,256),mode='area')*2-1
    wanted=F.interpolate(wanted.detach().float(),size=(256,256),mode='area')*2-1
    return metric(actual,wanted).mean()


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--TRAIN-perceptual-weight',type=float,default=.1)
    custom,remaining=p.parse_known_args()
    if not 0<custom.TRAIN_perceptual_weight<=.5:
        raise ValueError('Explicit weak positive TRAIN perceptual weight required')
    from . import scene_decoded_rollout_recovery as decoded
    from . import scene_joint_cached_rollout as base
    from .white_reference_scene_probe import digest,write
    old_agreement,old_load,old_state,old_run=decoded.image_agreement,base.load_generator,base.complete_state,base.run
    holder={}
    contract=dict(schema='scene_TRAIN_perceptual_rollout_v1',weight=custom.TRAIN_perceptual_weight,
        network='Official frozen LPIPS AlexNet v0.1',fixed_TRAIN_image_size=256,
        teacher='Original E30 terminal latents decoded with the original frozen VAE',
        independent_TEST_or_DEV_supervision=False,extra_deployment_parameters=0,
        reference_VAE_encoder_or_image_feedback=False)
    # The decoded wrapper multiplies its pixel component by this public weight.
    parsed=argparse.ArgumentParser(add_help=False)
    parsed.add_argument('--decoded-pixel-weight',type=float,default=.5)
    weights,_=parsed.parse_known_args(remaining)
    if weights.decoded_pixel_weight<=0:raise ValueError('Positive decoded pixel coefficient required')
    def load(a,state,trainable):
        import lpips
        pipe,head=old_load(a,state,trainable)
        holder['metric']=lpips.LPIPS(net='alex',version='0.1').cuda().eval().requires_grad_(False)
        return pipe,head
    def agreement(actual,wanted):
        pixel,edge=old_agreement(actual,wanted)
        feature=perceptual_agreement(holder['metric'],actual,wanted)
        return pixel+custom.TRAIN_perceptual_weight/weights.decoded_pixel_weight*feature,edge
    def state(folder):
        value,report=old_state(folder)
        return dict(value,perceptual_rollout_TRAIN_contract=contract),report
    def run(a):
        write(a.output/'TRAIN_perceptual_loss_contract.json',dict(contract,source_sha256=digest(Path(__file__))))
        old_run(a)
        report=json.loads((a.output/'report.json').read_text())
        report.update(TRAIN_perceptual_supervision=True,TRAIN_perceptual_contract=contract,
            full_E30_Qwen_UNet_forward_repeated=False,frozen_teacher_VAE_GPU_decode_performed=True)
        write(a.output/'report.json',report)
    decoded.image_agreement=agreement;base.load_generator=load;base.complete_state=state;base.run=run
    sys.argv=[sys.argv[0],*remaining]
    try:decoded.main()
    finally:
        decoded.image_agreement=old_agreement;base.load_generator=old_load;base.complete_state=old_state;base.run=old_run


if __name__=='__main__':main()
