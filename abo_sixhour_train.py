"""TRAIN-holdout robust continuation; no additional optical/electronic branch."""
import os,copy
from LightGenV2.tasks.t07_abo_image_retrieval.standalone import retrieval_refine as refine,robust_training as robust
mode=os.environ['ABO_RECOVERY_MODE']
profile=dict(refine.PROFILES['physical_bounded30_mlp448_paired'],holdout_selection=True,
    noise_probability=.5,pixel_shift=1,phase_dropout=.05,phase_dropout_noisy_only=True,
    paired_consistency=1.,teacher_weight=.5,weight_decay=.03,alpha_lr_multiplier=10.,
    phase_lr_multiplier=.1,router_lr_multiplier=.05,head_lr_multiplier=2.)
if mode=='electronic':profile.update(noise_probability=.65,phase_lr_multiplier=.02,router_lr_multiplier=.02)
if mode=='shift2':profile.update(pixel_shift=2,noise_probability=.5,paired_consistency=2.)
refine.PROFILES['sixhour_'+mode]=profile
original=robust.prepare
def prepare(payload,p):
    result=original(payload,p)
    n=result['metadata']['optical_training_noise'];n.update(ccd_mean=.025,ccd_std=.05,ccd_low=-.08,ccd_high=.18,phase_bypass=.04,router_logit_std=.08)
    result['metadata']['sixhour_recovery']=dict(mode=mode,noise='uncalibrated CCD proxy',electronic_dropout=.05,alpha_floor=.3)
    return result
robust.prepare=prepare
attach=robust.attach
def attach_all(model,p):
    attach(model,p)
    import torch
    for module in (model.vision,model.language):
        for block in module.blocks:
            def hook(m,args,value,owner=module):
                return torch.nn.functional.dropout(value,.05,training=True) if m.training and owner.optics.noise_enabled else value
            block.register_forward_hook(hook)
robust.attach=attach_all
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_adapt import main
if __name__=='__main__':main()
