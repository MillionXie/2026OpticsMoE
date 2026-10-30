"""Optional TRAIN forward matching the official task-LoRA BF16 safe merge.

Only existing LoRA parameters receive gradients through the BF16 casts. The
published renderer still performs the ordinary official merge/unload; no extra
deployment layer or parameter is introduced. This is a numerical-domain training
candidate, not a claim of quality or speed. The default on-policy trainer and all
old checkpoints remain unchanged.
"""
import types
import torch
import torch.nn.functional as F


def fused_linear_forward(self,x,*args,**kwargs):
    if args or kwargs:raise ValueError('Only the ordinary single-adapter linear call is supported')
    if self.disable_adapters or self.merged:raise ValueError('Training requires one unmerged active task adapter')
    if list(self.active_adapters)!=['scene_pair']:raise ValueError('Unexpected task adapters')
    base=self.get_base_layer();delta=self.get_delta_weight('scene_pair')
    # PEFT Linear.merge(safe_merge=True): round delta to the base dtype before
    # adding it to the base. Both casts retain the standard autograd derivative.
    effective=base.weight+delta.to(base.weight.dtype)
    return F.linear(x,effective,base.bias)


def install(unet):
    from peft.tuners.lora.layer import Linear
    before={n:(id(p),tuple(p.shape),p.requires_grad) for n,p in unet.named_parameters()};count=0
    for module in unet.modules():
        if not isinstance(module,Linear):continue
        if set(module.lora_A)!={'scene_pair'} or set(module.lora_B)!={'scene_pair'}:raise ValueError('Only the existing task adapter is allowed')
        if module.lora_variant or module.lora_bias['scene_pair'] or module.fan_in_fan_out:raise ValueError('Unsupported LoRA variant')
        if module.get_base_layer().weight.dtype!=torch.bfloat16:raise ValueError('Explicit BF16 base required')
        if module.lora_dropout['scene_pair'].p!=0:raise ValueError('No stochastic dropout in the deployment-matched forward')
        if not module.lora_A['scene_pair'].weight.requires_grad or not module.lora_B['scene_pair'].weight.requires_grad:raise ValueError('Existing LoRA must be trainable')
        if getattr(module,'_scene_fusion_aware',False):raise ValueError('Refuse repeated installation')
        module.forward=types.MethodType(fused_linear_forward,module);module._scene_fusion_aware=True;count+=1
    if not count:raise ValueError('No existing LoRA linear layers found')
    assert before=={n:(id(p),tuple(p.shape),p.requires_grad) for n,p in unet.named_parameters()}
    return dict(existing_linear_layers=count,added_parameters=0,base_dtype='bfloat16',
        training_forward='PEFT safe merge delta BF16 cast then base addition and F.linear',
        gradient='autograd through both BF16 casts; not a finite-difference claim',
        inference='ordinary renderer --fuse-task-lora; new full quality render required')


def main():
    from . import scene_onpolicy_tail as trainer
    original=trainer.load_generator
    def loader(a,state,trainable=False):
        pipe,head=original(a,state,trainable)
        if trainable:
            contract=install(pipe.unet);state['fusion_aware_training']=contract
            trainer.write(a.output/'fusion_aware_training.json',contract)
        return pipe,head
    trainer.load_generator=loader
    trainer.main()


if __name__=='__main__':main()
