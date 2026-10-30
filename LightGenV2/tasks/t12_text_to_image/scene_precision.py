"""Explicit UNet-only precision ablation; the original BF16 path is default.

Qwen and VAE retain BF16. Conditions keep the legacy BF16 rounding before an
optional FP16 cast. Unit noise is still drawn and hashed as BF16 for both models.
The public pipeline end-of-step callback casts ONLY the last latent to the VAE
dtype. This is a dtype bridge, not an extra sampling step or image feedback.
"""
import torch


def name(args):
    value=getattr(args,'unet_precision','bf16')
    if value not in ('bf16','fp16'):raise ValueError('Explicit bf16/fp16 UNet required')
    return value


def dtype(args):
    return torch.bfloat16 if name(args)=='bf16' else torch.float16


def condition(tensor,args):
    # Preserve the trained BF16 condition values; do not change the Qwen head.
    return tensor.bfloat16().to(dtype=dtype(args))


def final_vae_cast(pipe,step,timestep,values):
    result=dict(values)
    if step+1==pipe.num_timesteps:
        result['latents']=result['latents'].to(dtype=pipe.vae.dtype)
    return result


def generation_kwargs(args,pipe):
    if name(args)=='bf16':return {}
    if pipe.unet.dtype!=torch.float16 or pipe.vae.dtype!=torch.bfloat16:
        raise ValueError('FP16 UNet requires unchanged BF16 VAE')
    return dict(callback_on_step_end=final_vae_cast,callback_on_step_end_tensor_inputs=['latents'])


def descriptor(args):
    return dict(UNet=name(args),Qwen='bf16',head='fp32',VAE='bf16',
        condition_values_rounded_to_BF16_first=True,unit_noise_draw_dtype='bf16',
        last_latent_to_VAE_cast=name(args)=='fp16',image_feedback=False)
