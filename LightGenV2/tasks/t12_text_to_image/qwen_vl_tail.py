"""Channel/depth-pruned SD-Turbo UNet, initialized from the trained teacher.

Keep Qwen-first conditioning and the native VAE/scheduler. LoRA is folded into
copied weights; skip-concatenation and GEGLU halves are sliced separately.
"""
import copy
import re

import torch
from torch import nn

from .qwen_vl_spatial import SpatialInputConv


def merged_state(unet):
    result = {}
    for key, tensor in unet.state_dict().items():
        if '.lora_' in key:
            continue
        result[key.replace('.base_layer.', '.')] = tensor.detach()
    for name, module in unet.named_modules():
        if hasattr(module, 'get_base_layer') and hasattr(module, 'get_delta_weight'):
            weight = module.get_base_layer().weight.detach().float().clone()
            for adapter in module.active_adapters:
                if adapter in module.lora_A:
                    weight += module.get_delta_weight(adapter).detach().float()
            result[name+'.weight'] = weight
    return result


def resize_tensor(source, shape, key, *, concat=None):
    if len(source.shape) != len(shape) or any(a < b for a, b in zip(source.shape, shape)):
        raise ValueError('Cannot initialize student tensor '+key)
    indices = [torch.arange(n, device=source.device) for n in shape]
    if '.ff.net.0.proj.' in key:
        old, new = source.shape[0]//2, shape[0]//2
        indices[0] = torch.cat([torch.arange(new, device=source.device),
                                old+torch.arange(new, device=source.device)])
    if concat is not None:
        old_previous, new_previous = concat
        axis = 1 if key.endswith('conv1.weight') or key.endswith('conv_shortcut.weight') else 0
        n = shape[axis]
        indices[axis] = torch.cat([torch.arange(new_previous, device=source.device),
                                  old_previous+torch.arange(n-new_previous, device=source.device)])
    value = source
    for axis, index in enumerate(indices):
        value = value.index_select(axis, index)
    return value.contiguous()


def build_tail_student(teacher_unet, channels=(256, 512, 1024, 1024), heads=(4, 8, 16, 16)):
    from diffusers import UNet2DConditionModel
    cfg = dict(teacher_unet.config)
    for key in list(cfg):
        if key.startswith('_'):
            cfg.pop(key)
    old_channels = tuple(cfg['block_out_channels'])
    time_dim = teacher_unet.time_embedding.linear_2.out_features
    cfg.update(block_out_channels=tuple(channels), attention_head_dim=tuple(heads),
               layers_per_block=1, in_channels=4, time_embedding_dim=time_dim)
    student = UNet2DConditionModel.from_config(cfg)
    student.time_proj = copy.deepcopy(teacher_unet.time_proj).cpu()
    student.time_embedding = copy.deepcopy(teacher_unet.time_embedding).cpu().float()
    student.conv_in = SpatialInputConv(student.conv_in, 16)
    student.register_to_config(in_channels=20)
    original = merged_state(teacher_unet)
    copied = {}
    for key, tensor in student.state_dict().items():
        source_key = key
        match = re.match(r'up_blocks\.(\d+)\.(resnets|attentions)\.(\d+)\.(.*)', key)
        concat = None
        if match:
            block, family, layer, suffix = match.groups()
            b, j = int(block), int(layer)
            # The last up-resnet consumes the next resolution's skip. Preserve
            # that semantic position instead of blindly taking teacher index1.
            old_j = 0 if j == 0 else 2
            source_key = f'up_blocks.{b}.{family}.{old_j}.{suffix}'
            if family == 'resnets' and suffix in ('conv1.weight', 'conv_shortcut.weight', 'norm1.weight', 'norm1.bias'):
                old_previous = old_channels[-1] if b == 0 and j == 0 else old_channels[-b] if j == 0 else old_channels[-1-b]
                new_previous = channels[-1] if b == 0 and j == 0 else channels[-b] if j == 0 else channels[-1-b]
                concat = (old_previous, new_previous)
        if source_key not in original:
            raise ValueError('Missing teacher initialization '+source_key)
        copied[key] = resize_tensor(original[source_key].float().cpu(), tensor.shape, key, concat=concat)
    student.load_state_dict(copied, strict=True)
    return student, {'channels': list(channels), 'heads': list(heads), 'layers_per_block': 1,
                     'time_projection_preserved': True, 'lora_folded': True,
                     'teacher_parameters': sum(p.numel() for p in teacher_unet.parameters()),
                     'student_parameters': sum(p.numel() for p in student.parameters())}


def spatial_attention_map(value):
    if isinstance(value, tuple):
        value = value[0]
    value = value.float().square().mean(1, keepdim=True)
    return value/(value.square().mean((2, 3), keepdim=True).sqrt().clamp_min(1e-6))


def attention_hooks(unet, destination, detach=False):
    handles = []
    for group in ('down_blocks', 'up_blocks'):
        for i, block in enumerate(getattr(unet, group)):
            def hook(module, args, out, key=group+'.'+str(i)):
                value = spatial_attention_map(out)
                destination[key] = value.detach().cpu() if detach else value
            handles.append(block.register_forward_hook(hook))
    return handles
