"""Training-scope selection only; deployment architecture stays unchanged."""


def configure_unet_training(unet, mode='lora'):
    if mode not in {'lora','cross_attention','cross_attention_decoder'}:
        raise ValueError('Unknown UNet training scope '+mode)
    cross=0
    last_decoder=(f'up_blocks.{len(unet.up_blocks)-1}.' if mode=='cross_attention_decoder' else None)
    for name,param in unet.named_parameters():
        base_cross='.attn2.' in name and 'lora_' not in name
        param.requires_grad_('lora_' in name or (mode!='lora' and base_cross) or
                             (last_decoder is not None and name.startswith(last_decoder)))
        cross+=int(base_cross)
    if mode!='lora' and not cross:
        raise ValueError('No native cross-attention parameters were found')
