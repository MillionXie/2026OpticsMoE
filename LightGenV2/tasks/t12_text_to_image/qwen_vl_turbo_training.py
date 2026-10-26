"""Training-scope selection only; deployment architecture stays unchanged."""


def configure_unet_training(unet, mode='lora'):
    if mode not in {'lora','cross_attention'}:
        raise ValueError('Unknown UNet training scope '+mode)
    cross=0
    for name,param in unet.named_parameters():
        base_cross='.attn2.' in name and 'lora_' not in name
        param.requires_grad_('lora_' in name or (mode=='cross_attention' and base_cross))
        cross+=int(base_cross)
    if mode=='cross_attention' and not cross:
        raise ValueError('No native cross-attention parameters were found')
