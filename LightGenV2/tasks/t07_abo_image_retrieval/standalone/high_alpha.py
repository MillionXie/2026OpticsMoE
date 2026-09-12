"""High-optical-share training utilities; no additional inference branch."""
import math
import torch
from torch import nn
from PIL import Image,ImageOps,ImageEnhance,ImageFilter
from .curriculum import parameter_kind


def convert_payload(payload,config):
    lo,hi=config['fusion_alpha_min'],config['fusion_alpha_max']
    value=config['fusion_alpha_initial']
    if not .4<lo<value<hi<=1:raise ValueError('High-alpha contract invalid')
    meta=dict(payload['metadata']);state=dict(payload['state_dict'])
    same=meta.get('fusion_alpha_min')==lo and meta.get('fusion_alpha_max')==hi
    meta.update(fusion_alpha_min=lo,fusion_alpha_max=hi,optical_training_noise=config['optical_training_noise'])
    if not same:
        raw=math.log((value-lo)/(hi-value))
        for name in state:
            if 'optical_fusion_logit' in name:state[name]=torch.full_like(state[name],raw)
    return dict(payload,metadata=meta,state_dict=state)


def group_kind(name):
    kind=parameter_kind(name)
    return 'optical_electronic' if kind=='electronic' and '.optics.' in name else kind


def optical_heads(classes):
    return nn.ModuleDict({m:nn.Sequential(nn.LayerNorm(384),nn.Linear(384,classes)) for m in ('vision','language')})


def optical_classification_loss(model,heads,labels):
    losses=[]
    for m in ('vision','language'):
        x=getattr(model,m).last_optical
        if x is None:raise RuntimeError('Optical auxiliary supervision needs optical branch enabled')
        pooled=torch.cat((x.float().mean(1),x.float().amax(1)),dim=-1)
        losses.append(torch.nn.functional.cross_entropy(heads[m](pooled),labels,label_smoothing=.05))
    return torch.stack(losses).mean()


def white_margin_box(image):
    """Training-only conservative near-white margin detection, no labels.

    Every RGB pixel with any channel <254 remains inside the crop. This is a
    threshold guarantee, NOT a semantic guarantee about white object parts.
    Keep a2% guard, trim at most20% per side, and reject tiny/ambiguous objects.
    """
    import numpy as np
    if image.mode!='RGB' or image.size!=(224,224):
        raise ValueError('White margin augmentation expects RGB224 input')
    a=np.asarray(image);white=np.all(a>=254,axis=2)
    full=(0,0,224,224)
    border=np.concatenate((white[0],white[-1],white[:,0],white[:,-1]))
    ys,xs=np.nonzero(~white)
    if border.mean()<=.9 or not len(xs):return full
    x0,x1,y0,y1=int(xs.min()),int(xs.max()+1),int(ys.min()),int(ys.max()+1)
    if (x1-x0)*(y1-y0)<.15*224*224 or min(x1-x0,y1-y0)<8:return full
    margin=4;cap=44
    box=(min(max(x0-margin,0),cap),min(max(y0-margin,0),cap),
         max(min(x1+margin,224),224-cap),max(min(y1+margin,224),224-cap))
    return box if 224/max(box[2]-box[0],box[3]-box[1])>=1.05 else full


def augment(image,rng,cfg):
    probability=cfg.get('white_margin_zoom_probability',0.)
    if isinstance(probability,bool) or not isinstance(probability,(int,float)) or not math.isfinite(probability) or not 0<=probability<=1:
        raise ValueError('Invalid white margin zoom probability')
    if probability:
        if cfg['minimum_crop_side_fraction']!=1 or cfg['rotation_degrees']!=0 or 'contain_jitter_min_scale' in cfg:
            raise ValueError('White margin zoom cannot combine with arbitrary crop/rotation/placement jitter')
        if rng.random()<probability:
            box=white_margin_box(image)
            if box!=(0,0,224,224):
                resized=ImageOps.contain(image.crop(box),(224,224),Image.Resampling.BICUBIC)
                canvas=Image.new('RGB',(224,224),'white')
                canvas.paste(resized,((224-resized.width)//2,(224-resized.height)//2))
                image=canvas
    if 'contain_jitter_min_scale' in cfg:
        minimum=cfg['contain_jitter_min_scale']
        if not 0<minimum<=1 or cfg['minimum_crop_side_fraction']!=1 or cfg['rotation_degrees']!=0:
            raise ValueError('Full-object augmentation requires no crop or rotation')
        probability=cfg.get('contain_jitter_probability',1.)
        if not math.isfinite(probability) or not 0<=probability<=1:
            raise ValueError('Invalid full-object jitter probability')
        # No additional RNG draw for old profiles, preserving their trajectories.
        if 'contain_jitter_probability' not in cfg or rng.random()<probability:
            size=round(224*rng.uniform(minimum,1.))
            resized=image.resize((size,size),Image.Resampling.BICUBIC)
            canvas=Image.new('RGB',(224,224),(255,255,255))
            canvas.paste(resized,(rng.randint(0,224-size),rng.randint(0,224-size)))
            image=canvas
    side=round(224*rng.uniform(cfg['minimum_crop_side_fraction'],1.))
    left,top=[rng.randint(0,224-side) for _ in range(2)]
    image=image.crop((left,top,left+side,top+side)).resize((224,224),Image.Resampling.BICUBIC)
    if rng.random()<cfg['horizontal_flip_probability']:image=ImageOps.mirror(image)
    if rng.random()<.5:
        image=image.rotate(rng.uniform(-cfg['rotation_degrees'],cfg['rotation_degrees']),Image.Resampling.BICUBIC,fillcolor=(255,255,255))
    image=ImageEnhance.Brightness(image).enhance(rng.uniform(cfg['brightness_min'],cfg['brightness_max']))
    image=ImageEnhance.Contrast(image).enhance(rng.uniform(cfg['contrast_min'],cfg['contrast_max']))
    if 'color_min' in cfg:image=ImageEnhance.Color(image).enhance(rng.uniform(cfg['color_min'],cfg['color_max']))
    if rng.random()<cfg['blur_probability']:image=image.filter(ImageFilter.GaussianBlur(cfg['blur_radius']))
    return image


def phase_change(model,initial):
    result={}
    for name,p in model.named_parameters():
        if parameter_kind(name) not in ('phase','router'):continue
        raw=p.detach().cpu().float();before=initial[name].float()
        delta=2*torch.pi*(raw.sigmoid()-before.sigmoid())
        wrapped=torch.atan2(delta.sin(),delta.cos())
        result[name]=dict(raw_rms=float((raw-before).square().mean().sqrt()),phase_circular_rms_radians=float(wrapped.square().mean().sqrt()))
    return result


def phase_shuffle(model,seed=42):
    """Preserve each phase histogram, disturb its spatial pattern. Caller restores."""
    generator=torch.Generator().manual_seed(seed);saved={}
    with torch.no_grad():
        for name,p in model.named_parameters():
            if parameter_kind(name) not in ('phase','router'):continue
            saved[name]=p.detach().clone()
            order=torch.randperm(p.numel(),generator=generator).to(p.device)
            p.copy_(p.flatten()[order].reshape_as(p))
    return saved


def restore_phase(model,saved):
    with torch.no_grad():
        for name,p in model.named_parameters():
            if name in saved:p.copy_(saved[name])
