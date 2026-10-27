"""Auditable synthetic product-scene targets, for DATA preparation only.

ABO foregrounds, independent teacher room photos and same-split wooden tabletop
textures form paired supervision. No compositing operation exists in inference.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from .qwen_vl_turbo_data import normalized_product, sha256


@lru_cache(maxsize=256)
def asset_image(path, digest, size):
    if sha256(Path(path)) != digest:
        raise ValueError('Background asset bytes changed')
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB').resize((size,size),Image.Resampling.LANCZOS)).astype(np.float32)


def light_field(shape, tone, brightness, direction):
    if tone not in ('warm','cool','neutral') or brightness not in ('bright','softly lit') or direction not in ('left','right'):
        raise ValueError('Unknown scene illumination')
    h,w=shape;xx=np.linspace(0,1,w,dtype=np.float32)[None,:]
    directional=.90+.10*(xx if direction=='right' else 1-xx)
    tint={'warm':(1.05,1.,.92),'cool':(.93,1.,1.05),'neutral':(1.,1.,1.)}[tone]
    exposure=1. if brightness=='bright' else .76
    return directional[:,:,None]*np.asarray(tint,dtype=np.float32)[None,None,:]*exposure


def background_rgb(asset, table, size=256):
    if asset['split'] != table['split']:
        raise ValueError('Tabletop texture cannot cross TRAIN/DEV/TEST')
    room=asset_image(asset['image'],asset['sha256'],size).copy()
    texture=asset_image(table['image'],table['sha256'],size)
    # All rooms get a genuine close foreground display plane. This avoids a
    # desk-sized lamp pasted as a giant floor lamp into a wide room photograph.
    horizon=int(.67*size)
    crop=Image.fromarray(texture.clip(0,255).astype(np.uint8)).crop((0,int(.76*size),size,int(.98*size)))
    surface=np.asarray(crop.resize((size,size-horizon),Image.Resampling.BICUBIC)).astype(np.float32)
    surface*=np.linspace(.92,1.02,len(surface),dtype=np.float32)[:,None,None]
    room[horizon:]=surface
    room*=light_field((size,size),asset['tone'],asset['brightness'],asset['direction'])
    return room.clip(0,255)


def unmatte_white(rgb, alpha):
    a=np.asarray(alpha,dtype=np.float32)/255
    value=np.asarray(rgb,dtype=np.float32)
    # Raw ABO JPEGs were already composited on white. Recover foreground color
    # before reusing genuine alpha, preventing a second white fringe blend.
    return ((value-(1-a[:,:,None])*255)/np.maximum(a[:,:,None],1/255)).clip(0,255),a


def recolor_component(foreground, alpha, variant):
    if not variant:return foreground
    h,w=alpha.shape;yy=np.arange(h)[:,None]/h;rgb=foreground/255
    spread=rgb.max(2)-rgb.min(2)
    if variant['part']=='shade':
        mask=(yy<.62)&(rgb[:,:,1]>1.10*rgb[:,:,0])&(rgb[:,:,1]>1.05*rgb[:,:,2])&(spread>.15)&(alpha>.1)
    elif variant['part']=='base':
        mask=(yy>.48)&(rgb[:,:,1]>1.15*rgb[:,:,0])&(rgb[:,:,2]>1.15*rgb[:,:,0])&(spread>.15)&(alpha>.1)
    else:raise ValueError('Unsupported color-edit component')
    if int(mask.sum())<30:raise ValueError('Synthetic color label has insufficient real component pixels')
    color={'red':(.85,.12,.10),'blue':(.08,.25,.85)}[variant['color']]
    tint=np.asarray(color,dtype=np.float32);luminance=rgb@np.array([.2126,.7152,.0722],dtype=np.float32)
    shaded=(luminance[:,:,None]/float(tint@np.array([.2126,.7152,.0722])))*tint
    result=foreground.copy();result[mask]=shaded.clip(0,1)[mask]*255
    return result


def compose(product, asset, table, size=256, color_variant=None):
    rgb,alpha=normalized_product(product,size);foreground,a=unmatte_white(rgb,alpha)
    foreground=recolor_component(foreground,a,color_variant)
    foreground*=light_field((size,size),asset['tone'],asset['brightness'],asset['direction'])
    background=background_rgb(asset,table,size)
    ys,xs=np.where(a>.1)
    if not len(xs):raise ValueError('Empty ABO foreground')
    yy,xx=np.mgrid[0:size,0:size].astype(np.float32)/size
    center=(float(xs.min()+xs.max())/2/size)+(-.012 if asset['direction']=='right' else .012)
    bottom=min(.985,(float(ys.max())+2)/size)
    width=max(.06,(float(xs.max()-xs.min())/size)*.35)
    shadow=np.exp(-((xx-center)/width)**2-((yy-bottom)/.018)**2)
    background*=1-.18*shadow[:,:,None]
    result=foreground.clip(0,255)*a[:,:,None]+background*(1-a[:,:,None])
    return Image.fromarray(result.clip(0,255).round().astype(np.uint8))


def render_pair(row,size=256):
    if row['render_protocol']!='rich_product_scene_v1':raise ValueError('Wrong rich scene protocol')
    return (compose(row['source'],row['source_background_asset'],row['source_table_asset'],size),
            compose(row['target'],row['target_background_asset'],row['target_table_asset'],size,row.get('target_color_variant')))
