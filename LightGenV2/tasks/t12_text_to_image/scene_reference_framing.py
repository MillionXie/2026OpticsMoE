"""Input-only white-margin normalization; no target/mask/model information."""
import hashlib
import math
import numpy as np
from PIL import Image

FRAMING='white_rgb_bbox_pad12_v1'


def frame_reference(image,mode='original'):
    rgb=image.convert('RGB')
    if mode=='original':return rgb,dict(mode=mode,rotation=False,target_or_mask_used=False)
    if mode!=FRAMING:raise ValueError('Explicit supported reference framing required')
    pixels=np.asarray(rgb);visible=pixels.min(axis=2)<250
    if not visible.any():raise ValueError('White reference has no observable subject')
    y,x=np.nonzero(visible);box=(int(x.min()),int(y.min()),int(x.max())+1,int(y.max())+1)
    width=box[2]-box[0];height=box[3]-box[1];margin=max(8,math.ceil(max(width,height)*.12));side=max(width,height)+2*margin
    canvas=Image.new('RGB',(side,side),'white');canvas.paste(rgb.crop(box),((side-width)//2,(side-height)//2))
    framed=canvas.resize((1024,1024),Image.Resampling.LANCZOS)
    return framed,dict(mode=mode,original_size=list(rgb.size),source_nonwhite_bbox=list(box),threshold=250,
        square_side=side,margin_pixels=margin,model_input_size=[1024,1024],rotation=False,target_or_mask_used=False,
        source_visible_fraction=float(visible.mean()),framed_visible_fraction=float((np.asarray(framed).min(axis=2)<250).mean()),
        input_RGB_pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),framed_RGB_pixel_sha256=hashlib.sha256(np.asarray(framed).tobytes()).hexdigest())
