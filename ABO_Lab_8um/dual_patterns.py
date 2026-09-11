"""Paired masked gratings, matching the historical checker/large-block design.

Nominal 1:1 relay, two 8 um panels; NOT independent full-screen gratings.
Registration BMPs are nearest/binary; model amplitude interpolation is unchanged.
"""
import argparse
import sys
import numpy as np
from PIL import Image,ImageDraw
from common import ROOT,config,write,sha,hardware_identity,setup_imports
from patterns import raster,save
from phase_encoding import encode_gray,generated_root,mode

def logical_pairs():
    from registration import logical_pairs as portable_pairs
    return portable_pairs()

def preview(path,a,p):
    # Geometry illustration only: phase stripes are NOT simulated intensity.
    imgs=[a,p.astype(np.uint16)*255//128,np.where(a>0,np.where(p>0,255,70),0).astype(np.uint8)]
    panel=382; canvas=Image.new('RGB',(3*panel, panel+38),'white'); draw=ImageDraw.Draw(canvas)
    for i,(arr,label) in enumerate(zip(imgs,['Amplitude 0/255','Phase 0/pi (display stretched)','Pair geometry ONLY, not CCD'])):
        canvas.paste(Image.fromarray(arr.astype(np.uint8)).resize((panel,panel),Image.Resampling.NEAREST).convert('RGB'),(i*panel,38))
        draw.text((i*panel+8,12),label,fill='black')
    canvas.save(path)

def generate(c,output=None):
    dest=generated_root(ROOT,c)/'dual' if output is None else output
    if c['amplitude_slm']['pixel_pitch_um']!=8 or c['phase_slm']['pixel_pitch_um']!=8:
        raise ValueError('This paired registration suite is specifically 8 um / 8 um.')
    rows=[]
    for name,a,p,cell,axis in logical_pairs():
        folder=dest/name; folder.mkdir(parents=True,exist_ok=True)
        ac=raster(a,c['amplitude_slm'],c,nearest=True)
        pc=encode_gray(raster(p,c['phase_slm'],c,nearest=True),c)
        opposite=dict(c['phase_slm']); opposite['flip_vertical']=not opposite['flip_vertical']
        pv=encode_gray(raster(p,opposite,c,nearest=True),c)
        save(folder/'A.bmp',ac); save(folder/'P.bmp',pc); save(folder/'P_V.bmp',pv)
        preview(folder/'preview.png',a,p)
        row={'folder':name,'amplitude':'A.bmp','phase':'P.bmp','opposite_vertical_phase':'P_V.bmp',
             'phase_gray_encoding':mode(c),'phase_flip_vertical':bool(c['phase_slm']['flip_vertical']),
             'opposite_phase_flip_vertical':bool(opposite['flip_vertical']),
             'phase_flip_horizontal':bool(c['phase_slm']['flip_horizontal']),
             'amplitude_flip_vertical':bool(c['amplitude_slm']['flip_vertical']),
             'amplitude_flip_horizontal':bool(c['amplitude_slm']['flip_horizontal']),
             'cell_model_px':cell,'cell_device_px':cell*17/8,'grating_period_device_px':17.0,
             'grating_axis':axis,'sha256':{f:sha(folder/f) for f in ('A.bmp','P.bmp','P_V.bmp')},
             'preview':'canonical logical layout, NOT a simulated CCD or native playback bitmap'}
        write(folder/'pair.json',row); rows.append(row)
    report={'schema':1,'hardware_identity':hardware_identity(c),'relay_scale_k':1.0,
            'effective_width_um':8126,'effective_device_width_px':1015.75,
            'amplitude_size_wh':c['amplitude_slm']['expected_resolution_wh'],
            'phase_size_wh':c['phase_slm']['size_wh'],
            'amplitude_center_xy':c['amplitude_slm']['center_xy'],'phase_center_xy':c['phase_slm']['center_xy'],
            'sampling':'identical physical-coordinate nearest sampling on both panels, binary values only',
            'amplitude_gray_lut_applied':False,'pairs':rows}
    write(dest/'pairs.json',report)
    print('Paired registration BMPs:',dest)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config'); args=p.parse_args()
    generate(config(args.config)[0])
