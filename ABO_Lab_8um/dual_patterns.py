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

def logical_pairs():
    if (ROOT/'runtime/backend/experiments/hardware_sdk/generators/dual_slm_alignment.py').is_file(): setup_imports()
    else: sys.path.insert(0,str(ROOT.parent))
    from experiments.hardware_sdk.generators.dual_slm_alignment import _checker,_registered_checker_grating
    from experiments.hardware_sdk.generators.dual_slm_registration_sweep import large_block_mask,single_axis_masked_grating
    pairs=[]
    for name,cell in [('01_check64',64),('04_check16',16)]:
        a=_checker(478,cell)
        p=_registered_checker_grating(478,cell,8,a,orientation_mode='visible_checker_cells')
        pairs.append((name,a,p,cell,'alternating x/y in visible white cells'))
    a,_=large_block_mask(478,48)
    for name,axis in [('02_blocks_x','x'),('03_blocks_y','y')]:
        pairs.append((name,a,single_axis_masked_grating(a,8,axis),48,axis))
    return sorted(pairs,key=lambda item:item[0])

def preview(path,a,p):
    # Geometry illustration only: phase stripes are NOT simulated intensity.
    imgs=[a,p.astype(np.uint16)*255//128,np.where(a>0,np.where(p>0,255,70),0).astype(np.uint8)]
    panel=382; canvas=Image.new('RGB',(3*panel, panel+38),'white'); draw=ImageDraw.Draw(canvas)
    for i,(arr,label) in enumerate(zip(imgs,['Amplitude 0/255','Phase 0/pi (display stretched)','Pair geometry ONLY, not CCD'])):
        canvas.paste(Image.fromarray(arr.astype(np.uint8)).resize((panel,panel),Image.Resampling.NEAREST).convert('RGB'),(i*panel,38))
        draw.text((i*panel+8,12),label,fill='black')
    canvas.save(path)

def generate(c,output=None):
    dest=ROOT/'generated/dual' if output is None else output
    if c['amplitude_slm']['pixel_pitch_um']!=8 or c['phase_slm']['pixel_pitch_um']!=8:
        raise ValueError('This paired registration suite is specifically 8 um / 8 um.')
    rows=[]
    for name,a,p,cell,axis in logical_pairs():
        folder=dest/name; folder.mkdir(parents=True,exist_ok=True)
        ac=raster(a,c['amplitude_slm'],c,nearest=True)
        pc=raster(p,c['phase_slm'],c,nearest=True)
        opposite=dict(c['phase_slm']); opposite['flip_vertical']=not opposite['flip_vertical']
        pv=raster(p,opposite,c,nearest=True)
        save(folder/'A.bmp',ac); save(folder/'P.bmp',pc); save(folder/'P_V.bmp',pv)
        preview(folder/'preview.png',a,p)
        row={'folder':name,'amplitude':'A.bmp','phase':'P.bmp','opposite_vertical_phase':'P_V.bmp',
             'phase_flip_vertical':bool(c['phase_slm']['flip_vertical']),
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
