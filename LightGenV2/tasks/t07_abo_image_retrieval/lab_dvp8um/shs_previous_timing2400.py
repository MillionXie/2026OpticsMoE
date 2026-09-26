"""Replay the prior SHS Controller timing, with one held phase for the test."""
import copy
import json
from pathlib import Path
import numpy as np
from PIL import Image
import four_image_flow as flow
from shs_physical2400 import CORNERS
from slm_camera import Controller

OUT=flow.PROJECT/'runs/abo_i2i_20260926/shs_previous240'


def main():
    OUT.mkdir(parents=True,exist_ok=False);flow.BASE_CORNERS=CORNERS.copy()
    c=json.loads((flow.ROOT/'LGVQ_Spatial_Lab_SHS_8um/LAB.local.json').read_text(encoding='utf-8-sig'))
    c['camera']['exposure_us']=400.;c['camera']['gain']='Gain_X4';c['settle_delay_ms']=240.
    marker=np.zeros((478,478),np.uint8);marker[70:150,60:150]=160
    marker[280:350,300:430]=160;marker[350:430,80:140]=160
    paths={}
    for name,a in [('marker',marker),('inverse',160-marker)]:
        p=OUT/(name+'.bmp');Image.fromarray(flow.active_to_native(a)).save(p);paths[name]=p
    flat=flow.PROJECT/'runs/abo_i2i_20260926/shs_phase_verify/flat.bmp'
    rows=[];refs={}
    with flow.PhaseHDMI(flow.PHASE_SDK,flow.PHASE_LUT,settle_s=.8,pixel_format='rgba') as phase:
        phase.show(flat)
        with Controller(c) as hw:
            for i in range(30):
                for name in ('marker','inverse'):
                    raw,meta=hw.capture(paths[name]);a=flow.orient(flow.warp(raw),'flip_v')
                    if name not in refs:refs[name]=a
                    row={'index':i,'name':name,'pcc_correct':flow.pcc(a,refs[name]),
                         'p99':float(np.percentile(a,99)),'meta':meta}
                    if len(refs)==2:row['pcc_wrong']=flow.pcc(a,refs['inverse' if name=='marker' else 'marker'])
                    rows.append(row)
                    Image.fromarray(a).save(OUT/f'{i:02d}_{name}.png')
                    flow.write(OUT/'report.json',{'complete':False,'config':c,'rows':rows})
    flow.write(OUT/'report.json',{'complete':True,'config':c,'rows':rows})


if __name__=='__main__':main()
