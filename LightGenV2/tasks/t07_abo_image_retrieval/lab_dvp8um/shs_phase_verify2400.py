"""Independent phase-quadrant actuation test and a longer 300ms old-frame test."""
import numpy as np
from pathlib import Path
from PIL import Image
import four_image_flow as flow
from shs_physical2400 import SHSBench,CORNERS

OUT=flow.PROJECT/'runs/abo_i2i_20260926/shs_phase_verify'


def main():
    flow.BASE_CORNERS=CORNERS.copy()
    OUT.mkdir(parents=True,exist_ok=False)
    white=OUT/'white.bmp'
    Image.fromarray(flow.active_to_native(np.full((478,478),255,np.uint8))).save(white)
    flat=OUT/'flat.bmp';Image.fromarray(np.full((1200,1920),255,np.uint8)).save(flat)
    phases={'flat':flat}
    for name,(y,x) in zip(('TL','TR','BL','BR'),((30,30),(30,284),(284,30),(284,284))):
        g=np.zeros((478,478),np.uint8)
        g[y:y+164,x:x+164]=np.tile(((np.arange(164)//2)%2*255).astype(np.uint8),(164,1))
        p=OUT/(name+'.bmp');Image.fromarray(255-flow.active_to_native(g,'phase')).save(p);phases[name]=p
    measurements={}
    rows=[]
    with SHSBench(OUT,300,400,{}) as bench:
        for name in ('flat','TL','TR','BL','BR','flat'):
            key=f'{name}_{len(rows)}'
            a,_=bench.capture('phase_quadrants',phases[name],[white],[key],'flip_v')
            measurements[key]=a[0];rows.append(bench.rows[-1])
        reference=measurements['flat_0']
        regions={name:(slice(y,y+164),slice(x,x+164)) for name,(y,x) in
                 zip(('TL','TR','BL','BR'),((30,30),(30,284),(284,30),(284,284)))}
        contrasts={}
        for i,name in enumerate(('TL','TR','BL','BR'),1):
            difference=reference.astype(float)-measurements[f'{name}_{i}'].astype(float)
            contrasts[name]={region:float(difference[index].mean()) for region,index in regions.items()}
        bench.wait=.3
        marker=flow.PROJECT/'runs/abo_i2i_20260926/shs_verify/marker.bmp'
        inverse=flow.PROJECT/'runs/abo_i2i_20260926/shs_verify/inverse.bmp'
        refs={}
        for i in range(12):
            for name,path in [('marker',marker),('inverse',inverse)]:
                a,_=bench.capture('timing',flat,[path],[f'{i}_{name}'],'flip_v')
                if name not in refs:refs[name]=a[0]
                row=dict(bench.rows[-1]);row['pcc_to_first']=flow.pcc(a[0],refs[name]);rows.append(row)
    flow.write(OUT/'report.json',{'phase_quadrant_contrasts':contrasts,
              'flat_repeat_pcc':flow.pcc(reference,measurements['flat_5']), 'rows':rows})


if __name__=='__main__':main()
