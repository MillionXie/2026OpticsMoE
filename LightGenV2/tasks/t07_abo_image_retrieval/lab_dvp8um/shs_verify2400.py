"""Independent geometric orientation and repeat timing tests; no label fitting."""
import json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import four_image_flow as flow
from shs_physical2400 import SHSBench, CORNERS

OUT=flow.PROJECT/'runs/abo_i2i_20260926/shs_verify'


def main():
    flow.BASE_CORNERS=CORNERS.copy()
    OUT.mkdir(parents=True,exist_ok=False)
    flat=flow.PROJECT/'runs/abo_i2i_20260924/05_brightness_health/patterns/phase_flat_pi_inverted.bmp'
    marker=np.zeros((478,478),np.uint8)
    marker[70:150,60:150]=255
    marker[280:350,300:430]=160
    marker[350:430,80:140]=220
    patterns={'marker':marker,'inverse':255-marker,'black':np.zeros_like(marker),'white':np.full_like(marker,255)}
    paths={}
    for name,a in patterns.items():
        paths[name]=OUT/(name+'.bmp')
        Image.fromarray(flow.active_to_native(a)).save(paths[name])
    records=[]
    frames={}
    with SHSBench(OUT,300,400,{}) as bench:
        for name in ('black','marker','inverse','marker'):
            key=f'anchor_{name}_{len(records)}'
            imgs,_=bench.capture('anchor',flat,[paths[name]],[key],'identity')
            records.append(bench.rows[-1]);frames[key]=imgs[0]
        measured=frames['anchor_marker_1']
        scores={name:flow.pcc(cv2.GaussianBlur(a.astype(np.float32),(0,0),3),
                             cv2.GaussianBlur(marker.astype(np.float32),(0,0),3))
                for name,a in flow.camera_variants(measured).items()}
        orientation=max(scores,key=scores.get)
        for wait in (200,240,300,400):
            bench.wait=wait/1000
            for rep in range(3):
                for name in ('inverse','marker'):
                    key=f'w{wait}_r{rep}_{name}'
                    imgs,_=bench.capture('timing',flat,[paths[name]],[key],'identity')
                    row=dict(bench.rows[-1])
                    reference=frames['anchor_marker_1'] if name=='marker' else frames['anchor_inverse_2']
                    row['pcc_to_400ms_reference']=flow.pcc(imgs[0],reference)
                    records.append(row)
        digit_root=flow.ROOT.parent/'MNIST_10cm_8um_Bench_Test_20260923/02_mnist_10cm'
        phase=digit_root/'phase/B_RECOMMENDED_native8_best.bmp'
        bench.wait=.4
        for label in range(4):
            path=sorted((digit_root/'inputs_40_fixed').glob(f'*_y{label}.bmp'))[0]
            imgs,_=bench.capture('mnist',phase,[path],[path.stem],orientation)
            records.append(bench.rows[-1])
    flow.write(OUT/'report.json',{'camera_orientation':orientation,'orientation_scores':scores,
              'anchor_repeat_pcc':flow.pcc(frames['anchor_marker_1'],frames['anchor_marker_3']),
              'mnist_scope':'Four fixed digits, qualitative optical smoke, not dataset accuracy',
              'rows':records,'corners_screen_TL_TR_BR_BL':CORNERS.tolist()})


if __name__=='__main__':main()
