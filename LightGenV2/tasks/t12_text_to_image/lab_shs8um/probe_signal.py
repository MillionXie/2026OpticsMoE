"""Two identical bright probes and one dark probe, with unchanged hardware settings."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True)
    p.add_argument('--abo-project',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(a.abo_project/'lab_dvp8um'))
    import four_image_flow as flow
    from shs_physical2400 import SHSBench,CORNERS
    flow.BASE_CORNERS=CORNERS.copy()
    phase=a.project/'runs/physical_pilot_01/phase/language_router.bmp'
    paths=[];ids=['dark','white','white_repeat']
    for sid,gray in zip(ids,(0,255,255)):
        path=a.output/(sid+'.bmp')
        Image.fromarray(flow.active_to_native(np.full((478,478),gray,np.uint8))).save(path);paths.append(path)
    with SHSBench(a.output,400,240,{}) as bench:
        raw,receipt=bench.capture('signal_probe',phase,paths,ids,'flip_v')
        rows=bench.rows.copy()
    report=dict(status='complete',scope='same phase/dark-white diagnostic, not task performance',
        rows=rows,repeat_pcc=flow.pcc(raw[1],raw[2]),
        white_minus_dark_mean=float((raw[1].astype(float)-raw[0]).mean()))
    flow.write(a.output/'report.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
