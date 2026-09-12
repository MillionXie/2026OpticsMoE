"""Diagnostic phase codes only; no hardware access or formal-mask changes.

Native 1920x1200, 8 um. Both code polarities are explicit. The selected
linearVoltage LUT is not a calibrated linear-phase LUT, so the lens and ramp
are nominal diagnostic patterns, not certified physical phase profiles.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image


def generate(out):
    out=Path(out)
    out.mkdir(parents=True,exist_ok=False)
    yy,xx=np.indices((1200,1920))
    fields={'P_flat_0':np.zeros((1200,1920),np.uint8)}
    for period in (8,4):
        ramp=np.rint(255*(xx%period)/(period-1)).astype(np.uint8)
        fields[f'P_gx_p{period}']=ramp
        fields[f'P_gx_p{period}_inverse']=255-ramp
    # Full-aperture nominal focusing lens. Pixel centres, f=+0.10m.
    phase=-np.pi*((8e-6*(xx-959.5))**2+(8e-6*(yy-599.5))**2)/(532e-9*.10)
    lens=np.rint(np.remainder(phase,2*np.pi)/(2*np.pi)*255).astype(np.uint8)
    fields['P_lens_10cm']=lens
    fields['P_lens_10cm_inverse']=255-lens
    files=[]
    for name,ar in fields.items():
        p=out/(name+'.bmp');Image.fromarray(ar).save(p)
        files.append({'name':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    report={'purpose':'Qualitative phase-response diagnostic only',
        'resolution_wh':[1920,1200],'pixel_pitch_um':8,'nominal_wavelength_nm':532,
        'nominal_distance_m':.1,'spatial_flip':'none',
        'gray_encoding':'explicit raw codes; *_inverse is exactly 255-g; do not invert twice',
        'cautions':['linearVoltage does not certify linear phase response',
                    'Full aperture 10cm lens has undersampled outer zones; use central illuminated patch first',
                    'Camera displacement requires camera scale/magnification calibration; do not infer CCD pixels from SLM pitch',
                    'Do not substitute these diagnostics for trained inference masks'],
        'files':files}
    (out/'patterns.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();generate(a.out);print(a.out.resolve())
