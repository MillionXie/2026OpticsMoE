"""Pinned SALICON three-pass adapter for Meadowlark 17um + manual phase 8um + TUCam.

Reuses the hardware SDK's acquisition/geometry/LUT handling. Never imports the
SHS Controller. A calibrated, existing formal_hardware.yaml is required.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import time


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def hardware(path):
    import yaml
    path=Path(path).resolve()
    raw=yaml.safe_load(path.read_text(encoding='utf-8'))
    amp,phase,cam=raw['amplitude_slm'],raw['phase_slm'],raw['camera']
    if amp['driver']!='meadowlark_pcie' or cam['driver']!='tucam' or phase['driver']!='manual':
        raise ValueError('Require meadowlark_pcie + manual phase + tucam')
    if float(amp['pixel_pitch_um'])!=17 or float(phase['pixel_pitch_um'])!=8:
        raise ValueError('Wrong physical pixel pitches')
    if list(amp['expected_resolution_wh'])!=[1024,1024] or list(phase['expected_resolution_wh'])!=[1920,1200]:
        raise ValueError('Wrong panel dimensions')
    geo=cam['detector_geometry']
    if not geo['enabled'] or not geo['expected_file_sha256']:
        raise ValueError('Calibrate logical-corner homography before binding')
    if cam['saved_frame_size_wh']!=[478,478] or cam['saved_frame_bit_depth']!=8 or cam.get('saved_frame_input_range') is None:
        raise ValueError('Require canonical 478x478, 8-bit PNG and explicit fixed input range')
    if raw['output_extension']!='.png' or cam.get('auto_exposure',False) or float(cam['exposure_us'])<=0:
        raise ValueError('Require PNG and positive fixed exposure, no auto exposure')
    if not raw.get('require_phase_mask') or not raw.get('confirm_before_start',True):
        raise ValueError('Exact manual phase confirmation must be enabled')
    if raw.get('max_files') is not None:
        raise ValueError('Set max_files: null; sample count is selected by session')
    if float(raw['settle_delay_ms'])<0:
        raise ValueError('Invalid settle delay')
    assets={}
    for label,value in [('lut',amp['lut_file']),('geometry',geo['contract_file'])]:
        f=Path(value)
        if not f.is_absolute():f=path.parent/f
        f=f.resolve();assets[label]={'path':str(f),'sha256':sha(f)}
    if assets['geometry']['sha256']!=geo['expected_file_sha256']:
        raise ValueError('Geometry file SHA does not match formal hardware YAML')
    return raw,assets


def bind(a):
    from LightGenV2.tasks.t06_video_quality_assessment.lab_runtime import write
    dst=Path(a.config).resolve()
    if dst.exists():raise FileExistsError('Use a new binding filename; never overwrite active configuration')
    raw,assets=hardware(a.hardware_config)
    roi=raw['amplitude_roi']
    if [roi['width'],roi['height']]!=[478,478]:raise ValueError('Amplitude ROI must be 478x478')
    if a.phase_center is None or a.phase_gray_encoding is None or a.phase_orientation is None:
        raise ValueError('Specify phase center, gray encoding and orientation from YOUR calibration')
    # Legacy amplitude_roi uses pixel-center coordinates. raster uses continuous
    # boundary coordinates: 511.5 becomes 512, yielding BMP placement [273:751].
    c=dict(model_active_pixels=478,model_pitch_um=17,distance_m=.1,wavelength_nm=532,
           camera={'exposure_us':raw['camera']['exposure_us']},settle_delay_ms=raw['settle_delay_ms'],
           amplitude_slm=dict(size_wh=[1024,1024],pixel_pitch_um=17,
               center_xy=[float(roi['center_x'])+.5,float(roi['center_y'])+.5],
               flip_horizontal=a.amplitude_orientation in ('h','hv'),flip_vertical=a.amplitude_orientation in ('v','hv')),
           phase_slm=dict(size_wh=[1920,1200],pixel_pitch_um=8,center_xy=a.phase_center,
               gray_encoding=a.phase_gray_encoding,
               flip_horizontal=a.phase_orientation in ('h','hv'),flip_vertical=a.phase_orientation in ('v','hv')),
           detector_intensity_scale={s:1/255 for s in ('vision_router','vision_expert','vision_global')},
           native_sdk=dict(config=str(Path(a.hardware_config).resolve()),sha256=sha(a.hardware_config),assets=assets))
    write(dst,c)
    print('BOUND (no devices opened):',dst)
    print('LUT:',assets['lut']['path'],'SHA256:',assets['lut']['sha256'])
    print('Check physical alignment with 4 images BEFORE full acquisition.')


def verify_binding(a):
    c=read(a.config);b=c['native_sdk']
    if sha(b['config'])!=b['sha256']:raise ValueError('Hardware YAML changed: bind a new configuration/session')
    _,assets=hardware(b['config'])
    if assets!=b['assets']:raise ValueError('LUT or geometry changed: bind a new configuration/session')
    return c


def prepared(a):
    from LightGenV2.tasks.t03_saliency import lab_bench as bench
    root,s,state,c,release=bench.open_session(a)
    idx=bench.STAGES.index(a.stage)
    if state['measured_stages']!=list(bench.STAGES[:idx]):raise ValueError('Wrong acquisition order, or stage already completed')
    dest=s/'play'/a.stage;mf=read(dest/'manifest.json')
    if mf['hardware_sha256']!=state['hardware_sha256'] or mf['release_sha256']!=state['release_sha256'] or mf['stage']!=a.stage:
        raise ValueError('Prepared stage identity mismatch')
    expected={f['key'] for f in state['fields']}
    if len(mf['entries'])!=len(expected) or {e['key'] for e in mf['entries']}!=expected:
        raise ValueError('Wrong sample manifest')
    if {p.stem for p in dest.glob('*.bmp')}!=expected:raise ValueError('Extra or missing amplitude BMP')
    phase=s/mf['phase_file']
    if sha(phase)!=mf['phase_sha256']:raise ValueError('Phase BMP changed')
    for e in mf['entries']:
        if e['bmp']!=e['key']+'.bmp' or sha(dest/e['bmp'])!=e['sha256']:raise ValueError('Amplitude BMP changed')
        if set(e['upstream_ccd_sha256'])!=set(bench.STAGES[:idx]):raise ValueError('Wrong upstream prefix')
        for stage,digest in e['upstream_ccd_sha256'].items():
            _,record=bench.verified_ccd(s,stage,e['key'])
            if record['sha256']!=digest:raise ValueError('Upstream CCD changed; regenerate next input')
    return s,state,c,mf,phase


def import_captures(a):
    """Import ONLY SDK-recorded, canonical, fixed-scale PNGs; no second warp."""
    import numpy as np
    from PIL import Image
    from LightGenV2.tasks.t06_video_quality_assessment.lab_runtime import write
    s,state,c,mf,phase=prepared(a)
    with (s/'sdk_log'/a.stage/'capture_manifest.csv').open(encoding='utf-8-sig',newline='') as h:
        rows=list(csv.DictReader(h))
    expected={e['key']+'.png':e for e in mf['entries']}
    if len(rows)!=len(expected) or {r['ccd_capture'] for r in rows}!=set(expected):raise ValueError('Incomplete/duplicate SDK captures')
    records=[]
    for r in rows:
        e=expected[r['ccd_capture']];p=s/'ccd'/a.stage/r['ccd_capture']
        if r['output_sha256']!=sha(p) or r['amplitude_bmp_sha256']!=e['sha256'] or r['phase_mask_sha256']!=mf['phase_sha256']:
            raise ValueError('SDK image/SLM identity mismatch')
        if r['saved_frame_orientation']!='canonical_model_xy' or r['detector_geometry_file_sha256']!=c['native_sdk']['assets']['geometry']['sha256']:
            raise ValueError('CCD is not from the calibrated canonical warp')
        if r['per_frame_minmax_normalization'].lower()!='false' or r['background_subtraction'].lower()!='false':
            raise ValueError('Unexpected intensity processing')
        im=np.array(Image.open(p))
        if im.shape!=(478,478) or im.dtype!=np.uint8:raise ValueError('Invalid CCD shape/dtype')
        records.append((p.with_suffix('.record.json'),dict(sha256=sha(p),hardware_sha256=state['hardware_sha256'],
            amplitude_sha256=e['sha256'],phase_sha256=mf['phase_sha256'],upstream_ccd_sha256=e['upstream_ccd_sha256'],
            camera={'incomplete':False,'driver':'tucam','sdk_manifest_row':r},
            quality={'p99':float(np.percentile(im,99)),'saturation':float(np.mean(im==255))},
            processing='SDK homography + fixed 8-bit mapping; NO second warp/log/gamma/minmax',captured_at=time.time())))
    for p,r in records:write(p,r)
    state['measured_stages'].append(a.stage);write(s/'session.json',state)
    print('IMPORTED',a.stage,len(records),'actual CCD frames')


def capture(a):
    from experiments.hardware_sdk.workflows.acquire_folder import run
    s,state,c,mf,phase=prepared(a)
    report=run(c['native_sdk']['config'],input_override=s/'play'/a.stage,
        output_override=s/'ccd'/a.stage,log_override=s/'sdk_log'/a.stage,
        phase_override=phase,assume_yes=False,clear_output=False,validate_only=a.action=='check')
    if a.action=='capture':
        verify_binding(a)
        import_captures(a)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['bind','init','prepare','check','capture','import-captures','audit','evaluate'])
    p.add_argument('--project',default=str(Path(__file__).resolve().parent))
    p.add_argument('--config',default='LAB_Meadowlark.json')
    p.add_argument('--hardware-config',help='Existing calibrated formal_hardware.yaml; remains in its original folder')
    p.add_argument('--phase-center',type=float,nargs=2,help='Raster boundary-coordinate center, e.g. panel center 960 600; use measured alignment')
    p.add_argument('--phase-gray-encoding',choices=['normal','inverted_255_minus_g'])
    p.add_argument('--phase-orientation',choices=['none','h','v','hv'])
    p.add_argument('--amplitude-orientation',choices=['none','h','v','hv'],default='none')
    p.add_argument('--session',default='pilot4')
    p.add_argument('--stage',choices=['vision_router','vision_expert','vision_global'])
    p.add_argument('--fields',type=int,default=4,help='0=all 5000 test images; start with 4')
    p.add_argument('--device',default='cpu')
    a=p.parse_args();a.project=str(Path(a.project).resolve());a.config=str(Path(a.config).resolve())
    sys.path.insert(0,str(Path(a.project)/'runtime'))
    if a.fields<0:p.error('fields must be >=0')
    if a.action=='bind':
        if not a.hardware_config:p.error('--hardware-config required')
        return bind(a)
    if a.action in ('prepare','check','capture','import-captures','audit') and not a.stage:p.error('--stage required')
    verify_binding(a)
    from LightGenV2.tasks.t03_saliency import lab_bench as bench
    if a.action in ('check','capture'):return capture(a)
    if a.action=='import-captures':return import_captures(a)
    return getattr(bench,{'init':'initialize'}.get(a.action,a.action))(a)


if __name__=='__main__':main()
