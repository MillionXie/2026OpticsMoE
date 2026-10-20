"""One bounded same-settings diagnostic after an abrupt whole-sensor white event.

Only two identical zero inputs and the already-published marker/toggle check.
No TEST, exposure/gain search, LUT update, new training, or replacement CCD.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback


def run(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    import cv2
    import numpy as np
    from PIL import Image
    import lab_caltech20_dvp_layerwise_20261019 as m
    m.torch.set_num_threads(2)
    base = a.sequence / 'calibration/admission/admitted_hardware.json'
    digest = m.session_hardware_digest(a.sequence)
    assert m.sha(base) == digest and not m.LOCK.exists() and not a.out.exists()
    cfg = m.read(base)
    assert cfg['exposure_us'] == 60006.0 and cfg['gain'] == 1.0
    assert m.sha(cfg['response_lut']) == cfg['response_lut_SHA256']
    assert m.sha(cfg['phase_lut']) == cfg['phase_LUT_SHA256']
    a.out.mkdir(parents=True)
    m.save(a.out / 'protocol.json', dict(source_git=m.git_head(), source_SHA256=m.sha(__file__),
           base_hardware_SHA256=digest, exposure_us=60006.0, gain=1.0,
           known_patterns=4, TEST_used=False, TRAIN_used=False, no_model_accuracy=True,
           exposure_search=False, LUT_refitted=False, automatic_retry=False))
    with m.LOCK.open('x') as f:
        json.dump(dict(pid=os.getpid(), camera='DVP', session=str(a.sequence),
                       out=str(a.out), kind='bounded_same_settings_signal_diagnostic'), f)
    rows = []
    try:
        warp = cv2.getPerspectiveTransform(np.float32(m.corners(Path(cfg['coordinates']))),
                 np.float32([[0, 0], [477, 0], [477, 477], [0, 477]]))
        with m.Bench(cfg, a.out) as bench:
            flat = a.out / 'flat_phase.bmp'
            Image.fromarray(m.phase_native(np.zeros((478, 478), np.uint8))).save(flat)
            bench.stage(flat)
            drive = np.asarray(m.read(cfg['response_lut'])['drive'], np.uint8)
            zero = a.out / 'zero_input.bmp'
            Image.fromarray(m.native(drive[np.zeros((478, 478), np.uint8)], 'amplitude')).save(zero)
            for i in range(2):
                raw, meta = bench.capture(zero)
                crop = cv2.warpPerspective(raw, warp, (478, 478), flags=cv2.INTER_AREA,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                rawpath, croppath = a.out / f'zero{i}_raw.png', a.out / f'zero{i}_crop.png'
                Image.fromarray(raw).save(rawpath, compress_level=1)
                Image.fromarray(crop).save(croppath, compress_level=1)
                rows.append(dict(tag=f'zero{i}', raw_mean=float(raw.mean()),
                     raw_p99=float(np.percentile(raw, 99)), raw_saturation=float((raw == 255).mean()),
                     crop_mean=float(crop.mean()), crop_p99=float(np.percentile(crop, 99)),
                     crop_saturation=float((crop == 255).mean()), meta=meta,
                     raw_SHA256=m.sha(rawpath), crop_SHA256=m.sha(croppath)))
                m.save(a.out / 'zero_measurements.json', rows)
            # Existing frozen low-gray direction/current-frame algorithm, not a refit.
            proof = m.shared_geometry_check(bench, a.out / 'marker_toggle', cfg, warp)
        m.save(a.out / 'report.json', dict(state='complete', zero_measurements=rows,
             marker_toggle=proof, SDK_released=True, TEST_used=False,
             settings_unchanged=True, automatic_retry=False, completed_ns=time.time_ns()))
    except BaseException as error:
        m.save(a.out / 'failure.json', dict(error=repr(error), traceback=traceback.format_exc(),
             zero_measurements=rows, SDK_released=True, automatic_retry=False))
        raise
    finally:
        owner = m.read(m.LOCK)
        assert owner['pid'] == os.getpid() and owner['kind'] == 'bounded_same_settings_signal_diagnostic'
        m.LOCK.unlink()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--sequence', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    run(p.parse_args())
