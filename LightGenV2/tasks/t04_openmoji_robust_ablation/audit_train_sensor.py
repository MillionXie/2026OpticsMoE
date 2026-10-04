"""No-SDK TRAIN-only sensor statistics. Single frames are NOT temporal calibration.

Measure existing raw CCD without modifying exposure, model, or images. Spatial
residuals mix optical mismatch and sensor noise: never label them pure read/shot
noise. Repeated fixed-input/dark/flat captures are needed for that separation.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stride', type=int, default=20)
    args = parser.parse_args()
    assert args.stride > 0 and not args.output.exists()
    contract = json.loads((args.run/'contract.json').read_text(encoding='utf-8'))
    assert contract['scope'] == 'train', 'Do not fit sensor statistics on TEST'
    layers = {}
    for folder in sorted((args.run/'ccd').iterdir()):
        if not folder.is_dir():
            continue
        receipts = sorted(folder.glob('*.json'))
        assert len(receipts) == 1000
        rows = []
        for receipt in receipts[::args.stride]:
            j = json.loads(receipt.read_text(encoding='utf-8'))
            assert j['p99'] >= 15
            image = receipt.with_suffix('.png')
            x = np.asarray(Image.open(image), dtype=np.float64) / 255.
            assert x.ndim == 2
            # Common structure gives apparent spatial correlations, not temporal noise.
            gx = np.diff(x, axis=1)
            gy = np.diff(x, axis=0)
            rows.append(dict(id=receipt.stem, image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                mean=float(x.mean()), p01=float(np.quantile(x,.01)),
                p99=float(np.quantile(x,.99)), saturation=float((x>=1).mean()),
                horizontal_difference_rms=float(np.sqrt(np.mean(gx*gx))),
                vertical_difference_rms=float(np.sqrt(np.mean(gy*gy))),
                receipt_saturation=j['saturation_fraction'],phase_sha256=j['phase_sha256']))
        layers[folder.name] = dict(receipts=len(receipts), sampled=len(rows), rows=rows,
            median={k: float(np.median([r[k] for r in rows])) for k in
                    ('mean','p01','p99','saturation','horizontal_difference_rms','vertical_difference_rms')})
    assert len(layers)==6
    report=dict(status='complete', scope='train', no_sdk=True, no_model_updates=True,
        temporal_read_shot_parameters_identifiable=False,
        caveat='Single-frame spatial variation includes optical structure; not calibrated read/shot noise',
        contract_sha256=hashlib.sha256((args.run/'contract.json').read_bytes()).hexdigest(),
        stride=args.stride,layers=layers)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v['median'] for k,v in layers.items()}),flush=True)


if __name__=='__main__':
    main()
