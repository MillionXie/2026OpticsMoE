"""Camera-only diagnostic: never initialize or change either SLM."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import four_image_flow as flow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    flow.BASE_CORNERS = np.float32([[995,172],[4310,172],[4303,3440],[980,3435]])
    rows = []
    with flow.Camera(flow.CAMERA_DLL) as camera:
        for exposure in (3000, 10000, 20000):
            actual = camera.settings(exposure=exposure, gain=1.0)
            frames = [camera.capture() for _ in range(6)]
            image, meta = frames[-1]
            roi = flow.warp(image)
            Image.fromarray(roi).save(args.output / f'camera_only_{exposure}_roi.png')
            Image.fromarray(cv2.resize(image, (1370,912), interpolation=cv2.INTER_AREA)).save(
                args.output / f'camera_only_{exposure}_full4x.png')
            row = {'actual': actual, 'frame_ids': [m['frame_id'] for _,m in frames],
                   'full_p99': float(np.percentile(image,99)), 'full_max': int(image.max()),
                   'roi_mean': float(roi.mean()), 'roi_p99': float(np.percentile(roi,99)),
                   'roi_saturation': float(np.mean(roi == 255))}
            rows.append(row)
            print(json.dumps(row), flush=True)
    flow.write(args.output / 'report.json', {'slm_operations': 0, 'rows': rows})


if __name__ == '__main__':
    main()
