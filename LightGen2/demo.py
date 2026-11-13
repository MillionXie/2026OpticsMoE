"""Run sixteen distinct demonstration videos in one LightGen2 optical field."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

if __package__:
    from .infer import PROJECT, iter_video_fields
    from .model import load_lightgen2
    from .visualize import export_visualizations
else:
    from infer import PROJECT, iter_video_fields
    from model import load_lightgen2
    from visualize import export_visualizations


def create_demo_videos(directory):
    """Create reproducible synthetic clips with different motion and texture."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / f'video_{index + 1:02d}.mp4' for index in range(16)]
    yy, xx = np.mgrid[:256, :256]
    for index, path in enumerate(paths):
        if path.is_file():
            continue
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 12, (256, 256))
        if not writer.isOpened():
            raise RuntimeError('Cannot create demo video: ' + str(path))
        random = np.random.default_rng(100 + index)
        try:
            for frame_index in range(24):
                shift = frame_index * (index % 4 + 1)
                image = np.stack(((xx + shift + index * 13) % 256,
                                  (yy * 2 + index * 23) % 256,
                                  ((xx + yy) // 2 + shift) % 256), -1).astype(np.uint8)
                center = (int(60 + (frame_index * 4 + index * 9) % 140),
                          int(128 + 45 * np.sin(frame_index / 5 + index)))
                cv2.circle(image, center, 18 + index % 5, (240, 240, 240), -1)
                cv2.rectangle(image, (45, 55), (90 + index * 4, 85), (20, 40, 210), -1)
                noise = random.normal(0, index % 4 * 3, image.shape)
                image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
                writer.write(image)
        finally:
            writer.release()
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, default=PROJECT / 'weights/LightGen2.pt')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, default=PROJECT / 'outputs/demo')
    args = parser.parse_args()
    paths = create_demo_videos(PROJECT / 'demo_videos')
    _, pixels = next(iter_video_fields(paths))
    model, _ = load_lightgen2(args.weights, args.device)
    records = export_visualizations(model, paths, pixels.to(args.device), args.output)
    for record in records:
        print(f"slot {record['slot']:2d}  {record['video']}  MOS {record['MOS']:.6f}")
    print('Saved complete masks, optical fields and predictions to ' + str(args.output))


if __name__ == '__main__':
    main()
