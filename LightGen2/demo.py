"""Run the first sixteen evaluation videos in one LightGen2 optical field."""
import argparse
from pathlib import Path

import torch

if __package__:
    from .infer import PROJECT, iter_video_fields, read_evaluation_csv
    from .model import load_lightgen2
    from .visualize import export_visualizations
else:
    from infer import PROJECT, iter_video_fields, read_evaluation_csv
    from model import load_lightgen2
    from visualize import export_visualizations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, default=PROJECT / 'weights/LightGen2.pt')
    parser.add_argument('--video-root', type=Path, default=PROJECT / 'demo_videos')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, default=PROJECT / 'outputs/demo')
    args = parser.parse_args()
    torch.set_num_threads(4)
    _, paths, _ = read_evaluation_csv(PROJECT / 'test_videos.csv', args.video_root, limit=16)
    _, pixels = next(iter_video_fields(paths))
    model, _ = load_lightgen2(args.weights, args.device)
    records = export_visualizations(model, paths, pixels.to(args.device), args.output)
    for record in records:
        print(f"slot {record['slot']:2d}  {record['video']}  MOS {record['MOS']:.6f}")
    print('Saved complete masks, optical fields and predictions to ' + str(args.output))


if __name__ == '__main__':
    main()
