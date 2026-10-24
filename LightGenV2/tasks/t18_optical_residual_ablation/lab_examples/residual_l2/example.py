"""CPU simulation and canonical optical-field export; never opens a device."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'src/LightGenV2/tasks/t18_optical_residual_ablation')]


def verify_files():
    manifest = json.loads((ROOT / 'manifest.json').read_text(encoding='utf-8'))
    for name, identity in manifest['files'].items():
        path = ROOT / name
        assert path.is_file() and path.stat().st_size == identity['bytes'], name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == identity['sha256'], name
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, help='Optional RGB image; otherwise deterministic synthetic input')
    parser.add_argument('--export', type=Path, help='Write logical amplitude/phase arrays, NOT device-ready BMPs')
    parser.add_argument('--self-test', action='store_true', help='Compare packaged Python model with sealed original TorchScript')
    args = parser.parse_args()
    manifest = verify_files()
    import numpy as np
    import torch
    from lab_caltech20_rich_model_20261018 import load_checkpoint, encode_images, DIRECT
    torch.set_num_threads(2)
    model, _ = load_checkpoint(ROOT / 'weights/simulation_L2_rho03.pt',
                              manifest['checkpoint_sha256'], 2, .3, alpha_schema=DIRECT)
    if args.image:
        from PIL import Image
        image = np.asarray(Image.open(args.image).convert('RGB').resize((150, 150), Image.Resampling.BICUBIC)).copy()
        images = torch.from_numpy(image).permute(2, 0, 1)[None]
    else:
        images = torch.arange(3 * 150 * 150).remainder(256).to(torch.uint8).reshape(1, 3, 150, 150)
    if args.self_test:
        # The historical trace was sealed at batch8; the Python model supports batch1.
        images = images.repeat(8, 1, 1, 1)
    with torch.inference_mode():
        amplitude = encode_images(images)
        probability, capture = model(amplitude)
        result = dict(class_index=int(probability.argmax(1)[0]), probabilities=probability.tolist(),
                      alpha=model.alphas(), stage_order=['router', 'E1', 'G1'], camera_opened=False)
        if args.self_test:
            reference = torch.jit.load(str(ROOT / 'reference/source_reference.ts'), map_location='cpu').eval()
            expected, expected_capture = reference(amplitude)
            error = float((probability - expected).abs().max())
            capture_error = float((capture - expected_capture).abs().max())
            assert error <= 3e-6 and capture_error <= 3e-6, (error, capture_error)
            result['self_test'] = dict(passed=True, probability_maxabs=error, capture_maxabs=capture_error,
                                       scope='synthetic input parity, not a new full-dataset accuracy measurement')
        if args.export:
            assert not args.export.exists(), 'Refuse to overwrite an existing export'
            args.export.mkdir(parents=True)
            for key, label in [('router', 'router'), ('0', 'E1'), ('1', 'G1')]:
                value = model.last[key]
                np.savez_compressed(args.export / (label + '.npz'),
                                    amplitude=value['amplitude'].numpy(),
                                    phase_gray=value['phase_gray'].numpy(),
                                    simulation_intensity=value['intensity'].numpy())
            result['export'] = str(args.export)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
