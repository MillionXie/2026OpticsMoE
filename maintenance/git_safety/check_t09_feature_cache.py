"""Read-only CPU reconstruction of a SHA-bound T09 TRAIN/VAL feature cache.

No TEST, optimizer, CUDA, downloads, cache writes or accuracy evaluation.
Historical acquisition device identity is not inferred from numerical agreement.
"""
import argparse
import hashlib
import json
from pathlib import Path

RTOL = 1e-5
ATOL = 1e-5


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def reconstruct(checkpoint, images):
    import torch
    from LightGenV2.tasks.t09_multimodal_matching.vision import VisionEncoder
    torch.set_num_threads(2)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = VisionEncoder(state['model']['head.weight'].shape[0],
                          tuple(state.get('architecture', {}).get('widths', [16, 32, 64])))
    model.load_state_dict(state['model'], strict=True)
    model.requires_grad_(False).eval()
    with torch.inference_mode():
        return torch.cat([model(torch.from_numpy(images[start:start + 64].copy()))[0]
                          for start in range(0, len(images), 64)]).numpy()


def check(data_root, checkpoint, cache, historical_source):
    import numpy as np
    from LightGenV2.tasks.t09_multimodal_matching import vision
    data_root, checkpoint, cache = map(Path, (data_root, checkpoint, cache))
    sidecar = cache.with_suffix('.json')
    manifest_path = data_root / 'manifest.json'
    source = Path(vision.__file__)
    historical_source = Path(historical_source)
    paths = [sidecar, manifest_path, checkpoint, cache, source, historical_source]
    paths += [data_root / (split + '_images.npz') for split in ('train', 'val')]
    before = {str(p): sha(p) for p in paths}
    meta = json.loads(sidecar.read_text(encoding='utf8'))
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    result = dict(read_only=True, device='cpu', gradients=False, test_accessed=False,
                  accuracy_evaluated=False, cache_written=False,
                  rtol=RTOL, atol=ATOL, files_sha256=before, splits={}, errors=[])
    gates = [(checkpoint, meta['checkpoint_sha256']),
             (manifest_path, meta['data_manifest_sha256']), (cache, meta['cache_sha256'])]
    gates += [(data_root / (s + '_images.npz'), manifest['files'][s + '_images.npz'])
              for s in ('train', 'val')]
    for path, expected in gates:
        if before[str(path)] != expected:
            result['errors'].append('Bound SHA mismatch: ' + str(path))
    normalized = lambda p: hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    result['frontend_source_equal_lf'] = normalized(source) == normalized(historical_source)
    if not result['frontend_source_equal_lf']:
        result['errors'].append('Frontend source differs from historical runtime')
    if result['errors']:
        result['passed'] = False
        return result
    with np.load(cache, allow_pickle=False) as saved:
        for split in ('train', 'val'):
            with np.load(data_root / (split + '_images.npz'), allow_pickle=False) as archive:
                images = archive['images']
            expected = saved[split]
            if hashlib.sha256(expected.tobytes()).hexdigest() != meta['features'][split + '_feature_sha256']:
                result['errors'].append('Bound array SHA mismatch: ' + split)
                continue
            if expected.shape != (len(images), 128) or expected.dtype != np.float32 or not np.isfinite(expected).all():
                result['errors'].append('Invalid saved feature shape/dtype/finite values: ' + split)
                continue
            actual = reconstruct(checkpoint, images)
            if actual.shape != expected.shape or not np.isfinite(actual).all():
                result['errors'].append('Invalid reconstructed feature shape/finite values: ' + split)
                continue
            absolute = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
            close = np.isclose(actual, expected, rtol=RTOL, atol=ATOL)
            result['splits'][split] = dict(images=len(images), shape=list(actual.shape),
                exact_bytes=actual.tobytes() == expected.tobytes(), allclose=bool(close.all()),
                max_absolute_error=float(absolute.max()), mean_absolute_error=float(absolute.mean()),
                outside_tolerance=int((~close).sum()))
            if not close.all():
                result['errors'].append('Numerical reconstruction differs: ' + split)
    result['original_assets_unchanged'] = before == {str(p): sha(p) for p in paths}
    if not result['original_assets_unchanged']:
        result['errors'].append('Original assets changed during check')
    result['passed'] = not result['errors']
    result['full_task_reproduction_proven'] = False
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data-root', 'checkpoint', 'cache', 'historical-source'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = check(args.data_root, args.checkpoint, args.cache, args.historical_source)
    print(json.dumps(result, indent=2))
    return int(not result['passed'])


if __name__ == '__main__':
    raise SystemExit(main())
