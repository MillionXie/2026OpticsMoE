"""Read-only rank72 machine path inspection, without loading any SDK or model."""
import argparse
import hashlib
import json
from pathlib import Path


def inspect(machine_config, phase_sdk, phase_lut, amplitude_sdk=None):
    config_path = Path(machine_config).resolve()
    config = json.loads(config_path.read_text(encoding='utf-8-sig'))
    base = config_path.parent
    def resolve(value):
        if not value:
            raise ValueError('Required machine path is empty')
        p = Path(value)
        return (p if p.is_absolute() else base / p).resolve()
    camera = resolve(config['camera']['sdk_root'])
    amp = resolve(amplitude_sdk or config['amplitude_slm']['sdk_path'])
    binary = resolve(config['amplitude_slm']['binary_folder'])
    phase = Path(phase_sdk).resolve()
    lut = Path(phase_lut).resolve()
    required = {
        'camera_wrapper': camera / 'demo/base_dll/bin/CEasyCapS.dll',
        'camera_transport': camera / 'cti/x86_64/cxplink_gentl.cti',
        'amplitude_sdk': amp,
        'amplitude_binary_folder': binary,
        'amplitude_native_library': binary / 'holoeye_slmdisplaysdk.dll',
        'phase_wrapper': phase / 'Blink_C_wrapper.dll',
        'phase_lut': lut,
    }
    directories = {'amplitude_sdk', 'amplitude_binary_folder'}
    paths = {key: {'path': str(p), 'present': p.is_dir() if key in directories else p.is_file()}
             for key, p in required.items()}
    wrappers = (amp / 'slmdisplaysdk.py', amp / 'slmdisplaysdk/__init__.py',
                amp / 'holoeye/slmdisplaysdk/__init__.py')
    wrapper = next((p for p in wrappers if p.is_file()), None)
    paths['amplitude_python_wrapper'] = {'candidates': [str(p) for p in wrappers],
        'path': str(wrapper) if wrapper else None, 'present': wrapper is not None}
    if wrapper is not None:
        required['amplitude_python_wrapper'] = wrapper
    for key, p in required.items():
        if key not in directories and paths[key]['present']:
            with p.open('rb') as stream:
                paths[key]['sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'passed': all(r['present'] for r in paths.values()), 'paths': paths,
            'read_only': True, 'devices_opened': False,
            'effective_capture_settings': {'exposure_us': 400, 'gain': 'Gain_X4', 'wait_ms': 240},
            'original_config_settings': {'exposure_us': config['camera'].get('exposure_us'),
                'gain': config['camera'].get('gain'), 'wait_ms': config.get('settle_delay_ms')},
            'scope': 'Declared installed paths and current file hashes only; no DLL load, display, camera, license, ABI, signal or historical acquisition identity verification'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('machine-config', 'phase-sdk', 'phase-lut'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--amplitude-sdk', type=Path)
    args = parser.parse_args()
    try:
        result = inspect(args.machine_config, args.phase_sdk, args.phase_lut, args.amplitude_sdk)
    except (OSError, ValueError, KeyError, TypeError) as error:
        result = {'passed': False, 'read_only': True, 'devices_opened': False,
                  'error_type': type(error).__name__, 'error': str(error)}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
