"""Task-independent raw Holoeye/DVP control. No model, phase control or image enhancement."""
import argparse
import contextlib
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent


def drivers():
    # The ZIP carries the exact, unmodified driver used on the laboratory PC.
    path = ROOT / 'device_driver.py'
    if not path.exists():
        path = ROOT.parents[1] / 'experiments/hardware_sdk/devices.py'
    spec = importlib.util.spec_from_file_location('control_kit_driver', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_config(path=ROOT / 'config.json'):
    path = Path(path).resolve()
    c = json.loads(path.read_text(encoding='utf-8-sig'))
    if c['amplitude_slm']['driver'] != 'holoeye' or c['camera']['driver'] != 'dvp_subprocess':
        raise ValueError('This kit supports Holoeye + DVP subprocess ONLY.')
    camera = c['camera']
    if camera.get('saved_frame_size_wh') is not None or camera.get('saved_frame_bit_depth') is not None:
        raise ValueError('Raw kit forbids resizing or bit-depth conversion.')
    if camera.get('saved_frame_resize_mode') != 'none':
        raise ValueError('Raw kit requires saved_frame_resize_mode=none.')
    if float(c['settle_delay_ms']) < 0 or float(camera['exposure_us']) <= 0:
        raise ValueError('Invalid settle delay / exposure.')
    if not c['amplitude_slm'].get('preload') or not c['amplitude_slm'].get('wait_until_visible'):
        raise ValueError('Keep preload and wait_until_visible enabled for deterministic display ordering.')
    return c, path.parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


class Controller:
    """Context manager; raw capture is returned as a NumPy uint8/uint16 array."""
    def __init__(self, config_path=ROOT / 'config.json', *, use_slm=True, use_camera=True):
        self.config, base = load_config(config_path)
        d = drivers()
        self.slm = d.build_slm(self.config['amplitude_slm'], base) if use_slm else None
        self.camera = d.build_camera(self.config['camera'], base) if use_camera else None
        self.stack = contextlib.ExitStack()
        self.last_display = None

    def __enter__(self):
        try:
            if self.slm: self.stack.enter_context(self.slm)
            if self.camera: self.stack.enter_context(self.camera)
            print(json.dumps(self.info(), ensure_ascii=False, indent=2), flush=True)
            return self
        except BaseException:
            self.stack.close()
            raise

    def __exit__(self, *args):
        return self.stack.__exit__(*args)

    def info(self):
        return {'slm': self.slm.device_info() if self.slm else None,
                'camera': self.camera.device_info() if self.camera else None}

    def display(self, bmp):
        if self.slm is None: raise RuntimeError('SLM was not enabled.')
        bmp = Path(bmp).resolve()
        with Image.open(bmp) as im:
            if im.format != 'BMP' or im.mode != 'L':
                raise ValueError('Use an 8-bit grayscale (L) BMP, not RGB or 1-bit.')
        t = time.perf_counter()
        self.slm.preload_files([bmp])  # Release prior handle; never cache thousands of images.
        self.slm.display_file(bmp)    # Wait until SDK reports Visible.
        time.sleep(float(self.config['settle_delay_ms']) / 1000)
        self.last_display = {'bmp': str(bmp), 'sha256': digest(bmp),
                             'display_and_settle_ms': (time.perf_counter() - t) * 1000}

    def capture(self, directory, *, frames=1):
        """Create a NEW directory. Raw NPY + lossless TIFF + settings/identity JSON."""
        if self.camera is None: raise RuntimeError('Camera was not enabled.')
        if frames < 1: raise ValueError('frames must be >=1')
        out = Path(directory).resolve()
        out.mkdir(parents=True, exist_ok=False)  # Never overwrite old data.
        report = {'config': self.config, 'devices_at_open': self.info(),
                  'display': self.last_display, 'postprocessing': 'none', 'frames': []}
        def save_report():
            (out / 'capture.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        save_report()
        last = None
        try:
            for i in range(frames):
                p = out / f'{i:04d}.npy'
                t = time.perf_counter()
                self.camera.capture(p)
                elapsed = (time.perf_counter() - t) * 1000
                last = np.load(p, allow_pickle=False)
                if last.ndim != 2 or last.dtype not in (np.uint8, np.uint16):
                    raise ValueError('Expected MONO uint8/uint16; original NPY retained.')
                tif = p.with_suffix('.tif')
                Image.fromarray(last).save(tif, compression='tiff_deflate')
                with Image.open(tif) as im:
                    if not np.array_equal(last, np.asarray(im)): raise RuntimeError('TIFF roundtrip mismatch')
                report['frames'].append({'npy': p.name, 'tif': tif.name,
                    'sha256': {p.name: digest(p), tif.name: digest(tif)},
                    'shape_hw': list(last.shape), 'dtype': str(last.dtype),
                    'min': int(last.min()), 'max': int(last.max()), 'mean': float(last.mean()),
                    'dtype_max_fraction': float(np.mean(last == np.iinfo(last.dtype).max)),
                    'capture_ms': elapsed, 'devices': self.info()})
                save_report()
                print(f'Captured {i+1}/{frames}: {p}', flush=True)
        except BaseException:
            report['interrupted_or_failed'] = True
            save_report()
            raise
        return last


def patterns(c, directory):
    out = Path(directory); out.mkdir(parents=True, exist_ok=False)
    w, h = c['amplitude_slm']['expected_resolution_wh']
    y, x = np.indices((h, w))
    arrays = {'black': np.zeros((h, w), np.uint8), 'gray128': np.full((h, w), 128, np.uint8),
              'white': np.full((h, w), 255, np.uint8),
              'checker64': (((x//64+y//64) % 2)*255).astype(np.uint8)}
    for name, a in arrays.items(): Image.fromarray(a).save(out / (name+'.bmp'))
    print(f'Generated {w}x{h} 8-bit BMPs in {out.resolve()}; no device opened.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['check', 'patterns', 'slm', 'camera', 'capture'])
    p.add_argument('--config', default=str(ROOT / 'config.json'))
    p.add_argument('--bmp'); p.add_argument('--out')
    p.add_argument('--frames', type=int, default=1)
    p.add_argument('--seconds', type=float, default=10)
    a = p.parse_args()
    c, base = load_config(a.config)
    if a.frames < 1 or a.seconds < 0: p.error('frames>=1 and seconds>=0 required')
    if a.action == 'patterns':
        if not a.out: p.error('--out required (new folder)')
        patterns(c, a.out); return
    if a.action == 'check':
        d = drivers()
        slm = d.build_slm(c['amplitude_slm'], base); cam = d.build_camera(c['camera'], base)
        slm.validate_runtime(); cam.validate_runtime()
        dll = slm.binary_folder / 'holoeye_slmdisplaysdk.dll' if slm.binary_folder else None
        if dll is None or not dll.is_file(): raise FileNotFoundError(f'Holoeye native DLL missing: {dll}')
        # Existing driver's check validates executable + bitness; also enforce this ZIP's ABI.
        import subprocess
        version = subprocess.check_output([cam.python_executable, '-c',
            'import sys; print("%s.%s" % sys.version_info[:2])'], text=True).strip()
        if version != '3.6': raise RuntimeError('Bundled dvp.pyd requires CPython 3.6 x64; got '+version)
        print('Paths / DVP Python 3.6 x64 OK. No devices opened; DLL dependencies and hardware still need real tests.')
        return
    if a.action in ('slm', 'capture') and not a.bmp: p.error('--bmp required')
    if a.action in ('camera', 'capture') and not a.out: p.error('--out required (new folder)')
    if a.out and Path(a.out).exists(): raise FileExistsError('Choose a NEW --out directory; old captures are retained.')
    print('Manual PHASE SLM is not controlled. Close competing amplitude/camera applications first.', flush=True)
    with Controller(a.config, use_slm=a.action!='camera', use_camera=a.action!='slm') as hw:
        if a.bmp: hw.display(a.bmp)
        if a.action == 'slm': time.sleep(a.seconds)
        else: hw.capture(a.out, frames=a.frames)


if __name__ == '__main__': main()
