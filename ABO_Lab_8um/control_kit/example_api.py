"""Run from the kit directory AFTER editing config.json and placing input.bmp."""
from pathlib import Path
from control import Controller

if __name__ == '__main__':
    # Keep devices open while changing inputs. The phase SLM is never touched.
    with Controller('config.json') as hw:
        hw.display(Path('input.bmp'))
        raw = hw.capture(Path('captures/api_example'), frames=3)
        print('Last raw CCD frame:', raw.shape, raw.dtype)
    # Camera-only: Controller('config.json', use_slm=False)
    # SLM-only: Controller('config.json', use_camera=False)
