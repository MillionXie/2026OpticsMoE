# Camera backends

The public entry point is always `devices.build_camera(config, base)`.

- `driver: tucam`: new Dhyana 400BSI V3 / Mosaic TUCam SDK. The local vendor
  files remain under `../ccd_2_mosaic/` and are ignored by Git.
- `driver: dvp_subprocess`: legacy DVP camera through its vendor-compatible
  Python subprocess.
- `driver: dvp`: legacy in-process DVP adapter.

Backend modules only translate the shared `open/capture/close/device_info`
interface. Acquisition, calibration, and postprocessing must not import a
vendor SDK directly.
