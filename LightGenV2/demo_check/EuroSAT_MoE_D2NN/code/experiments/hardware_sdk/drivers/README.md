# Hardware backends

The public entry point is always `devices.build_camera(config, base)`.

- `driver: tucam`: new Dhyana 400BSI V3 / Mosaic TUCam SDK. The vendor files
  are versioned under `../vendor_sdk/camera_tucam_mosaic/` so a cloned bench
  project receives the same Python binding, DLL, demos and manuals.
- `driver: dvp_subprocess`: legacy DVP camera through its vendor-compatible
  Python subprocess.
- `driver: dvp`: legacy in-process DVP adapter.

Backend modules only translate the shared `open/capture/close/device_info`
interface. Acquisition, calibration, and postprocessing must not import a
vendor SDK directly.

The public amplitude-SLM entry point is `devices.build_slm(config, base)`.

- `driver: meadowlark_pcie`: board-indexed Meadowlark Blink PCIe API used by
  the 1024×1024, 17 µm amplitude device. It calls `Create_SDK`, loads an
  explicit calibrated LUT, and performs `Write_image` followed by
  `ImageWriteComplete` for every frame. Input must already be an exact-size
  8-bit grayscale BMP; the driver never resizes, flips, stretches, or applies
  a second LUT.
- `driver: holoeye`: legacy display/GPU SLM backend.
- `driver: manual`: operator-controlled SLM marker.

The repository contains the vendor wrapper DLLs, but the laboratory Windows
computer must also have the Meadowlark Blink Plus PCIe driver/runtime
installed. Select the LUT for the physical device and its operating
temperature in YAML; do not silently fall back to the linear demonstration
LUT.
