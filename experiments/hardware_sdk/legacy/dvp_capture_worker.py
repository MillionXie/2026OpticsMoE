"""Legacy-Python-compatible persistent DVP capture worker.

Keep this file syntax compatible with Python 3.5.  It is intentionally
standalone and communicates with the Python-3.11 orchestrator using JSONL.
"""

from __future__ import print_function

import argparse
import json
import os
import sys

import numpy as np


def configure_camera(camera, module, args):
    """Apply saved vendor settings, then deterministic command-line overrides."""
    if args.config_file:
        camera.LoadConfig(args.config_file)
    camera.TriggerState = False
    if args.resolution_mode is not None:
        camera.ResolutionModeSel = args.resolution_mode
    if args.device_roi_xywh is not None:
        roi = camera.Roi
        roi.X, roi.Y, roi.W, roi.H = args.device_roi_xywh
        camera.Roi = roi
    if args.auto_exposure is not None:
        camera.AeOperation = (
            module.AeOperation.AE_OP_CONTINUOUS
            if args.auto_exposure == "on"
            else module.AeOperation.AE_OP_OFF
        )
    if args.anti_flicker_hz is not None:
        values = {
            0: module.AntiFlick.ANTIFLICK_DISABLE,
            50: module.AntiFlick.ANTIFLICK_50HZ,
            60: module.AntiFlick.ANTIFLICK_60HZ,
        }
        camera.AntiFlick = values[args.anti_flicker_hz]
    if args.exposure_us is not None:
        camera.Exposure = args.exposure_us
    if args.analog_gain is not None:
        camera.AnalogGain = args.analog_gain


def safe_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "value"):
        return value.value
    return str(value)


def camera_info(camera):
    result = {}
    for name in (
        "TriggerState",
        "Exposure",
        "AnalogGain",
        "AeOperation",
        "AntiFlick",
        "ResolutionModeSel",
    ):
        try:
            result[name] = safe_value(getattr(camera, name))
        except Exception:
            result[name] = None
    try:
        roi = camera.Roi
        result["device_roi_xywh"] = [int(roi.X), int(roi.Y), int(roi.W), int(roi.H)]
    except Exception:
        result["device_roi_xywh"] = None
    return result


def frame_to_array(frame_buffer, module):
    frame, buffer_value = frame_buffer
    dtype = np.uint8 if frame.bits == module.Bits.BITS_8 else np.uint16
    if module.ImageFormat.FORMAT_MONO <= frame.format <= module.ImageFormat.FORMAT_BAYER_RG:
        channels = 1
    elif frame.format in (module.ImageFormat.FORMAT_BGR24, module.ImageFormat.FORMAT_RGB24):
        channels = 3
    elif frame.format in (module.ImageFormat.FORMAT_BGR32, module.ImageFormat.FORMAT_RGB32):
        channels = 4
    else:
        raise RuntimeError("unsupported DVP image format: {0}".format(frame.format))
    value = np.frombuffer(buffer_value, dtype=dtype).reshape(
        frame.iHeight, frame.iWidth, channels
    )
    if channels == 1:
        return value[..., 0].copy()
    rgb = value[..., :3]
    if not (np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 0], rgb[..., 2])):
        raise RuntimeError("camera returned color data; configure raw MONO output")
    return rgb[..., 0].copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdk-path", default=None)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--timeout-ms", type=int, default=4000)
    parser.add_argument("--config-file", default=None)
    parser.add_argument("--auto-exposure", choices=("on", "off"), default=None)
    parser.add_argument("--exposure-us", type=float, default=None)
    parser.add_argument("--analog-gain", type=float, default=None)
    parser.add_argument("--anti-flicker-hz", type=int, choices=(0, 50, 60), default=None)
    parser.add_argument("--device-roi-xywh", type=int, nargs=4, default=None)
    parser.add_argument("--resolution-mode", type=int, default=None)
    parser.add_argument("--warmup-frames", type=int, default=3)
    parser.add_argument("--discard-frames-after-display", type=int, default=1)
    args = parser.parse_args()
    dll_directory = None
    if args.sdk_path:
        sdk_path = os.path.abspath(args.sdk_path)
        if sys.platform.startswith("win"):
            os.environ["PATH"] = sdk_path + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                dll_directory = os.add_dll_directory(sdk_path)
        sys.path.insert(0, sdk_path)
    import dvp

    devices = dvp.Refresh()
    if not devices:
        raise RuntimeError("DVP Refresh() found no camera")
    camera = dvp.Camera(args.camera_index)
    configure_camera(camera, dvp, args)
    camera.Start()
    for _ in range(args.warmup_frames):
        camera.GetFrame(args.timeout_ms)
    print(json.dumps({"ready": True, "device": camera_info(camera)}))
    sys.stdout.flush()
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if request.get("command") == "close":
                    break
                if request.get("command") != "capture":
                    raise RuntimeError("unknown command")
                for _ in range(args.discard_frames_after_display):
                    camera.GetFrame(args.timeout_ms)
                array = frame_to_array(camera.GetFrame(args.timeout_ms), dvp)
                np.save(request["path"], array)
                response = {"ok": True, "shape": list(array.shape), "dtype": str(array.dtype)}
            except Exception as exc:
                response = {"ok": False, "error": "{0}: {1}".format(type(exc).__name__, exc)}
            print(json.dumps(response))
            sys.stdout.flush()
    finally:
        camera.Stop()
        camera.Close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
