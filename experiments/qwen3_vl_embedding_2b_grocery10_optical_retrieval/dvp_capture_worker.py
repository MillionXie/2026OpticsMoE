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
    parser.add_argument("--sdk-path", required=True)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--timeout-ms", type=int, default=4000)
    parser.add_argument("--config-file", default=None)
    args = parser.parse_args()
    sys.path.insert(0, os.path.abspath(args.sdk_path))
    import dvp

    devices = dvp.Refresh()
    if not devices:
        raise RuntimeError("DVP Refresh() found no camera")
    camera = dvp.Camera(args.camera_index)
    camera.TriggerState = False
    if args.config_file:
        camera.LoadConfig(args.config_file)
    camera.Start()
    print("READY")
    sys.stdout.flush()
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if request.get("command") == "close":
                    break
                if request.get("command") != "capture":
                    raise RuntimeError("unknown command")
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
