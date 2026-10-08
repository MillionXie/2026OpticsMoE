"""Read-only DVP camera enumeration with repeated SDK refreshes."""
import ctypes as C
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DLL = ROOT.parent / "小相机/SDK二次开发包/DVP2  SDK 中性版本/DVP2 SDK/library/Visual C++/bin/x64/DVPCamera64.dll"
dll = C.WinDLL(str(DLL))
dll.dvpRefresh.argtypes = [C.POINTER(C.c_uint32)]
dll.dvpRefresh.restype = C.c_int
for i in range(5):
    count = C.c_uint32()
    status = int(dll.dvpRefresh(C.byref(count)))
    print(json.dumps(dict(attempt=i, status=status, camera_count=count.value)), flush=True)
    time.sleep(1)
