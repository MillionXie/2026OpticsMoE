"""CPU-only rental hardware guard and unchanged native LightGen execution."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_lightgen_rental_20261029 as entry
import inspect
import lgvq_live_frontend as live

def main():
    expected=entry.GPU+', NVIDIA A100-SXM4-80GB, 400.00'
    assert entry.validate_hardware(expected)['power_limit_w']==400
    for bad in (expected.replace(entry.GPU,'foreign'),expected.replace('80GB','40GB'),expected.replace('400.00','250.00')):
        try:entry.validate_hardware(bad)
        except ValueError:pass
        else:raise AssertionError('Reject other hardware or cap')
    assert entry.ROOT.name=='lightgen_rental_20261029_01'
    assert entry.runtime.execute.__module__=='run_lgvq_compute_scope_20261029'
    assert entry.entry.preprocessing.prepare_video.__module__=='lgvq_resize_norm_20261029'
    assert entry.runtime.GPU!=entry.GPU  # Never change laboratory default at import.
    _,counts=entry.frontend_storage.constructor_tree(inspect.getsource(live.LiveFront.__init__))
    assert counts=={'path':1,'container':1,'size_gate':1}
    assert 'torch' not in sys.modules
    print('{"status":"passed","tests":8,"torch_imported":false,"native_execute_unchanged":true}')

if __name__=='__main__':main()
