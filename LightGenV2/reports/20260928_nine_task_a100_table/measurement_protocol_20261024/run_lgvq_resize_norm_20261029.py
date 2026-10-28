"""User-approved resize/normalization-inclusive LG measurement before baselines."""
from pathlib import Path
import run_lgvq_compute_scope_20261029 as runtime
import lgvq_compute_scope_20261029 as scope
import lgvq_resize_norm_20261029 as preprocessing
import lgvq_patch_layout_20261029 as layout

ENTRY=Path(__file__).resolve();NATIVE_SOURCES=runtime.sources;NATIVE_WRITE=runtime.write

def sources():
    return {**NATIVE_SOURCES(),**{p.relative_to(runtime.REPO).as_posix():runtime.sha(p)
        for p in (ENTRY,Path(runtime.__file__),Path(preprocessing.__file__),Path(layout.__file__))}}

def write(path,value):
    if path.name=='input_preparation.json':
        value=dict(value,included='native resize, processor resize, native rescale/normalize and exact patch recomputed in every formal field',
            excluded='file reading/decoding, RGB conversion/65% crop and quality-pixel input preparation',
            online_input='unresized original RGB crop buffers, no normalized pixels or learned features cached')
    elif path.name=='compute_identity.json':
        value=dict(value,profile='LG_RESIZE_NORMALIZATION_PATCH_COMPUTE_SCOPE_20261029',CPU_resize_and_normalization_included=True)
    NATIVE_WRITE(path,value)

if __name__=='__main__':
    import cv2
    cv2.setNumThreads(1)
    runtime.ENTRY=ENTRY;runtime.sources=sources;runtime.write=write
    scope.prepare_cpu_worker=preprocessing.prepare_cpu_worker;scope.prepare_video=preprocessing.prepare_video
    scope.patch_video=preprocessing.patch_video;scope.summarize_compute=preprocessing.summarize_compute
    runtime.main()
