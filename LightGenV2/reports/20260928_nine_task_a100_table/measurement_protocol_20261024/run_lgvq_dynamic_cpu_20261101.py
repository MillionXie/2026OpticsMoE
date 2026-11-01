"""Fresh 3x32 LG fields: continuous dynamic FIFO16 CPU and 1886Hz projection."""
from pathlib import Path
import run_lgvq_resize_norm_20261029 as wrapper
import run_lgvq_compute_scope_20261029 as runtime
import lgvq_compute_scope_20261029 as scope
import lgvq_resize_norm_20261029 as prep
import lgvq_dynamic_cpu_20261101 as policy

ENTRY=Path(__file__).resolve()


def sources():
    return {**wrapper.sources(),**{p.relative_to(runtime.REPO).as_posix():runtime.sha(p)
        for p in (ENTRY,Path(policy.__file__),Path(policy.optical.__file__))}}


def write(path,value):
    if path.name=='compute_identity.json':
        value=dict(value,profile='LG_DYNAMIC_FIFO16_CONTINUOUS_CPU_1886_20261101',
            CPU_resize_and_normalization_included=True,CPU_FIFO_workers=16,
            CPU_jobs_independent_of_GPU_completion=True,SLM_hz=1886,
            physical_optics_measured=False,backend_cross_group_pipeline_modelled=True)
        wrapper.NATIVE_WRITE(path,value)
    else:wrapper.write(path,value)


def configure():
    import cv2
    cv2.setNumThreads(1)
    runtime.ENTRY=ENTRY;runtime.sources=sources;runtime.write=write;runtime.verify=wrapper.verify
    runtime.SCHEDULING_POLICY=policy
    scope.prepare_cpu_worker=prep.prepare_cpu_worker;scope.prepare_video=policy.prepare_video
    scope.patch_video=policy.patch_video;scope.summarize_compute=policy.summarize_compute


if __name__=='__main__':
    configure();runtime.main()
