"""Same missing21 native baselines, different isolated direct-NVML receiver."""
from pathlib import Path
import run_baselines_fullscope_20261029 as baseline
import nvml_direct_observer_20261029 as observer

ENTRY=Path(__file__).resolve();NATIVE_SOURCES=baseline.source_map;NATIVE_CONFIG=baseline.config

def sources():
    return {**NATIVE_SOURCES(),**{p.relative_to(baseline.REPO).as_posix():baseline.base.sha(p)
        for p in (ENTRY,Path(baseline.__file__),Path(observer.__file__))}}

def config():
    value=NATIVE_CONFIG()
    return dict(value,additional_units=['ms_per_video','estimated_joules_per_video','512_video_continuous_wall_seconds'],
        included=[s.replace('199_internal_gaps','511_internal_gaps') for s in value['included']],
        first512_ids_sha256='48b7fea193abc07f48bec9b091db0351b949387c9b44fbd228399ce3a03b655a',
        observer='direct_NVML_actual_API_receipt_no_stdout_hop_v1',
        observer_change_does_not_change_native_inference_or_250ms_gate=True)

if __name__=='__main__':
    from LightGenV2.tasks.t06_video_quality_assessment import e2e_telemetry_observer as original
    # Only the command that creates the independent receiver changes. Native
    # inference, input preparation, clocks, integration and gates stay original.
    original.observer_command=observer.observer_command
    baseline.ENTRY=ENTRY;baseline.source_map=sources;baseline.config=config
    baseline.main()
