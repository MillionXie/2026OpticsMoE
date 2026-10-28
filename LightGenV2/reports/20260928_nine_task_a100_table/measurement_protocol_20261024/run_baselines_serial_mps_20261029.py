"""One finite missing15 recovery after baseline02 terminal MPS-query failure.

Only guard check transaction serialization differs. Keep original native model,
full-input timing, direct NVML receiver, identity/power/exclusion gates. Retain
Qwen50, Qwen100 and GLM50 triples; never adopt the failed GLM100 point.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

import run_baselines_direct_nvml_20261029 as direct
from serialized_mps_guard_20261029 import serialized_guard_class
import serialized_mps_guard_20261029 as guard_module

baseline = direct.baseline
ENTRY = Path(__file__).resolve()
ROOT = baseline.SIM / 'baselines_fullscope_20261029_03'
RETAINED = baseline.SIM / 'baselines_fullscope_20261029_02'
RETAINED_NAMES = [f'{m}_cap{c}_repeat{r:02d}' for m, c in [('qwen', 100), ('glm', 50)] for r in (1, 2, 3)]
NATIVE_PLAN = baseline.missing_plan
NATIVE_SOURCES = direct.sources
NATIVE_CONFIG = direct.config
NATIVE_INITIALIZE = baseline.base.initialize_profile
NATIVE_PREDECESSOR = baseline.verify_predecessor


def sources():
    return {**NATIVE_SOURCES(), **{p.relative_to(baseline.REPO).as_posix(): baseline.base.sha(p)
        for p in (ENTRY, Path(guard_module.__file__))}}


def missing_plan(profile, models=None, caps=None, repeats=None):
    return [(m, c, r) for m, c, r in NATIVE_PLAN(profile)
        if f'{m}_cap{c}_repeat{r:02d}' not in RETAINED_NAMES]


def config():
    return dict(NATIVE_CONFIG(), retained_baseline02_points=RETAINED_NAMES,
        guard_query_serialization='per_worker_RLock_whole_original_check_no_retry_v1',
        native_inference_timing_and_rejection_gates_unchanged=True)


def initialize_profile(args):
    legacy = NATIVE_INITIALIZE(args)
    legacy.engine.MPSGuard = serialized_guard_class(legacy.engine.MPSGuard)
    return legacy


def verify_retained():
    state = baseline.base.read(RETAINED / 'suite_status.json')
    if (state['status'] != 'stopped' or state['completed'] != RETAINED_NAMES
            or not state.get('mps_released')):
        raise ValueError('Exact terminal baseline02 with six accepted points required')
    for key in ('pid', 'child_pid'):
        if Path('/proc', str(state[key])).exists():
            raise ValueError('Stopped predecessor PID still exists')
    failed = baseline.base.read(RETAINED / 'glm_cap100_repeat01/status.json')
    if failed['status'] != 'failed' or 'MPS control failed' not in failed.get('error', ''):
        raise ValueError('Exact preserved MPS-query failure required')
    result = subprocess.run(['nvidia-cuda-mps-control'], input='get_server_list\n',
        text=True, capture_output=True, timeout=5,
        env=dict(os.environ, CUDA_MPS_PIPE_DIRECTORY='/DATA/DATA1/guest3/.cache/' + RETAINED.name))
    if result.returncode == 0:
        raise ValueError('Predecessor private MPS still responds')
    evidence = []
    for name in RETAINED_NAMES:
        leaf = RETAINED / name
        report = baseline.base.read(leaf / 'report.json')
        audit = baseline.base.read(leaf / 'cpu_audit.json')
        if (audit['status'] != 'passed' or audit['measured_videos'] != 512
                or report['batch'] != 1 or report['explicit_warmup_forwards'] != 0):
            raise ValueError('Accepted retained scope required')
        for filename, key in [('report.json', 'report_sha256'), ('calls.csv', 'calls_sha256'),
                ('telemetry.csv', 'power_sha256')]:
            if baseline.base.sha(leaf / filename) != audit[key]:
                raise ValueError('Retained raw SHA changed: ' + name + '/' + filename)
        evidence.append(dict(name=name, report_sha256=audit['report_sha256'],
            audit_sha256=baseline.base.sha(leaf / 'cpu_audit.json')))
    return evidence


def verify_predecessor(lg_root):
    NATIVE_PREDECESSOR(lg_root)
    evidence = verify_retained()
    with ROOT.with_name(ROOT.name + '.retained_baseline02.json').open('x', encoding='utf8') as stream:
        json.dump(evidence, stream, indent=2, ensure_ascii=True)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--phase', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--suite-root', type=Path)
    args, _ = parser.parse_known_args()
    if (args.suite_root or args.output).resolve() != ROOT.resolve():
        raise ValueError('Only new canonical recovery03; no blind root loop')
    from LightGenV2.tasks.t06_video_quality_assessment import e2e_telemetry_observer as original
    original.observer_command = direct.observer.observer_command
    baseline.ENTRY = ENTRY
    baseline.source_map = sources
    baseline.config = config
    baseline.missing_plan = missing_plan
    baseline.verify_predecessor = verify_predecessor
    baseline.base.initialize_profile = initialize_profile
    baseline.main()


if __name__ == '__main__':
    main()
