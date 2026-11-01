"""Four scopes, each fresh triple, three real warm fields per process."""
import argparse
from pathlib import Path
import re
import time
import run_lgvq_dynamic_cpu_20261101 as prior
import lgvq_front_scopes_20261101 as policy
import lgvq_mean_pipeline_20261101 as mean

runtime = prior.runtime
ENTRY = Path(__file__).resolve()
VARIANT = None


def sources():
    return {**prior.sources(), **{p.relative_to(runtime.REPO).as_posix(): runtime.sha(p)
                                 for p in (ENTRY, Path(policy.__file__), Path(mean.__file__))}}


def write(path, value):
    if path.name == 'input_preparation.json':
        value = dict(value, variant=VARIANT,
                     external_preparation_is_reference_validation_not_formal_timing=True,
                     included_in_estimate=['decode', 'RGB', 'crop'][:VARIANT]+['resize', 'normalization', 'patch'],
                     excluded_from_estimate=['decode', 'RGB', 'crop'][VARIANT:]+['quality', 'H2D', 'CPU_dispatch_and_off_CPU_wait'],
                     online_input='native RGB crops' if VARIANT == 0 else 'original video files; excluded RGB/crop still execute as dependencies')
    elif path.name == 'compute_identity.json':
        value = dict(value, profile='LG_FOUR_CUMULATIVE_MEAN_SCOPE_20261101', variant=VARIANT,
                     CPU_FIFO_workers=16, explicit_extra_warm_fields=3,
                     CPU_jobs_independent_of_GPU_completion=True, SLM_hz=1886,
                     mean_stage_pipeline_is_estimate=True, load_imbalance_charged=False,
                     frontend_decoder=policy.DECODER_RECEIPT,
                     CPU_off_CPU_wait_excluded_from_estimate=True)
    prior.wrapper.NATIVE_WRITE(path, value)


def configure(variant):
    global VARIANT
    VARIANT = variant
    prior.configure(); policy.configure(variant)
    runtime.ENTRY = ENTRY; runtime.sources = sources; runtime.write = write
    runtime.SCHEDULING_POLICY = policy
    prior.scope.prepare_video = policy.prepare_video
    prior.scope.patch_video = policy.patch_video
    prior.scope.summarize_compute = policy.summarize_compute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', type=int, choices=range(4), required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    parser.add_argument('--source-receipt-sha256', required=True)
    args = parser.parse_args(); args.root = args.root.resolve(); args.output = args.output.resolve()
    if args.root.parent != runtime.SIM or args.root.name != 'lgvq_compute_scope_20261029_15':
        raise ValueError('New root15 only; old evidence immutable')
    if args.output.parent != args.root or not re.fullmatch(r'v'+str(args.variant)+r'_repeat0[123]', args.output.name):
        raise ValueError('Exactly four predetermined triples')
    if not 0 < args.deadline-time.time() <= 14400:
        raise ValueError('Finite four-hour maximum')
    args.pipe = '/DATA/DATA1/guest3/.cache/'+args.root.name
    configure(args.variant); runtime.worker(args)


if __name__ == '__main__':
    main()
