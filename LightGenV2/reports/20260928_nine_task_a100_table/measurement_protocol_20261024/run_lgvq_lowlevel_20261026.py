"""Single LG-only 2048-video candidate, original exclusive suite/guard reused.

Example (same original lab environment, finite deadline required):
  python run_lgvq_lowlevel_20261026.py --phase suite --lg-only --videos 2048 \
    --output <T06>/runs/simulation/a100_lowlevel_capture_20261026_01 \
    --deadline <finite-Unix-seconds>

Keep the published MEASUREMENT_PUBLISHED_COMMIT environment from the existing
launcher. Qwen is never dispatched by this thin entry. Original projection
deployment SHA still binds the original module; the candidate has a separate
source identity, and all three new/original runtime files enter suite freezing.
"""

import argparse
import json
import os
from pathlib import Path

import run_corrected_boundaries_20261026 as entry
import lgvq_physical_projection_20261026 as projection
import lgvq_lowlevel_capture_20261026 as candidate

ORIGINAL_ENTRY = Path(entry.__file__).resolve()
PREVIOUS_SOURCES = entry.sources
THIS_ENTRY = Path(__file__).resolve()


def sources():
    result = PREVIOUS_SOURCES()
    for path in (ORIGINAL_ENTRY, THIS_ENTRY, Path(projection.__file__).resolve(),
                 Path(candidate.__file__).resolve()):
        result[path.relative_to(entry.REPO).as_posix()] = entry.base.sha(path)
    return result


def ensure_original_projection_deployment(root):
    """Before loading any model, bind the unchanged original-module SHA gate."""
    path = root/'LG_module_deployment.json'
    digest = entry.base.sha(Path(projection.__file__))
    if path.exists():
        receipt = json.loads(path.read_text(encoding='utf-8'))
        if receipt['sha256'] != digest:
            raise RuntimeError('Do not overwrite an inconsistent original deployment')
        return
    receipt = dict(
        sha256=digest, source=str(Path(projection.__file__).resolve()),
        published_commit=os.environ['MEASUREMENT_PUBLISHED_COMMIT'],
        original_projection_unchanged=True,
        candidate_has_separate_lowlevel_capture_identity=True,
    )
    with path.open('x', encoding='utf-8') as handle:
        json.dump(receipt, handle, indent=2)
        handle.write('\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['suite', 'worker'], required=True)
    parser.add_argument('--system', choices=['lg'], default='lg')
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--videos', type=int, choices=[2048], default=2048)
    parser.add_argument('--deadline', type=float, required=True)
    parser.add_argument('--lg-only', action='store_true', default=True)
    args = parser.parse_args(argv)
    if args.repeat != 1:
        raise ValueError('The isolated candidate runs one fresh LG round only')
    args.output = args.output.resolve()
    entry.ENTRY = THIS_ENTRY
    entry.N = args.videos
    entry.sources = sources
    projection.worker = candidate.worker
    if args.phase == 'suite':
        entry.suite(args)
    else:
        ensure_original_projection_deployment(args.output.parent)
        entry.worker(args)


if __name__ == '__main__':
    main()
