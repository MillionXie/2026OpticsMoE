"""One fresh 2048-video score-only LG round; do not dispatch Qwen.

Use --phase suite --lg-only --videos 2048 --output
<T06>/runs/simulation/a100_scoreonly_20261026_01 --deadline <finite Unix>.
Original exclusive suite, finite limits and module deployment gate are reused.
"""

from pathlib import Path

import run_lgvq_lowlevel_20261026 as runner
import lgvq_scoreonly_20261026 as scoreonly

THIS_ENTRY = Path(__file__).resolve()
ORIGINAL_THIN_ENTRY = Path(runner.__file__).resolve()
PREVIOUS_SOURCES = runner.sources


def sources():
    result = PREVIOUS_SOURCES()
    paths = (THIS_ENTRY, ORIGINAL_THIN_ENTRY, Path(scoreonly.__file__).resolve(),
             Path(scoreonly.lowlevel.__file__).resolve(), Path(scoreonly.projection.__file__).resolve())
    for path in paths:
        result[path.relative_to(runner.entry.REPO).as_posix()] = runner.entry.base.sha(path)
    return result


def main(argv=None):
    runner.candidate = scoreonly
    runner.THIS_ENTRY = THIS_ENTRY
    runner.sources = sources
    runner.main(argv)


if __name__ == '__main__':
    main()
