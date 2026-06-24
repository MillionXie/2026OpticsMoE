from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, Path]


def repo_root_from_file(file_path: PathLike) -> Path:
    return Path(file_path).resolve().parents[2]


def make_run_dir(
    experiments_root: PathLike,
    family: str,
    run_name: str,
    exist_ok: bool = True,
) -> Path:
    run_dir = Path(experiments_root) / family / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=exist_ok)
    for child in [
        "checkpoints",
        "metrics",
        "diagnostics",
        "figures/light_fields",
        "figures/prompt",
        "figures/phase_masks",
        "figures/detector_outputs",
        "figures/samples",
        "summary_for_master",
    ]:
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    return run_dir


def write_text(path: PathLike, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def relative_or_str(path: Optional[Path]) -> str:
    return "" if path is None else str(path)
