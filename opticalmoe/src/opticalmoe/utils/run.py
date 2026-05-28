from pathlib import Path


def create_run_dir(run_name: str, base_dir: str = "runs") -> Path:
    run_dir = Path(base_dir) / run_name
    (run_dir / "phases").mkdir(parents=True, exist_ok=True)
    (run_dir / "light_fields").mkdir(parents=True, exist_ok=True)
    (run_dir / "sample_outputs").mkdir(parents=True, exist_ok=True)
    return run_dir
