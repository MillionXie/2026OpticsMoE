import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--master_dir", default="dataset_switching/results")
    parser.add_argument("--out_dir", default="dataset_switching/figures/reports/dataset_switching")
    parser.add_argument("--name", default="dataset_switching")
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = Path(__file__).resolve().parent
    for script, name in [
        ("plot_task_accuracy.py", f"{args.name}_task_accuracy"),
        ("plot_prompt_swap_matrix.py", f"{args.name}_prompt_swap"),
        ("plot_expert_usage_heatmap.py", f"{args.name}_expert_usage"),
    ]:
        subprocess.run([sys.executable, str(base / script), "--master_dir", args.master_dir, "--out_dir", str(out), "--name", name], check=False)
    (out / "README.md").write_text("Dataset-switching report figures generated from master tables.\n", encoding="utf-8")


if __name__ == "__main__":
    main()
