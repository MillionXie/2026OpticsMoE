import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print(" ".join(str(part) for part in cmd))
    subprocess.run(cmd, check=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--master_dir", default="same_input_multitask/results")
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--name", default="same_input_report")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    common = ["--master_dir", args.master_dir, "--out_dir", str(out)]
    if args.run_id:
        common += ["--run_id", args.run_id]
    run([sys.executable, here / "plot_prompt_swap_matrix.py", *common, "--name", f"{args.name}_prompt_swap"])
    run([sys.executable, here / "plot_same_input_predictions.py", *common, "--name", f"{args.name}_same_input"])
    run([sys.executable, here / "plot_expert_usage_heatmap.py", *common, "--name", f"{args.name}_expert_usage"])
    run([sys.executable, here / "plot_prompt_similarity.py", *common, "--name", f"{args.name}_prompt_similarity"])
    (out / "README.md").write_text(f"# {args.name}\n\nGenerated same-input multitask report.\n", encoding="utf-8")


if __name__ == "__main__":
    main()
