from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    out = Path(args.out) if args.out else run_dir / "transfer_report.md"
    lines = [
        "# Transfer Adaptation Report",
        "",
        f"- run_id: {summary.get('run_id')}",
        f"- source_tasks: {', '.join(summary.get('source_tasks', []))}",
        f"- target_task: {summary.get('target_task')}",
        f"- source_checkpoint: {summary.get('source_checkpoint')}",
        f"- final_target_acc: {summary.get('final_target_metrics', {}).get('acc')}",
        f"- target_prompt_gap: {summary.get('target_prompt_swap_summary', {}).get('target_prompt_gap')}",
        f"- source_retention_max_drop: {summary.get('source_retention_summary', {}).get('max_source_acc_drop')}",
        f"- total_trainable_params: {summary.get('freeze', {}).get('total_trainable_params')}",
        f"- trainable_electronic_params: {summary.get('freeze', {}).get('trainable_electronic_params')}",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()

