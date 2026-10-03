"""Generate four self-contained engineering projects from one audited code source.

Generated projects are artifacts, not four separately maintained model forks.
Teacher checkpoint is reference-only; group checkpoints must come from new runs.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from study import GROUPS, make_config, sha256, write_json

PROJECTS = {
    "r0_post": "01_baseline_post",
    "r1_ccd_post": "02_ccd_post",
    "r2_ccd_dc_post": "03_ccd_dc_post",
    "r3_ccd_dc_intrain": "04_ccd_dc_intrain",
}
EXCLUDE = {"runs", "assets", "releases", "projects", "__pycache__", ".pytest_cache", ".git", "paths.local.json"}


def verify_teacher(package):
    pin = json.loads((ROOT / "reference/source_manifest.json").read_text(encoding="utf-8"))
    if sha256(package / "release.json") != pin["source_release_json_sha256"]:
        raise ValueError("Teacher package is not pinned final_v2")
    hashes = json.loads((package / "SHA256.json").read_text(encoding="utf-8"))
    for relative, expected in hashes.items():
        path = (package / relative).resolve()
        if not path.is_relative_to(package.resolve()) or not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Teacher integrity mismatch: {relative}")
    if sha256(package / "weights/best_checkpoint.pt") != pin["checkpoint_sha256"]:
        raise ValueError("Teacher checkpoint identity mismatch")
    return pin


def build(output, *, teacher=None, checkpoints=None):
    import yaml
    output = output.resolve()
    if output.exists():
        raise ValueError("Use a new output directory; existing projects/results are never overwritten")
    if ROOT.is_relative_to(output):
        raise ValueError("Output must not contain the source tree")
    if teacher:
        verify_teacher(teacher)
    checkpoints = checkpoints or {}
    # Validate all supplied weights before creating any project.
    for group, checkpoint in checkpoints.items():
        import torch
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved.get("study_group") != group or "state_dict" not in saved:
            raise ValueError(f"Not a newly trained {group} checkpoint")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    output.mkdir(parents=True)
    projects = []
    for group, name in PROJECTS.items():
        destination = output / name
        destination.mkdir()
        for source in ROOT.iterdir():
            if source.name in EXCLUDE:
                continue
            if source.is_dir():
                shutil.copytree(source, destination / source.name,
                                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))
            else:
                shutil.copy2(source, destination / source.name)
        for directory in ("weights", "inputs", "runs", "assets"):
            (destination / directory).mkdir(exist_ok=True)
        checkpoint = checkpoints.get(group)
        if checkpoint:
            shutil.copy2(checkpoint, destination / "weights/best_checkpoint.pt")
        if teacher:
            shutil.copytree(teacher, destination / "teacher_reference",
                            ignore=shutil.ignore_patterns("runs", "__pycache__", "*.pyc", ".pytest_cache"))
        metadata = {"group": group, "condition": GROUPS[group], "source_commit": commit,
                    "status": "trained_checkpoint_included" if checkpoint else "code_ready_group_not_trained",
                    "reference_checkpoint_is_ablation_result": False,
                    "checkpoint_sha256": sha256(checkpoint) if checkpoint else None,
                    "teacher_reference_included": bool(teacher)}
        write_json(destination / "project.json", metadata)
        for purpose in ("train", "deployment"):
            raw = make_config(group, purpose=purpose, output=Path("runs/simulation") / group)
            # Snapshot is readable and portable; run.py resolves runtime paths locally.
            reference = yaml.safe_load((ROOT / "reference/teacher_v2.yaml").read_text(encoding="utf-8"))
            raw["data"] = reference["data"]
            raw["output_dir"] = f"runs/simulation/{group}"
            (destination / "configs" / f"{purpose}.yaml").write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        readme = f"""# {name}: 时间一致性消融工程

条件锁定：`{group}`。完整模型实现位于 runtime/，噪声适配位于 ccd.py。
训练配置：configs/train.yaml；共同设备评价配置：configs/deployment.yaml。
这两份 YAML 是展开快照；实际训练以 run 的 resolved.yaml 与 study.json 为准。

## 入口

```text
python -I project.py plan
python -I project.py preflight --paths paths.local.json
python -I project.py smoke --device cpu
python -I project.py train --paths paths.local.json --device cuda --allow-uncalibrated-noise
python -I project.py evaluate --checkpoint weights/best_checkpoint.pt --paths paths.local.json --device cuda
python -I project.py reference --device cpu --fields 1 --output runs/reference_smoke.json
python -I project.py infer --device cpu --fields 0 --output runs/group_deployment.json
```

reference 使用 teacher_reference/weights/best_checkpoint.pt，只核验旧导师模型，
不代表本组结果、不用于本组初始化。weights/best_checkpoint.pt 只有本组重新训练后才有；
当前状态见 project.json。inputs/ 留给本组部署输入，旧导师的35场在 teacher_reference/inputs/。
infer 使用本组新PT和冻结35场，不需要完整训练缓存；新PT未训练时明确拒绝运行。
训练只保存 best/last；不会自动开始训练或采集。

CCD 默认是未标定的 Poisson + 零均值 Gaussian pilot，不是 ACCEL 参数复刻。
四组唯一允许的差异是 CCD、相干直流、训练网格；常规架构/参数量相同。
硬件接口及暂存导出见 hardware.py、build_lab_package.py；实际采集仍需核验 SDK/LUT/ROI/振幅合同。
完整协议与边界见 ENGINEERING_PROTOCOL.md。
"""
        shutil.copy2(destination / "README.md", destination / "ENGINEERING_PROTOCOL.md")
        (destination / "README.md").write_text(readme, encoding="utf-8")
        write_json(destination / "SHA256.json", {str(p.relative_to(destination)).replace("\\", "/"): sha256(p)
                   for p in destination.rglob("*") if p.is_file() and p != destination / "SHA256.json"})
        projects.append(metadata | {"directory": name})
    write_json(output / "projects_manifest.json", {"source_commit": commit, "projects": projects})
    (output / "README.md").write_text(
        "# 四组时间一致性消融\n\n每个子目录均自包含，可单独复制运行。四份由同一代码源生成，不手工维护四套模型。\n"
        "旧导师参考权重若打包，位于各工程 teacher_reference/weights；不是四组消融权重。\n"
        + "\n".join(f"- {name}: {group}" for group, name in PROJECTS.items()) + "\n", encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher-package", type=Path)
    parser.add_argument("--checkpoints", type=Path, help="JSON group -> newly trained checkpoint path")
    args = parser.parse_args()
    checkpoints = {g: Path(p) for g, p in json.loads(args.checkpoints.read_text()).items()} if args.checkpoints else {}
    if set(checkpoints) - set(GROUPS):
        parser.error("Unknown checkpoint group")
    print(build(args.output, teacher=args.teacher_package, checkpoints=checkpoints))


if __name__ == "__main__":
    main()
