"""Build a self-contained T06 hardware-control and fine-tuning ZIP."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from .project import CURRENT_PROFILE, REPO_ROOT, TASK_DIR, load_profile, repo_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=CURRENT_PROFILE)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--bench", choices=("legacy", "shs"), default="legacy")
    parser.add_argument("--target", choices=("spatial", "temporal"))
    parser.add_argument("--source-root", default=str(REPO_ROOT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-fields", type=int, default=0)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--shs-code-update", action="store_true", help="Small SHA-checked runtime update; no weights or data")
    parser.add_argument("--base-manifest", help="SHA256.json of the exact installed release for a checked runtime update")
    args = parser.parse_args()
    if args.shs_code_update:
        import hashlib,json,zipfile
        if not args.output:parser.error('Code update requires --output ZIP')
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO_ROOT,text=True).strip()
        base='8e869473787f4ffceb2a6a77f4430b94c206f459'
        prefix='LightGenV2/tasks/t06_video_quality_assessment/'
        files={};manifest={'source_commit':commit,'base_runtime_commit':base,'files':{}}
        baseline=json.loads(Path(args.base_manifest).read_text(encoding='utf-8')) if args.base_manifest else None
        if baseline is not None:
            manifest['base_manifest_sha256']=hashlib.sha256(Path(args.base_manifest).read_bytes()).hexdigest()
            manifest['base_runtime_commit']=None
        for name in ['lab_bench.py','lab_exposure_session.py','lab_runtime.py']:
            path=prefix+name;data=subprocess.check_output(['git','show',commit+':'+path],cwd=REPO_ROOT)
            old=subprocess.run(['git','show',base+':'+path],cwd=REPO_ROOT,capture_output=True)
            dest='runtime/'+path;files[dest]=data
            old_digest=baseline.get(dest) if baseline is not None else (hashlib.sha256(old.stdout).hexdigest() if old.returncode==0 else None)
            manifest['files'][dest]={'sha256':hashlib.sha256(data).hexdigest(),'base_sha256':old_digest}
        output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(output,'x',zipfile.ZIP_DEFLATED) as z:
            for name,data in files.items():z.writestr(name,data)
            z.writestr('runtime_update.json',json.dumps(manifest,indent=2))
        print(json.dumps({'path':str(output),'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'source_commit':commit}))
        return 0
    if args.bench == "shs":
        if args.target is None or args.output is None:
            parser.error("SHS requires --target and --output (new directory; ZIP is adjacent)")
        from .lab_bundle import build
        build(args)
        return 0
    profile = load_profile(args.profile)
    backend = profile["backend"]
    artifact = profile["artifacts"]
    checkpoint = (
        repo_path(artifact.get("canonical_checkpoint", artifact["checkpoint"]))
        if args.checkpoint is None
        else Path(args.checkpoint).expanduser().resolve()
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Canonical checkpoint is not present on this machine: {checkpoint}\n"
            "Use --checkpoint or run this command on the source training server."
        )
    safe_profile = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in args.profile
    )
    output = (
        TASK_DIR
        / "releases"
        / f"{datetime.now().strftime('%Y%m%d')}_{safe_profile}_full_lab.zip"
        if args.output is None
        else Path(args.output).expanduser().resolve()
    )
    command = [
        sys.executable,
        "-m",
        f"{backend['package']}.build_delivery_packages",
        "lab",
        "--repo-root",
        str(REPO_ROOT),
        "--config",
        str(repo_path(backend["config"])),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--guide",
        str(repo_path(backend["lab_guide"])),
    ]
    print(subprocess.list2cmdline(command), flush=True)
    return subprocess.run(command, cwd=REPO_ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
