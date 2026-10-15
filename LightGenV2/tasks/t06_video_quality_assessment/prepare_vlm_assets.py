"""CPU-only public fixed-revision asset download; verify official blob/LFS hashes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from . import smolvlm2_temporal_baseline as shared


def validate_profile(profile):
    if len(profile["model_revision"]) != 40 or not profile["official_assets"]:
        raise ValueError("A fixed official revision and explicit assets are required")
    for name, spec in profile["official_assets"].items():
        if Path(name).name != name or name.startswith("."):
            raise ValueError("Only explicit model asset basenames")
        if "sha256" in spec:
            if len(spec["sha256"]) != 64 or spec.get("size", 0) <= 0:
                raise ValueError("Official LFS SHA and size required")
        elif len(spec.get("git_blob_sha1", "")) != 40:
            raise ValueError("Official Git blob SHA required")


def main():
    import requests
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", choices=("https://huggingface.co", "https://hf-mirror.com"), default="https://huggingface.co")
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text())
    validate_profile(profile)
    output = args.output.resolve()
    shared.new_output(output)
    output.mkdir(parents=True)
    args.model.mkdir(parents=True, exist_ok=True)
    identity = dict(model_id=profile["model_id"], revision=profile["model_revision"],
        profile_sha256=shared.frozen.sha256(args.profile),
        source_sha256=shared.frozen.sha256(Path(__file__)), command=sys.argv,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=shared.REPO, text=True).strip(),
        gpu_initialized=False, created_at_unix=time.time())
    shared.frozen.write_json(output / "run_identity.json", identity)
    try:
        for name in sorted(profile["official_assets"], key=lambda n: n.endswith("model.safetensors")):
            spec = profile["official_assets"][name]
            path = args.model / name
            if path.exists():
                if not shared.asset_valid(path, spec):
                    raise ValueError("Existing asset differs; preserve: " + name)
                continue
            partial = path.with_suffix(path.suffix + ".partial")
            if partial.exists():
                raise FileExistsError("Partial preserved, no blind retry: " + name)
            url = f"{args.endpoint}/{profile['model_id']}/resolve/{profile['model_revision']}/{name}"
            written = 0
            print("[download] " + name, flush=True)
            with requests.get(url, stream=True, timeout=(20, 60)) as response:
                response.raise_for_status()
                with partial.open("xb") as stream:
                    for chunk in response.iter_content(4*1024*1024):
                        if chunk:
                            stream.write(chunk)
                            written += len(chunk)
                            if written < 8*1024*1024 or written % (64*1024*1024) == 0:
                                shared.frozen.write_json(output / "status.json", dict(status="downloading", file=name,
                                    bytes=written, updated_at_unix=time.time(), gpu_initialized=False))
            if not shared.asset_valid(partial, spec):
                raise ValueError("Official asset SHA mismatch; preserve partial: " + name)
            partial.replace(path)
        hashes = {name: shared.frozen.sha256(args.model/name) for name in profile["official_assets"]}
        shared.frozen.write_json(output / "model_asset_manifest.json", dict(identity=identity, sha256=hashes,
            verified_official_git_blobs_and_lfs_sha=True))
        shared.frozen.write_json(output / "status.json", dict(status="complete", updated_at_unix=time.time(), gpu_initialized=False))
    except BaseException as error:
        shared.frozen.write_json(output / "status.json", dict(status="failed", error=repr(error), updated_at_unix=time.time()))
        raise


if __name__ == "__main__":
    main()
