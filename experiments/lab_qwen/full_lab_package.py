"""Build one complete, short-path Qwen optical laboratory ZIP.

This combines the sealed quick210 bundle with current calibration, agreement,
four-stage, capture, evaluation, and plotting assets.  Historical calibration
payloads and historical command documents are intentionally excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


ARCHIVE_ROOT = "experiments/lab_qwen"
DEFAULT_NAME = "qwen_full_lab.zip"
REQUIRED_FRESNEL = {
    "A_WHITE.bmp",
    "P1_POINT.bmp",
    "P4_POINT.bmp",
    "P9_POINT.bmp",
    "P1_CROSS.bmp",
    "P4_CROSS.bmp",
    "P9_CROSS.bmp",
    "manifest.json",
    "targets.csv",
}


@dataclass(frozen=True)
class Entry:
    archive_path: str
    category: str
    source_path: Path | None = None
    source_zip: Path | None = None
    source_member: str | None = None
    literal: bytes | None = None
    transform: str | None = None

    def read(self) -> bytes:
        if self.literal is not None:
            return self.literal
        if self.source_path is not None:
            data = self.source_path.read_bytes()
        else:
            if self.source_zip is None or self.source_member is None:
                raise RuntimeError(f"Entry has no source: {self.archive_path}")
            with zipfile.ZipFile(self.source_zip) as archive:
                data = archive.read(self.source_member)
        if self.transform == "compact_478_to_native_1024_bmp":
            from PIL import Image

            compact = Image.open(io.BytesIO(data)).convert("L")
            if compact.size != (478, 478):
                raise RuntimeError(
                    f"Expected 478x478 compact amplitude, got {compact.size}: "
                    f"{self.source_member}"
                )
            native = Image.new("L", (1024, 1024), color=0)
            native.paste(compact, (273, 273))
            buffer = io.BytesIO()
            native.save(buffer, format="BMP")
            return buffer.getvalue()
        if self.transform is not None:
            raise RuntimeError(f"Unknown entry transform: {self.transform}")
        return data


def _safe_name(value: str) -> str:
    name = PurePosixPath(value.replace("\\", "/"))
    if name.is_absolute() or ".." in name.parts or not name.parts:
        raise ValueError(f"Unsafe archive path: {value!r}")
    return name.as_posix()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _put(entries: dict[str, Entry], entry: Entry, *, replace: bool = False) -> None:
    name = _safe_name(entry.archive_path)
    if name in entries and not replace:
        raise RuntimeError(f"Duplicate package path: {name}")
    entries[name] = Entry(
        archive_path=name,
        category=entry.category,
        source_path=entry.source_path,
        source_zip=entry.source_zip,
        source_member=entry.source_member,
        literal=entry.literal,
        transform=entry.transform,
    )


def _map_directory(
    entries: dict[str, Entry],
    source: Path,
    destination: str,
    category: str,
    *,
    replace: bool = False,
) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"Required {category} directory is missing: {source}")
    files = sorted(path for path in source.rglob("*") if path.is_file())
    if not files:
        raise RuntimeError(f"Required {category} directory is empty: {source}")
    for path in files:
        _put(
            entries,
            Entry(
                f"{destination}/{path.relative_to(source).as_posix()}",
                category,
                source_path=path,
            ),
            replace=replace,
        )


def _add_ready_bmps_from_compact(
    entries: dict[str, Entry],
    *,
    source_session: Path,
    destination: str,
    category: str,
) -> int:
    compact_files = sorted(source_session.glob("*/compact_amplitude/*.png"))
    for compact in compact_files:
        stage_name = compact.parent.parent.name
        _put(
            entries,
            Entry(
                f"{destination}/{stage_name}/amplitude_to_play/{compact.stem}.bmp",
                category,
                source_path=compact,
                transform="compact_478_to_native_1024_bmp",
            ),
        )
    return len(compact_files)


def _formal_entries(entries: dict[str, Entry], formal_zip: Path) -> None:
    if not formal_zip.is_file():
        raise FileNotFoundError(f"Sealed formal ZIP is missing: {formal_zip}")
    with zipfile.ZipFile(formal_zip) as archive:
        names = set(archive.namelist())
        required = {
            "payload/checkpoint/ema_best_train_loss_checkpoint.pt",
            "payload/quick210/manifest.csv",
            "payload/quick210/04_language_global/offline_downstream/cache.pt",
            "payload/quick210/04_language_global/offline_downstream/downstream_state.pt",
        }
        missing = required.difference(names)
        if missing:
            raise RuntimeError(f"Formal ZIP is incomplete: {sorted(missing)}")
        for member in sorted(names):
            if member.endswith("/"):
                continue
            destination: str | None = None
            category = "formal_runtime"
            if member == "payload/checkpoint/ema_best_train_loss_checkpoint.pt":
                destination = f"{ARCHIVE_ROOT}/model/ema.pt"
                category = "formal_checkpoint"
            elif member.startswith("payload/quick210/"):
                destination = f"{ARCHIVE_ROOT}/last/{member[len('payload/quick210/'):]}"
                category = "last_stage_quick210"
            elif member.startswith("payload/four_phase_export/"):
                destination = (
                    f"{ARCHIVE_ROOT}/reference/four_phase_export/"
                    f"{member[len('payload/four_phase_export/'):]}"
                )
                category = "formal_phase_reference"
            elif member.startswith("reference/training_evidence/"):
                destination = f"{ARCHIVE_ROOT}/{member}"
                category = "training_evidence"
            elif member.startswith("reference/fixed_simulation_report/"):
                destination = f"{ARCHIVE_ROOT}/{member}"
                category = "fixed_simulation_report"
            elif member.startswith("experiments/"):
                destination = member
                category = "runtime_or_vendor"
            elif member == "bundle_manifest.json":
                destination = f"{ARCHIVE_ROOT}/reference/sealed_bundle_manifest.json"
                category = "formal_provenance"
            # README, historical calibration and copied historical project
            # source/docs are deliberately excluded from the current-only ZIP.
            if destination is not None:
                _put(
                    entries,
                    Entry(
                        destination,
                        category,
                        source_zip=formal_zip,
                        source_member=member,
                    ),
                )
        compact_prefix = "payload/quick210/04_language_global/compact_amplitude/"
        compact_members = sorted(
            member
            for member in names
            if member.startswith(compact_prefix) and member.lower().endswith(".png")
        )
        if len(compact_members) != 210:
            raise RuntimeError(
                f"Formal ZIP must contain 210 compact quick inputs, got {len(compact_members)}"
            )
        for member in compact_members:
            stem = Path(member).stem
            _put(
                entries,
                Entry(
                    f"{ARCHIVE_ROOT}/last/04_language_global/amplitude_to_play/{stem}.bmp",
                    "last_stage_ready_bmp",
                    source_zip=formal_zip,
                    source_member=member,
                    transform="compact_478_to_native_1024_bmp",
                ),
            )


def _current_source_files(repo_root: Path) -> Iterable[Path]:
    roots = (
        "experiments/hardware_sdk",
        "experiments/lab_qwen",
        "experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5",
        "experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust",
        "experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval",
        "experiments/qwen3_vl_embedding_2b_caltech101_electronic_retrieval",
        "experiments/qwen3_vl_embedding_2b_caltech101_robust_hybrid_retrieval",
        "experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval",
    )
    suffixes = {".py", ".yaml", ".yml", ".md", ".txt"}
    forbidden_parts = {
        "__pycache__",
        "runs",
        "hardware_sessions",
        "lab_bundles",
        "validation_bundles",
        "lab_full_bundles",
        "generated",
        "tests",
        "data",
        "work",
        "results",
    }
    for relative in roots:
        root = repo_root / relative
        if not root.is_dir():
            raise FileNotFoundError(f"Required source tree is missing: {root}")
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            if any(part in forbidden_parts for part in path.relative_to(root).parts):
                continue
            yield path


def create_bundle(
    *,
    repo_root: Path,
    formal_zip: Path,
    fresnel_dir: Path,
    dual_dir: Path,
    exposure_dir: Path,
    agreement_session: Path,
    four_session: Path,
    output: Path,
    overwrite: bool,
) -> dict[str, Any]:
    root = repo_root.resolve()
    entries: dict[str, Entry] = {}
    _formal_entries(entries, formal_zip.resolve())

    observed_fresnel = {path.name for path in fresnel_dir.iterdir() if path.is_file()}
    missing_fresnel = REQUIRED_FRESNEL.difference(observed_fresnel)
    if missing_fresnel:
        raise RuntimeError(f"Current Fresnel payload is incomplete: {sorted(missing_fresnel)}")
    _map_directory(entries, fresnel_dir, f"{ARCHIVE_ROOT}/calib/fresnel", "fresnel_v4")
    _put(
        entries,
        Entry(
            f"{ARCHIVE_ROOT}/calib/fresnel/amplitude_manifest.csv",
            "fresnel_v4",
            literal=b"amplitude_bmp\r\nA_WHITE.bmp\r\n",
        ),
    )
    _map_directory(entries, dual_dir, f"{ARCHIVE_ROOT}/calib/dual", "dual_slm_k1")
    _map_directory(entries, exposure_dir, f"{ARCHIVE_ROOT}/calib/exposure", "exposure_32x3")
    _map_directory(entries, agreement_session, f"{ARCHIVE_ROOT}/agree", "agreement_session")
    _map_directory(entries, four_session, f"{ARCHIVE_ROOT}/four", "four_stage_initial")
    agreement_ready = _add_ready_bmps_from_compact(
        entries,
        source_session=agreement_session,
        destination=f"{ARCHIVE_ROOT}/agree",
        category="agreement_ready_bmp",
    )
    four_ready = _add_ready_bmps_from_compact(
        entries,
        source_session=four_session,
        destination=f"{ARCHIVE_ROOT}/four",
        category="four_stage_ready_bmp",
    )
    if agreement_ready == 0:
        raise RuntimeError("Agreement session contains no compact amplitudes to reconstruct")
    if four_ready == 0:
        raise RuntimeError("Four-stage session contains no compact amplitudes to reconstruct")

    for path in _current_source_files(root):
        relative = path.relative_to(root).as_posix()
        _put(
            entries,
            Entry(relative, "current_source", source_path=path),
            replace=True,
        )

    # The all-white frame must be physically all 255, not merely named so.
    try:
        from PIL import Image
        import numpy as np
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Pillow and NumPy are required to validate calibration BMPs") from error
    white = np.asarray(Image.open(fresnel_dir / "A_WHITE.bmp"))
    if white.shape != (1024, 1024) or not bool(np.all(white == 255)):
        raise RuntimeError("A_WHITE.bmp must be 1024x1024 with every pixel equal to 255")

    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists (use --overwrite): {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True
    ) as archive:
        for name, entry in sorted(entries.items()):
            data = entry.read()
            archive.writestr(name, data)
            records.append(
                {
                    "path": name,
                    "category": entry.category,
                    "bytes": len(data),
                    "sha256": _sha256_bytes(data),
                }
            )
        manifest = {
            "schema_version": 1,
            "type": "qwen_complete_short_path_optical_lab_bundle",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "archive_root": ARCHIVE_ROOT,
            "formal_source_zip_sha256": _sha256_file(formal_zip),
            "workflow": {
                "dual_slm_calibration": True,
                "fresnel_ccd_calibration": "full-white amplitude; point and cross phase sets",
                "brightness_calibration": "32 gray levels x 3 frames",
                "sim_to_real_agreement": True,
                "last_stage_quick210": True,
                "four_stage": "initial stage included; subsequent stages depend on preceding measured CCD",
            },
            "file_count_excluding_manifest": len(records),
            "files": records,
        }
        archive.writestr(
            f"{ARCHIVE_ROOT}/PACKAGE_MANIFEST.json",
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        )
    digest = _sha256_file(output)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{digest}  {output.name}\n", encoding="ascii"
    )
    result = verify_bundle(output)
    result.update({"sha256": digest, "bytes": output.stat().st_size})
    return result


def verify_bundle(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError("ZIP contains duplicate paths")
        manifest_name = f"{ARCHIVE_ROOT}/PACKAGE_MANIFEST.json"
        manifest = json.loads(archive.read(manifest_name))
        for record in manifest["files"]:
            data = archive.read(record["path"])
            if len(data) != int(record["bytes"]) or _sha256_bytes(data) != record["sha256"]:
                raise RuntimeError(f"ZIP verification failed: {record['path']}")
        required = {
            f"{ARCHIVE_ROOT}/COMMANDS.md",
            f"{ARCHIVE_ROOT}/config/hardware.yaml",
            f"{ARCHIVE_ROOT}/calib/fresnel/A_WHITE.bmp",
            f"{ARCHIVE_ROOT}/calib/fresnel/P4_POINT.bmp",
            f"{ARCHIVE_ROOT}/calib/dual/k1_pair_manifest.json",
            f"{ARCHIVE_ROOT}/agree/agreement_manifest.json",
            f"{ARCHIVE_ROOT}/last/04_language_global/offline_downstream/cache.pt",
            f"{ARCHIVE_ROOT}/model/ema.pt",
        }
        missing = required.difference(names)
        if missing:
            raise RuntimeError(f"ZIP required-data check failed: {sorted(missing)}")
        four_amplitudes = [
            name for name in names if name.startswith(f"{ARCHIVE_ROOT}/four/") and "/amplitude_to_play/" in name and name.lower().endswith(".bmp")
        ]
        if not four_amplitudes:
            raise RuntimeError("ZIP contains no ready-to-play four-stage amplitude BMP")
        last_amplitudes = [
            name for name in names if name.startswith(f"{ARCHIVE_ROOT}/last/04_language_global/amplitude_to_play/") and name.lower().endswith(".bmp")
        ]
        if len(last_amplitudes) != 210:
            raise RuntimeError(f"ZIP must contain 210 last-stage inputs, got {len(last_amplitudes)}")
    return {
        "path": str(path),
        "files": len(names),
        "last_stage_inputs": len(last_amplitudes),
        "four_stage_amplitude_bmps": len(four_amplitudes),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify-zip")
    mode.add_argument("--create", action="store_true")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--formal-zip")
    parser.add_argument("--fresnel-dir")
    parser.add_argument("--dual-dir")
    parser.add_argument("--exposure-dir", default="experiments/lab_qwen/calib/exposure")
    parser.add_argument("--agreement-session")
    parser.add_argument("--four-session")
    parser.add_argument("--output", default=DEFAULT_NAME)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.verify_zip:
        report = verify_bundle(Path(args.verify_zip).expanduser().resolve())
    else:
        required = {
            "formal_zip": args.formal_zip,
            "fresnel_dir": args.fresnel_dir,
            "dual_dir": args.dual_dir,
            "agreement_session": args.agreement_session,
            "four_session": args.four_session,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            parser.error(f"creation requires {', '.join('--' + name.replace('_', '-') for name in missing)}")
        root = Path(args.repo_root).expanduser().resolve()
        report = create_bundle(
            repo_root=root,
            formal_zip=Path(args.formal_zip).expanduser().resolve(),
            fresnel_dir=Path(args.fresnel_dir).expanduser().resolve(),
            dual_dir=Path(args.dual_dir).expanduser().resolve(),
            exposure_dir=(root / args.exposure_dir).resolve(),
            agreement_session=Path(args.agreement_session).expanduser().resolve(),
            four_session=Path(args.four_session).expanduser().resolve(),
            output=Path(args.output).expanduser().resolve(),
            overwrite=args.overwrite,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
