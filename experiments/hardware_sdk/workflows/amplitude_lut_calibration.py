"""Dense global amplitude-SLM response calibration and LUT linearization.

The input LUT is never overwritten.  A dense fixed-exposure CCD scan is made
with the selected base LUT, the higher-dynamic-range monotonic branch around
the measured dark state is fitted with isotonic regression, and its inverse is
interpolated onto all 256 requested gray levels.  For optical-network inputs,
``field_amplitude`` is the default contract: requested gray ``g`` targets CCD
intensity ``(g/255)^2``.  ``linear_intensity`` remains available as an explicit
diagnostic alternative.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .calibration_common import load_yaml_config, resolve_path
from .roi_calibration import run_exposure


TRANSFER_MODES = {"field_amplitude", "linear_intensity"}
LUT_SIZE = 256
DAC_MIN = 0
DAC_MAX = 4095


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_global_lut(path: str | Path) -> np.ndarray:
    """Read a strict 256-line Meadowlark ``gray DAC`` global LUT."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Base LUT is missing: {source}")
    rows: list[tuple[int, int]] = []
    for line_number, raw in enumerate(
        source.read_text(encoding="ascii").splitlines(), start=1
    ):
        line = raw.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(
                f"LUT line {line_number} must contain exactly 'gray DAC': {raw!r}"
            )
        try:
            gray, dac = (int(value) for value in fields)
        except ValueError as error:
            raise ValueError(f"LUT line {line_number} is not integer data") from error
        rows.append((gray, dac))
    expected = list(range(LUT_SIZE))
    observed = [gray for gray, _ in rows]
    if observed != expected:
        raise ValueError(
            "A global 8-bit Meadowlark LUT must contain exactly gray 0..255 once "
            f"in order; observed {len(rows)} rows"
        )
    values = np.asarray([dac for _, dac in rows], dtype=np.int64)
    if np.any(values < DAC_MIN) or np.any(values > DAC_MAX):
        raise ValueError(f"LUT DAC values must remain in {DAC_MIN}..{DAC_MAX}")
    return values


def write_global_lut(path: str | Path, values: Sequence[int], *, overwrite: bool) -> Path:
    destination = Path(path).expanduser().resolve()
    array = np.asarray(values, dtype=np.int64)
    if array.shape != (LUT_SIZE,):
        raise ValueError(f"Generated LUT must have 256 DAC values, got {array.shape}")
    if np.any(array < DAC_MIN) or np.any(array > DAC_MAX):
        raise ValueError(f"Generated DAC values leave {DAC_MIN}..{DAC_MAX}")
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Generated LUT already exists: {destination}. Use "
            "--overwrite-generated-lut only after checking its provenance."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(f"{gray} {int(dac)}\r\n" for gray, dac in enumerate(array))
    destination.write_bytes(payload.encode("ascii"))
    # Reparse the actual bytes that the vendor SDK will receive.
    if not np.array_equal(read_global_lut(destination), array):
        raise RuntimeError("Generated LUT failed round-trip validation")
    return destination


def _read_response(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"SLM response CSV is missing: {source}")
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"SLM response CSV is empty: {source}")
    exposures = {float(row["exposure_us"]) for row in rows}
    if len(exposures) != 1:
        raise ValueError(f"LUT fit requires one fixed exposure, got {sorted(exposures)}")
    gray = np.asarray([int(row["gray_value"]) for row in rows], dtype=np.float64)
    energy = np.asarray([float(row["integrated_energy"]) for row in rows])
    if gray[0] != 0 or gray[-1] != 255 or np.any(np.diff(gray) <= 0):
        raise ValueError("LUT response grays must be increasing and include 0/255")
    if len(gray) < 16:
        raise ValueError("LUT calibration requires at least 16 measured gray points")
    if not np.isfinite(energy).all() or float(np.ptp(energy)) <= 0.0:
        raise ValueError("LUT response energy must be finite with non-zero range")
    saturation = max(float(row["saturated_pixel_fraction"]) for row in rows)
    return gray, energy, {
        "path": str(source),
        "sha256": _sha256(source),
        "exposure_us": next(iter(exposures)),
        "measured_points": len(rows),
        "maximum_saturated_pixel_fraction": saturation,
    }


def isotonic_non_decreasing(values: Sequence[float]) -> np.ndarray:
    """Unweighted pool-adjacent-violators fit, expanded to the input length."""

    y = np.asarray(values, dtype=np.float64)
    if y.ndim != 1 or y.size == 0 or not np.isfinite(y).all():
        raise ValueError("isotonic input must be one finite non-empty vector")
    levels: list[float] = []
    weights: list[int] = []
    starts: list[int] = []
    ends: list[int] = []
    for index, value in enumerate(y):
        levels.append(float(value))
        weights.append(1)
        starts.append(index)
        ends.append(index + 1)
        while len(levels) >= 2 and levels[-2] > levels[-1]:
            combined_weight = weights[-2] + weights[-1]
            combined_level = (
                levels[-2] * weights[-2] + levels[-1] * weights[-1]
            ) / combined_weight
            levels[-2:] = [combined_level]
            weights[-2:] = [combined_weight]
            ends[-2:] = [ends[-1]]
            starts.pop()
    fitted = np.empty_like(y)
    for level, start, end in zip(levels, starts, ends):
        fitted[start:end] = level
    return fitted


def _compress_monotonic_axis(
    response: np.ndarray, source_gray: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    unique, inverse = np.unique(response, return_inverse=True)
    averaged = np.asarray(
        [float(source_gray[inverse == index].mean()) for index in range(len(unique))]
    )
    if len(unique) < 2:
        raise RuntimeError("Isotonic response collapsed to one value")
    return unique, averaged


def _target_intensity(gray: np.ndarray, transfer_mode: str) -> np.ndarray:
    unit = np.asarray(gray, dtype=np.float64) / 255.0
    if transfer_mode == "field_amplitude":
        return np.square(unit)
    if transfer_mode == "linear_intensity":
        return unit
    raise ValueError(f"transfer_mode must be one of {sorted(TRANSFER_MODES)}")


def fit_linearized_lut(
    *,
    base_lut: np.ndarray,
    measured_gray: np.ndarray,
    measured_energy: np.ndarray,
    transfer_mode: str,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """Return a 256-entry LUT and its measured-response inverse mapping."""

    if transfer_mode not in TRANSFER_MODES:
        raise ValueError(f"Unknown target transfer: {transfer_mode}")
    if base_lut.shape != (LUT_SIZE,):
        raise ValueError("base_lut must have 256 entries")
    if measured_gray.shape != measured_energy.shape:
        raise ValueError("measured gray and energy vectors must have equal shape")

    dark_index = int(np.argmin(measured_energy))
    dark_energy = float(measured_energy[dark_index])
    left_range = float(measured_energy[0] - dark_energy)
    right_range = float(measured_energy[-1] - dark_energy)
    if max(left_range, right_range) <= 0.0:
        raise RuntimeError("Neither endpoint is brighter than the measured dark state")
    if right_range >= left_range:
        branch_name = "dark_to_gray255"
        branch_indices = np.arange(dark_index, len(measured_gray))
    else:
        branch_name = "dark_to_gray0"
        branch_indices = np.arange(dark_index, -1, -1)
    branch_gray = measured_gray[branch_indices]
    branch_energy = measured_energy[branch_indices]
    if len(branch_gray) < 8:
        raise RuntimeError(
            f"Selected monotonic branch has only {len(branch_gray)} measured points; "
            "increase dense gray count or inspect the optical polarizer setting"
        )
    isotonic_energy = isotonic_non_decreasing(branch_energy)
    span = float(isotonic_energy[-1] - isotonic_energy[0])
    if span <= 0.05 * float(np.ptp(measured_energy)):
        raise RuntimeError("Selected monotonic branch has insufficient usable dynamic range")
    branch_response = np.clip(
        (isotonic_energy - isotonic_energy[0]) / span, 0.0, 1.0
    )
    response_axis, source_axis = _compress_monotonic_axis(
        branch_response, branch_gray
    )
    requested_gray = np.arange(LUT_SIZE, dtype=np.float64)
    target = _target_intensity(requested_gray, transfer_mode)
    mapped_source_gray = np.interp(target, response_axis, source_axis)
    mapped_dac_float = np.interp(
        mapped_source_gray, np.arange(LUT_SIZE, dtype=np.float64), base_lut
    )
    generated = np.rint(mapped_dac_float).astype(np.int64)
    if np.any(generated < DAC_MIN) or np.any(generated > DAC_MAX):
        raise RuntimeError("Interpolated DAC values leave the valid 12-bit range")

    endpoint_direction = int(np.sign(generated[-1] - generated[0]))
    differences = np.diff(generated)
    voltage_reversals = int(
        np.count_nonzero(differences < 0)
        if endpoint_direction >= 0
        else np.count_nonzero(differences > 0)
    )
    if voltage_reversals:
        raise RuntimeError(
            "The base LUT is not voltage-monotonic on the selected optical branch; "
            "refusing to hide the reversals in a generated LUT"
        )
    predicted = np.interp(mapped_source_gray, branch_gray[::-1], branch_response[::-1]) if branch_gray[0] > branch_gray[-1] else np.interp(mapped_source_gray, branch_gray, branch_response)
    predicted_rmse = float(np.sqrt(np.mean(np.square(predicted - target))))
    rows = [
        {
            "requested_gray": int(gray),
            "target_field_amplitude": float(gray / 255.0),
            "target_normalized_intensity": float(target[index]),
            "mapped_base_gray": float(mapped_source_gray[index]),
            "mapped_base_dac_float": float(mapped_dac_float[index]),
            "generated_dac_integer": int(generated[index]),
            "fit_predicted_normalized_intensity": float(predicted[index]),
        }
        for index, gray in enumerate(range(LUT_SIZE))
    ]
    diagnostics = {
        "transfer_mode": transfer_mode,
        "dark_state_measured_gray": float(measured_gray[dark_index]),
        "dark_state_measured_energy": dark_energy,
        "selected_branch": branch_name,
        "selected_branch_endpoint_gray": float(branch_gray[-1]),
        "selected_branch_measured_points": len(branch_gray),
        "selected_branch_dynamic_range": span,
        "discarded_opposite_branch_points": len(measured_gray) - len(branch_gray),
        "isotonic_adjustment_rmse_fraction_of_branch_span": float(
            np.sqrt(np.mean(np.square(isotonic_energy - branch_energy))) / span
        ),
        "predicted_normalized_intensity_rmse": predicted_rmse,
        "generated_unique_dac_values": int(len(np.unique(generated))),
        "generated_dac_min": int(generated.min()),
        "generated_dac_max": int(generated.max()),
        "generated_voltage_direction": (
            "increasing" if endpoint_direction > 0 else "decreasing"
            if endpoint_direction < 0
            else "constant"
        ),
        "generated_voltage_reversals": voltage_reversals,
        "fit_method": "PAVA isotonic branch fit plus piecewise-linear inverse interpolation",
        "opposite_branch_is_not_used": True,
    }
    return generated, rows, diagnostics


def _resolved_lut_path(config: Mapping[str, Any], config_path: Path) -> Path:
    value = config.get("amplitude_slm", {}).get("lut_file")
    if not value:
        raise ValueError("hardware config amplitude_slm.lut_file is required")
    return resolve_path(value, config_path.parent)


def _calibration_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    raw = config.get("amplitude_lut_calibration", {})
    if not isinstance(raw, Mapping):
        raise ValueError("amplitude_lut_calibration must be a YAML mapping")
    count = int(raw.get("gray_point_count", 64))
    frames = int(raw.get("frames_per_gray", 3))
    mode = str(raw.get("target_transfer", "field_amplitude"))
    output = str(
        raw.get(
            "output_lut_filename",
            "slm7930_at532_70C_linearized_field_amplitude.lut",
        )
    ).strip()
    if not 32 <= count <= 256:
        raise ValueError("LUT gray_point_count must be in 32..256")
    if frames != 3:
        raise ValueError("Audited LUT calibration requires frames_per_gray=3")
    if mode not in TRANSFER_MODES:
        raise ValueError(f"target_transfer must be one of {sorted(TRANSFER_MODES)}")
    if Path(output).name != output or Path(output).suffix.lower() != ".lut":
        raise ValueError("output_lut_filename must be one plain .lut file name")
    return {
        "gray_point_count": count,
        "frames_per_gray": frames,
        "target_transfer": mode,
        "output_lut_filename": output,
        "maximum_verification_intensity_rmse": float(
            raw.get("maximum_verification_intensity_rmse", 0.10)
        ),
        "maximum_verification_field_amplitude_rmse": float(
            raw.get("maximum_verification_field_amplitude_rmse", 0.08)
        ),
        "maximum_verification_monotonic_violations": int(
            raw.get("maximum_verification_monotonic_violations", 2)
        ),
        "maximum_saturation_fraction": float(
            raw.get("maximum_saturation_fraction", 0.001)
        ),
    }


def _scan_config(
    source: Mapping[str, Any],
    *,
    root: Path,
    scan_name: str,
    lut_file: Path,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    config = copy.deepcopy(dict(source))
    config["amplitude_slm"]["lut_file"] = str(lut_file)
    config.setdefault("paths", {})["masks_dir"] = str(root / "masks")
    config["paths"]["results_dir"] = str(root / scan_name)
    exposure = config.setdefault("exposure_calibration", {})
    exposure.pop("gray_values", None)
    exposure["gray_point_count"] = int(settings["gray_point_count"])
    exposure["frames_per_gray"] = int(settings["frames_per_gray"])
    exposure["preview_gray_values"] = [0, 128, 255]
    exposure["saturation_fraction_warning"] = float(
        settings["maximum_saturation_fraction"]
    )
    return config


def _verification_metrics(
    response_csv: Path,
    *,
    transfer_mode: str,
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    gray, energy, metadata = _read_response(response_csv)
    span = float(energy[-1] - energy[0])
    if span <= 0.0:
        raise RuntimeError("Generated LUT verification has non-positive endpoint range")
    normalized = (energy - energy[0]) / span
    target = _target_intensity(gray, transfer_mode)
    intensity_rmse = float(np.sqrt(np.mean(np.square(normalized - target))))
    measured_field = np.sqrt(np.clip(normalized, 0.0, None))
    target_field = gray / 255.0
    field_rmse = float(np.sqrt(np.mean(np.square(measured_field - target_field))))
    violations = int(np.count_nonzero(np.diff(normalized) < -0.01))
    pcc = float(np.corrcoef(normalized, target)[0, 1])
    passed = bool(
        intensity_rmse <= thresholds["maximum_verification_intensity_rmse"]
        and field_rmse <= thresholds["maximum_verification_field_amplitude_rmse"]
        and violations <= thresholds["maximum_verification_monotonic_violations"]
        and metadata["maximum_saturated_pixel_fraction"]
        <= thresholds["maximum_saturation_fraction"]
    )
    return {
        **metadata,
        "endpoint_dynamic_range": span,
        "normalized_intensity_rmse": intensity_rmse,
        "field_amplitude_rmse": field_rmse,
        "normalized_intensity_pcc": pcc,
        "large_monotonicity_violations": violations,
        "thresholds": dict(thresholds),
        "passed": passed,
        "normalization_for_verification_only": "(I-I_gray0)/(I_gray255-I_gray0)",
        "no_network_ccd_postprocessing_added": True,
    }


def _plot_fit(
    *,
    measured_gray: np.ndarray,
    measured_energy: np.ndarray,
    mapping_rows: list[dict[str, Any]],
    verification_csv: Path | None,
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import font_manager, pyplot as plt

    installed = {item.name for item in font_manager.fontManager.ttflist}
    font = "Arial" if "Arial" in installed else "Liberation Sans" if "Liberation Sans" in installed else "DejaVu Sans"
    plt.rcParams.update(
        {
            "font.family": font,
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6,
            "axes.linewidth": 0.6,
            "svg.fonttype": "none",
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(18.3 / 2.54, 5.5 / 2.54), dpi=300)
    normalized_base = (measured_energy - measured_energy.min()) / max(
        float(measured_energy.max() - measured_energy.min()), 1e-12
    )
    axes[0].plot(measured_gray, normalized_base, "o-", markersize=2.0, linewidth=0.8, color="#4C78A8")
    axes[0].set_xlabel("Base-LUT input gray")
    axes[0].set_ylabel("Measured intensity (display norm.)")
    axes[0].set_title("a  Dense base response", loc="left", fontweight="bold")

    requested = np.asarray([row["requested_gray"] for row in mapping_rows])
    mapped = np.asarray([row["mapped_base_gray"] for row in mapping_rows])
    axes[1].plot(requested, mapped, linewidth=0.9, color="#D55E00")
    axes[1].set_xlabel("Requested gray")
    axes[1].set_ylabel("Mapped base-LUT gray")
    axes[1].set_title("b  Inverse mapping", loc="left", fontweight="bold")

    target = np.asarray([row["target_normalized_intensity"] for row in mapping_rows])
    axes[2].plot(requested, target, linestyle="--", linewidth=0.8, color="#222222", label="target")
    if verification_csv is not None and verification_csv.is_file():
        verify_gray, verify_energy, _ = _read_response(verification_csv)
        verify = (verify_energy - verify_energy[0]) / max(float(verify_energy[-1] - verify_energy[0]), 1e-12)
        axes[2].plot(verify_gray, verify, "o-", markersize=2.0, linewidth=0.8, color="#009E73", label="measured")
    else:
        predicted = np.asarray([row["fit_predicted_normalized_intensity"] for row in mapping_rows])
        axes[2].plot(requested, predicted, linewidth=0.8, color="#009E73", label="fit prediction")
    axes[2].set_xlabel("Requested gray")
    axes[2].set_ylabel("Normalized CCD intensity")
    axes[2].set_title("c  Target and verification", loc="left", fontweight="bold")
    axes[2].legend(frameon=False)
    for axis in axes:
        axis.set_xlim(0, 255)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.4, alpha=0.65)
    figure.tight_layout(w_pad=1.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=600, bbox_inches="tight")
    figure.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def run(
    *,
    phase: str,
    config_path: str | Path,
    assume_yes: bool,
    overwrite_generated_lut: bool,
) -> dict[str, Any]:
    config, resolved_config = load_yaml_config(config_path)
    settings = _calibration_settings(config)
    base_lut_path = _resolved_lut_path(config, resolved_config)
    base_lut = read_global_lut(base_lut_path)
    output_lut = base_lut_path.parent / settings["output_lut_filename"]
    if output_lut.resolve() == base_lut_path.resolve():
        raise ValueError("Generated LUT filename must differ from the selected base LUT")
    if (
        phase in {"fit", "all"}
        and output_lut.exists()
        and not overwrite_generated_lut
    ):
        raise FileExistsError(
            f"Generated LUT already exists: {output_lut}. Refusing to spend time "
            "on a hardware scan that cannot be committed. Use a new output name "
            "or explicitly pass --overwrite-generated-lut."
        )
    root = resolved_config.parents[1] / "results" / "lut_calibration" / output_lut.stem
    root.mkdir(parents=True, exist_ok=True)
    baseline_csv = root / "base_scan" / "slm_response.csv"
    verification_csv = root / "verification_scan" / "slm_response.csv"
    report: dict[str, Any] = {
        "phase": phase,
        "base_lut": str(base_lut_path),
        "base_lut_sha256": _sha256(base_lut_path),
        "generated_lut": str(output_lut),
        "calibration_root": str(root),
        "settings": settings,
    }
    if phase in {"capture-base", "all"}:
        baseline_config = _scan_config(
            config,
            root=root,
            scan_name="base_scan",
            lut_file=base_lut_path,
            settings=settings,
        )
        report["base_capture"] = run_exposure(
            baseline_config, resolved_config, assume_yes=assume_yes
        )
    if phase in {"fit", "all"}:
        gray, energy, response_metadata = _read_response(baseline_csv)
        if response_metadata["maximum_saturated_pixel_fraction"] > settings[
            "maximum_saturation_fraction"
        ]:
            raise RuntimeError(
                "Base response is saturated. Reduce camera_exposure_us, rerun "
                "prepare_lab, then repeat LUT calibration."
            )
        generated, mapping_rows, fit = fit_linearized_lut(
            base_lut=base_lut,
            measured_gray=gray,
            measured_energy=energy,
            transfer_mode=settings["target_transfer"],
        )
        write_global_lut(
            output_lut, generated, overwrite=overwrite_generated_lut
        )
        _write_csv(root / "lut_mapping.csv", mapping_rows)
        fit_report = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "base_lut": str(base_lut_path),
            "base_lut_sha256": _sha256(base_lut_path),
            "base_response": response_metadata,
            "generated_lut": str(output_lut),
            "generated_lut_sha256": _sha256(output_lut),
            "generated_lut_is_new_file": output_lut.resolve() != base_lut_path.resolve(),
            "fit": fit,
            "network_contract": (
                "requested uint8 represents optical field amplitude; target CCD "
                "intensity is squared amplitude"
                if settings["target_transfer"] == "field_amplitude"
                else "requested uint8 targets linear CCD intensity"
            ),
        }
        _write_json(root / "lut_fit_report.json", fit_report)
        _plot_fit(
            measured_gray=gray,
            measured_energy=energy,
            mapping_rows=mapping_rows,
            verification_csv=None,
            output=root / "lut_calibration",
        )
        report["fit"] = fit_report
    if phase in {"verify", "all"}:
        if not output_lut.is_file():
            raise FileNotFoundError(f"Generated LUT is missing: {output_lut}")
        verification_config = _scan_config(
            config,
            root=root,
            scan_name="verification_scan",
            lut_file=output_lut,
            settings=settings,
        )
        report["verification_capture"] = run_exposure(
            verification_config, resolved_config, assume_yes=assume_yes
        )
        verification = _verification_metrics(
            verification_csv,
            transfer_mode=settings["target_transfer"],
            thresholds=settings,
        )
        report["verification"] = verification
        gray, energy, _ = _read_response(baseline_csv)
        with (root / "lut_mapping.csv").open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            mapping_rows = list(csv.DictReader(handle))
        converted_rows = [
            {key: float(value) if key != "requested_gray" and key != "generated_dac_integer" else int(value) for key, value in row.items()}
            for row in mapping_rows
        ]
        _plot_fit(
            measured_gray=gray,
            measured_energy=energy,
            mapping_rows=converted_rows,
            verification_csv=verification_csv,
            output=root / "lut_calibration",
        )
        final = {
            "schema_version": 1,
            "generated_lut": str(output_lut),
            "generated_lut_sha256": _sha256(output_lut),
            "verification": verification,
            "recommended_for_use": bool(verification["passed"]),
            "next_step_if_passed": (
                "Set LAB_CONFIG.yaml amplitude_lut_filename to "
                f"{output_lut.name}, rerun prepare_lab, then repeat the normal "
                "32-point exposure check."
            ),
            "rollback": f"Restore amplitude_lut_filename to {base_lut_path.name}",
            "base_lut_was_overwritten": False,
        }
        _write_json(root / "final_lut_report.json", final)
        report["final"] = final
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=("capture-base", "fit", "verify", "all"), default="all"
    )
    parser.add_argument(
        "--config", default="experiments/lab_qwen/generated/formal_hardware.yaml"
    )
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--overwrite-generated-lut", action="store_true")
    args = parser.parse_args(argv)
    report = run(
        phase=args.phase,
        config_path=args.config,
        assume_yes=args.yes,
        overwrite_generated_lut=args.overwrite_generated_lut,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
