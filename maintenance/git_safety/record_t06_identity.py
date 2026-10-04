"""Record bounded T06 provenance audits, without copying code, PT or datasets."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    audit_root = ROOT / ".codex_tmp/t11_source_20261004"
    reports = {}
    for key, name in (
        ("runtime_inventory", "t06_runtime_inventory.json"),
        ("profile_contracts", "t06_profile_contract_audit.json"),
        ("named_asset_locations", "t06_named_asset_locations.json"),
        ("verified_releases", "t06_verified_release_source.json"),
        ("strict_reload", "t06_release_strict_reload.json"),
    ):
        transport = json.loads((audit_root / name).read_text(encoding="utf-8"))
        if transport["exit_code"]:
            raise RuntimeError("Remote audit did not complete: " + name)
        reports[key] = json.loads(transport["stdout"])
    for release in reports["verified_releases"]["releases"]:
        if not release["declared_checkpoint_matches"] or not all(
            row["declared_sha_matches"] and row["pinned_git_matches"]
            for row in release["source_files"]
        ):
            raise RuntimeError("Release source/checkpoint identity failed")
    if not all(row["strict_load"] for row in reports["strict_reload"]["versions"]):
        raise RuntimeError("A fixed deployment model failed strict reload")
    report = {"audited_on": "2026-10-04", "read_only": True,
              "runtime_migration_complete": False,
              "interpretation": "Verified physical release source and PT; legacy profiles separately report mismatch. No scientific reevaluation or file removal.",
              **reports}
    target = ROOT / "maintenance/storage/T06_RUNTIME_IDENTITY_20261004.json"
    if target.exists():
        raise FileExistsError("Earlier audit is protected")
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"releases_checked": len(report["verified_releases"]["releases"]),
                      "strict_models_loaded": len(report["strict_reload"]["versions"]),
                      "runtime_roots": len(report["runtime_inventory"]["runtime_roots"]),
                      "published_source_changed": False}))


if __name__ == "__main__":
    main()
