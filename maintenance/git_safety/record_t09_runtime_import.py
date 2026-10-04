"""Record tested T09 source identities, without moving data or changing models."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
TASK = 'LightGenV2/tasks/t09_multimodal_matching/'
SOURCE = '190ceb9da7add8d93ea75c3086ace9ce58019d71'


def main():
    raw = json.loads((ROOT/'.codex_tmp/t11_source_20261004/t09_runtime_candidate_cpu.json').read_text(encoding='utf-8'))
    if raw['exit_code'] or not re.search(r'\b29 passed\b', raw['stdout']):
        raise RuntimeError('Incomplete T16 test result')
    cpu = json.loads(raw['stdout'][raw['stdout'].index('{\n'):])
    if (cpu['commit'] != '5a413827bc0cb4c80d8034bb915de8169bf3a197'
            or len(cpu['optical_gradient_cases']) != 8 or len(cpu['strict_reloads']) != 7
            or cpu['t16_test_exit_code'] or cpu['dataset_evaluated']):
        raise RuntimeError('Unexpected CPU contract scope')
    additions = json.loads((ROOT/'maintenance/storage/T09_RUNTIME_ADDITIONS_20261004.json').read_text())
    replacements = json.loads((ROOT/'maintenance/storage/T09_VISION_REPLACEMENT_20261004.json').read_text())
    paths = [r['path'] for r in additions['paths']]+additions['existing_identical_dependencies']
    paths += [r['path'] for r in replacements['paths']]
    rows = []
    for path in sorted(paths):
        content = subprocess.check_output(['git', '-C', str(ROOT), 'show', SOURCE+':'+path])
        candidate = subprocess.check_output(['git', '-C', str(ROOT), 'show', cpu['commit']+':'+path])
        if candidate != content:
            raise RuntimeError('Candidate not actual source: '+path)
        rows.append(dict(path=path, source_blob_sha256=hashlib.sha256(content).hexdigest()))
    report = dict(schema_version=1, source_head='1c7222a1119385475e1b464ca7efdb67cdc21a4c',
                  source_overlay_commit=SOURCE,
                  source_runtime='/DATA/DATA1/guest3/demo_reproduction_20260915',
                  files=rows, cpu_contract_evidence=cpu, t16_tests_passed=29,
                  shared_t16_default_compatibility='maintenance/storage/T09_T16_VISION_COMPATIBILITY_20261004.json',
                  protected_assets='maintenance/storage/T09_RUNTIME_IDENTITY_20261004.json',
                  datasets_and_run_assets_moved=False, historical_metrics_replayed=False,
                  source_import_is_not_permission_to_delete_original_runtime=True,
                  historical_protocol_warning='Current server D2NN forward uses whole-field bilinear enlarge_full_aperture. Older reports describe separate modality/tile nearest-neighbor enlargement. Strict PT loading only proves tensor compatibility, not historical forward equivalence; reproducing those metrics requires the original per-run source commit and contract.')
    dest = ROOT/TASK/'source_import_20261004.json'
    if dest.exists():
        raise FileExistsError(dest)
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(source_identities=len(rows), strict_reloads=7, t16_tests=29,
                         metrics_replayed=False)))


if __name__ == '__main__':
    main()
