"""Read-only identities for preserved T09 report assets; no metric reevaluation."""
import hashlib
import json
from pathlib import Path


FOLDERS = (
    'text_encoding_s17_20260917_v2', 'test_s17_20260917',
    'generalization_s17_20260917', 'phase_budget_s17_20260917',
    'audio_matching_s17_20260917',
)
ROOTS = ('.worktrees/t12_cross_modal', '.codex_tmp/demo_reproduction_worktree')
RELATIVE = 'LightGenV2/tasks/t09_multimodal_matching/reports/figures'


def inspect(root):
    inventories = {}
    for source in ROOTS:
        rows = {}
        base = root / source / RELATIVE
        for folder in FOLDERS:
            for path in sorted((base / folder).rglob('*')):
                if path.is_file():
                    content = path.read_bytes()
                    rows[path.relative_to(base).as_posix()] = {
                        'bytes': path.stat().st_size,
                        'sha256': hashlib.sha256(content).hexdigest(),
                    }
                    if path.suffix.lower() in ('.json', '.svg'):
                        rows[path.relative_to(base).as_posix()]['lf_sha256'] = hashlib.sha256(
                            content.replace(b'\r\n', b'\n')).hexdigest()
        inventories[source] = rows
    first, second = (inventories[name] for name in ROOTS)
    shared = set(first) & set(second)
    return {
        'read_only': True, 'metrics_reevaluated': False,
        'assets_moved_or_deleted': False, 'relative_asset_root': RELATIVE,
        'inventories': inventories,
        'exact_shared_files': sorted(p for p in shared if first[p] == second[p]),
        'different_shared_files': sorted(p for p in shared if first[p] != second[p]),
        'different_bytes_equal_lf': sorted(p for p in shared
            if first[p] != second[p] and first[p].get('lf_sha256')
            and first[p]['lf_sha256'] == second[p].get('lf_sha256')),
        'first_only_files': sorted(set(first) - set(second)),
        'second_only_files': sorted(set(second) - set(first)),
        'limitations': 'Current local file identity, not acquisition-time provenance or server completeness',
    }


if __name__ == '__main__':
    print(json.dumps(inspect(Path(__file__).resolve().parents[2]), indent=2))
