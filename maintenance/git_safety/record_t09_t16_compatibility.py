"""Record actual T16 shared-vision PT compatibility, not dataset accuracy."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    raw = json.loads((ROOT/'.codex_tmp/t11_source_20261004/t09_t16_formal_vision_compatibility.json').read_text(encoding='utf-8'))
    if raw['exit_code']:
        raise RuntimeError('Failed CPU compatibility check')
    report = json.loads(raw['stdout'])
    pt = report['formal_checkpoint_compatibility']
    if (pt['sha256'] != 'd20273e1828fec4e5a7a5545fd892a722f3909f7205a791e0e682b437526aa11'
            or not pt['old_and_new_strict_load'] or not pt['synthetic_outputs_identical']
            or report['data_read'] or report['t16_runtime_metrics_revalidated']):
        raise RuntimeError('Unexpected compatibility scope')
    report['old_source_ref'] = '1c7222a1119385475e1b464ca7efdb67cdc21a4c'
    report['new_overlay_ref'] = '190ceb9da7add8d93ea75c3086ace9ce58019d71'
    report['formal_dataset_metrics_replayed'] = False
    report['main_shared_source_changed'] = False
    target = ROOT/'maintenance/storage/T09_T16_VISION_COMPATIBILITY_20261004.json'
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'strict_formal_pt_compatibility': True, 'dataset_evaluation': False}))


if __name__ == '__main__':
    main()
