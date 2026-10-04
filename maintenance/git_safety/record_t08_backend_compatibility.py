"""Record a bounded, verified old/main CPU comparison; no model/data writes."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read_result(name):
    raw = json.loads((ROOT/'.codex_tmp/t11_source_20261004'/name).read_text(encoding='utf-8'))
    if raw['exit_code']:
        raise RuntimeError('Remote CPU audit failed: '+name)
    report = json.loads(raw['stdout'])
    if report['body_sha256'] != 'cc977b83286a8e90398ebd30064428886bc3c06f1eb0557e470060f7ff5c1cae':
        raise RuntimeError('Wrong adopted body')
    if report['dataset_evaluated'] or report['runtime_assets_modified'] or report['cuda_visible_devices']:
        raise RuntimeError('Unexpected audit scope')
    return report


def main():
    old = read_result('t08_original_forward_cpu_20261004.json')
    new = read_result('t08_main_forward_cpu_20261004.json')
    if old['commit'] != 'd0662a7d240340817948a3496c2cd43f4240e76d' or new['commit'] != '5d3a9845556cf36b059a992eb294a41f18c3fc2b':
        raise RuntimeError('Unexpected source identity')
    for key in ('synthetic_forward', 'strict_reloads', 'config_fixtures_verified', 'geometry_m'):
        if old[key] != new[key]:
            raise RuntimeError('Old/main contract differs: '+key)
    changed = {path for path in new['imports'] if old['imports'].get(path) != new['imports'][path]}
    allowed = {'LightGenV2/tasks/t01_object_retrieval/modeling.py',
               'experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust/settings.py'}
    if changed != allowed:
        raise RuntimeError('Unexpected shared runtime dependency change')
    report = dict(schema_version=1, original_runtime=old['commit'], verified_main=new['commit'],
                  adopted_body_sha256=new['body_sha256'], geometry_m=0.10,
                  config_fixtures_verified=new['config_fixtures_verified'],
                  strict_reloads=new['strict_reloads'], synthetic_forward=new['synthetic_forward'],
                  synthetic_outputs_bit_identical=True, changed_shared_paths=sorted(changed),
                  scope='Four synthetic tokens per modality through both blocks, and two synthetic readout inputs; same CPU/environment and adopted PT. Not exhaustive forward equivalence or dataset reproduction.',
                  imported_dependency_sha256=new['imports'],
                  runtime_assets_modified=False, scientific_metrics_revalidated=False,
                  hardware_deployment_verified=False, reverse_entry_published=False,
                  original_runtime_safe_to_delete=False)
    target = ROOT/'maintenance/storage/T08_BACKEND_COMPATIBILITY_20261004.json'
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(strict_pt_components=3, config_fixtures=21, synthetic_outputs_bit_identical=True,
                         reverse_entry_published=False)))


if __name__ == '__main__':
    main()
