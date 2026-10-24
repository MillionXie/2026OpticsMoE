"""CPU-only per-file SHA index of this authorized completed 80GB rental."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
TASK=REPO/'LightGenV2/tasks/t06_video_quality_assessment'
BASE=TASK/'runs/simulation/lgvq_mps_e2e_a100_20261024'
GPU='GPU-6d1916d1-fbd3-524b-3d98-7fb38512c388'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def build():
    groups=[(BASE/'rental_80gb_20261025_01','qualified_DS_Kimi_costs'),
        (BASE/'rental_qwen_anchor_20261025_02','qualified_supplementary_same80GB_Qwen_anchor'),
        (BASE/'rental_qwen_anchor_20261025_01','rejected_auxiliary_metadata_identity_gate_zero_cost_calls'),
        (TASK/'runs/simulation/a100_rental_original_metadata_20261025_01','original_metadata_and_frozen_heads'),
        (TASK/'runs/simulation/rental_qwen_auxiliary_records_20261025_01','preserved_auxiliary_download_records')]
    points=[]
    for root,role in groups[:2]:
        state=read(root/'suite_status.json')
        expected=12 if root.name=='rental_80gb_20261025_01' else 3
        if state['status']!='complete' or not state['mps_released'] or len(state['completed'])!=expected:
            raise ValueError('Complete released original finite suite required')
        for name in state['completed']:
            point=root/name;report=read(point/'report.json');audit=read(point/'local_independent_audit01.json')
            if (report['status']!='complete' or audit['status']!='passed' or audit['report_sha256']!=sha(point/'report.json')
                    or report['gpu_uuid']!=GPU or report['cost']['videos']!=200):
                raise ValueError('Original point and local independent replay are not bound')
            points.append(dict(path=point.relative_to(REPO).as_posix(),role=role,
                model=report['model'],cap=report['cap'],repeat=report['repeat'],
                report_sha256=sha(point/'report.json'),local_audit_sha256=sha(point/'local_independent_audit01.json')))
    for root,_ in groups[:4]:
        receipt='backup_receipt_asset_metadata_rejection01.json' if root.name=='rental_qwen_anchor_20261025_01' else 'backup_receipt_complete01.json'
        for item in read(root/receipt)['verified_files']:
            if sha(root/item['path'])!=item['sha256']:raise ValueError('A backed-up original differs')
    meta=groups[3][0]
    for item in read(meta/'manifest.json')['files']:
        if sha(meta/item['path'])!=item['sha256']:raise ValueError('Original metadata/head identity differs')
    release=read(BASE/'post_all_rental_release01.json')
    if release['gpu_uuid']!=GPU or release['compute']['stdout'].strip():raise ValueError('GPU release evidence missing')
    if any(v['returncode']==0 for v in release['private_controls'].values()):raise ValueError('Private MPS still active')
    records=[]
    def add(path,role):
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(TASK):
            raise ValueError('Only explicit canonical task DATA files')
        records.append(dict(path=path.relative_to(REPO).as_posix(),role=role,bytes=path.stat().st_size,sha256=sha(path)))
    for root,role in groups:
        for path in sorted(root.rglob('*')):
            if path.is_file():
                if path.name.endswith(('.part','.tmp')):raise ValueError('Incomplete transfer in raw bundle')
                add(path,role)
    for name in ('rental_80gb_20261025_01.controller.log','rental_80gb_20261025_01.admission.json',
            'rental_qwen_anchor_20261025_01.controller.log','rental_qwen_anchor_20261025_02.controller.log',
            'rental_qwen_auxiliary_metadata_archive_20261025_01.json','post_all_rental_release01.json'):
        add(BASE/name,'original_external_controller_or_identity_release_receipt')
    return dict(status='ALL_15_RENTAL_POINTS_BACKED_UP_AND_INDEPENDENTLY_AUDITED',
        gpu_uuid=GPU,hardware='A100-SXM4-80GB /400W',qualified_points=points,
        rejected_run_separate=True,not_the_final_two_Excel_delivery=True,
        all_original_accuracy_is_confirmed_full558_not_new200=True,
        supplementary_Qwen_not_replacement_of_shared40GB_anchor=True,
        records=records,files=len(records),bytes=sum(v['bytes'] for v in records),
        program_sha256=sha(__file__),program_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Immutable raw index only')
    result=build()
    with a.output.open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
    print(json.dumps(dict(status=result['status'],qualified_points=len(result['qualified_points']),files=result['files'],bytes=result['bytes'])))


if __name__=='__main__':main()
