"""Append only the four newly qualified costs to the two sealed prior indices.

No model import/forward, no remeasurement, and no replay of the original fourteen
qualified items. New leaves must already have matching server/local independent
raw audits and exact data-only backup receipts. Prior indices stay immutable.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
MAIN=HERE/'measured_index_partial_17.json'
TEMP=HERE/'temporal_matrix_partial_07.json'
MAIN_SHA='5d05ea0c395b2dfdc07135b4e8cf15957c3171568af466cda7a83fb2c4593bda'
TEMP_SHA='3b439f3cb1b6f9b61a0349147304247259d3514e767265b6ea5a368ff34c2b1b'
T12=REPO/'LightGenV2/tasks/t12_text_to_image/runs/simulation'
LGVQ=REPO/'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation'
GPU_KIND='same_A100_native_GPU_equivalence_20261025'
PROOFS={'t12_ours':'a37514afe55c0774d411e672a0dd66641c30bbd6ad49fbf8cf2728ab47ccd1c0',
        't12_qwen':'a678dfda4fb146a76f64c64bfc86add37313b966499b59384047d6359901559d',
        'lgvq':'7045cb50c488b846c4ae55691f0fa782efebd4325a12d208191c2d9b976d1cc5'}


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rel(path):return Path(path).relative_to(REPO).as_posix()
def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def close(a,b):
    if not math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-7):raise ValueError(('Cost denominator differs',a,b))


def checked_leaf(folder,repeat,kind):
    identity=read(folder/'identity.json');summary=read(folder/'summary.json')
    auditpath=folder/'local_independent_audit01.json';serverpath=folder.parent/(folder.name+'.cpu_audit.json')
    local=read(auditpath)
    if (local!=read(serverpath) or local['status']!='passed' or local['repeat']!=repeat
            or identity['repeat']!=repeat or summary['status']!='complete'):
        raise ValueError('Both independently computed original raw audits must agree')
    receipt=folder/'backup_receipt_complete01.json';backup=read(receipt)
    pinned={r['path']:r for r in backup['verified_files']}
    manifest=read(folder/'files_manifest.json')
    for r in [*manifest,dict(file='files_manifest.json',bytes=(folder/'files_manifest.json').stat().st_size,
                             sha256=sha(folder/'files_manifest.json'))]:
        name=r['file'];p=folder/name;b=pinned.get(name)
        if Path(name).name!=name or b is None or b['sha256']!=r['sha256'] or b['bytes']!=r['bytes']:
            raise ValueError('Original sealed manifest/data-only backup do not agree')
        if p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:raise ValueError('Original sealed raw changed')
    for key in ('seconds200','joules200','sample_ms','sample_j'):close(local[key],summary[key])
    proof=identity['cpu_contract_sha256']
    if proof!=PROOFS[kind]:raise ValueError('Do not replace a qualified proof')
    if kind!='t12_ours':
        if (identity['numerical_proof_kind']!=GPU_KIND or local['numerical_proof_kind']!=GPU_KIND
                or not identity['legacy_cpu_contract_fields_are_path_SHA_aliases_not_CPU_execution_claims']
                or local['numerical_contract_sha256']!=proof):raise ValueError('Native GPU proof mislabeled as CPU')
    else:
        if local['CPU_contract_sha256']!=proof:raise ValueError('Retained original Ours CPU proof differs')
    return dict(raw_run_path=rel(folder),**local,local_audit_sha256=sha(auditpath),
                server_audit_path=rel(serverpath),server_audit_sha256=sha(serverpath),
                backup_receipt_path=rel(receipt),backup_receipt_sha256=sha(receipt)),identity,summary


def summed(refs,**extra):
    if [r['repeat'] for r in refs]!=[1,2,3]:raise ValueError('All three original fresh rounds required')
    seconds=sum(r['seconds200'] for r in refs);joules=sum(r['joules200'] for r in refs)
    return dict(status='independently_audited_600_calls',calls=600,sample_ms=seconds/.6,
                sample_j=joules/600,raw_references=refs,**extra)


def t12_pair():
    result={};signatures={};tokens=None
    for system in ('ours','qwen'):
        root=T12/('a100_input_inclusive_20261025_07' if system=='ours' else 'a100_gpu_checked_remaining_20261025_01')
        refs=[];identities=[];summaries=[];signatures[system]=[]
        for repeat in (1,2,3):
            folder=root/f'{system}_repeat{repeat:02d}';audit,identity,summary=checked_leaf(folder,repeat,'t12_'+system)
            calls=rows(folder/'calls.csv')
            if len(calls)!=200:raise ValueError('T12 first200 records required')
            signature=[tuple(c[k] for k in ('index','sample_id','input_ids','attention_mask')) for c in calls]
            signatures[system].append(signature);refs.append(audit);identities.append(identity);summaries.append(summary)
        if signatures[system][1:]!=[signatures[system][0]]*2 or len({r['outputs_sha256'] for r in refs})!=1:
            raise ValueError('Fresh T12 repeats do not have identical workload/full RGB outputs')
        result[system]=summed(refs,unique_input_pairs=200,equivalent_frame_ms=sum(r['seconds200'] for r in refs)/.6,
            equivalent_frame_j=sum(r['joules200'] for r in refs)/600,
            full2304_reference_PSNR=refs[0]['formal_full2304_reference_PSNR'],
            validation_platform='retained original CPU equivalence' if system=='ours' else 'same native A100 GPU equivalence, not CPU',
            numerical_contract_sha256=PROOFS['t12_'+system],
            native_unit='one original RGB256 reference and UTF8 instruction to full float CPU RGB256',
            scope='original decode/tokenizer/H2D/embedding/full original NN/output and all199 gaps; model loading excluded')
        if system=='ours':
            result[system].update(optical_ms_total=sum(r['optical_ms_total'] for r in refs),
                charged_electronic_ms_total=sum(r['charged_electronic_ms_total'] for r in refs),
                actual_software_seconds_total=sum(r['software_wall_s'] for r in refs),
                optical_passes_per_query=6,not_a_device_measured_e2e=True)
        else:
            tokens=sum(r['native_valid_input_tokens200'] for r in refs)
            result[system]['time_weighted_board_w']=sum(r['diagnostic_software_board_j200'] for r in refs)/sum(r['seconds200'] for r in refs)
    if signatures['ours']!=signatures['qwen']:raise ValueError('Both T12 models must share actual sample IDs/native IDs/masks')
    if tokens!=19596:raise ValueError('Actual same-input native token numerator differs')
    for system in result:
        result[system].update(common_input_tokens_total=tokens,common_tokens_per_s=tokens/(result[system]['sample_ms']*.6))
    return result,common_pair(result['ours'],result['qwen'],tokens,'same original RGB/UTF8/sample IDs and actual native Qwen IDs/masks')


def common_pair(ours,qwen,tokens,basis):
    os=ours['sample_ms']*.6;qs=qwen['sample_ms']*.6
    return dict(total=tokens,ours_per_s=tokens/os,qwen_per_s=tokens/qs,rate_ratio=qs/os,time_ratio=qs/os,
                basis=basis,not_generation_throughput=True)


def lgvq_result(root,target,qwen):
    refs=[];signatures=[]
    for repeat in (1,2,3):
        folder=root/f'{target}_repeat{repeat:02d}';audit,identity,summary=checked_leaf(folder,repeat,'lgvq')
        if audit['target']!=target or audit['valid_videos']!=200:raise ValueError('Wrong original LGVQ task/workload')
        signatures.append([(r['test_indices'],r['valid'],r['source_ids'],r['original_native_prompt']) for r in rows(folder/'fields.csv')])
        refs.append(audit)
    if signatures[1:]!=[signatures[0]]*2 or len({r['predictions_sha256'] for r in refs})!=1:
        raise ValueError('Fresh LGVQ fields/full predictions differ')
    seconds=sum(r['seconds200'] for r in refs);joules=sum(r['joules200'] for r in refs)
    tokens=qwen['common_input_tokens_total']
    value=summed(refs,unique_videos=200,frames_per_video=4,equivalent_frame_ms=seconds/2.4,
        equivalent_frame_j=joules/2400,common_input_tokens_total=tokens,common_tokens_per_s=tokens/seconds,
        optical_ms_total=1000*sum(r['parameter_optical_s'] for r in refs),
        charged_electronic_ms_total=1000*sum(r['frontend_wall_s']+r['selected_body_boundary_s']+r['all_field_gaps_s'] for r in refs),
        frontend_wall_ms_total=1000*sum(r['frontend_wall_s'] for r in refs),
        selected_body_boundary_ms_total=1000*sum(r['selected_body_boundary_s'] for r in refs),
        actual_software_seconds_total=sum(read(root/f'{target}_repeat{i:02d}'/'summary.json')['actual_software_wall_seconds'] for i in (1,2,3)),
        fields_per_repeat=13 if target=='temporal' else 200,executed_slots_per_repeat=208 if target=='temporal' else 200,
        optical_passes_per_field=6,validation_platform='same native A100 GPU equivalence, not CPU',
        numerical_contract_sha256=PROOFS['lgvq'],not_a_device_measured_e2e=True,
        scope='full real four-frame original video frontend + original selected NN incl actual merger + six optical parameter passes/field + all true gaps; temporal tail padding fully charged')
    return value,common_pair(value,qwen,tokens,'same first200 original video byte identities and original four-frame views; numerator from qualified native Qwen input IDs/masks, not Ours38-token prompt')


def attach_t12_student_op(ours):
    path=HERE/'t12_student_qualified600_shape_MAC_01.json'
    if sha(path)!='0784e041531414e3b05a80bff35ae6f0767060c2fae024746f881f39bd7c8fb1':raise ValueError('Retain original zero-forward formula receipt')
    value=read(path);op=2*value['formal600_NN_MAC']
    close(sum(r['seconds200'] for r in value['cost_references']),ours['sample_ms']*.6)
    close(sum(r['joules200'] for r in value['cost_references']),ours['sample_j']*600)
    if value['model_or_GPU_forwards']!=0 or value['physical_optical_dense_equivalent_OP'] is not None:raise ValueError('NN subtotal scope differs')
    ours['derived_NN_dense_MAC_subtotal']=dict(OP600=op,TOP_per_input=op/600e12,
        TOPS=op/1e12/(ours['sample_ms']*.6),TOPS_per_W=op/1e12/(ours['sample_j']*600),
        local_receipt=rel(path),local_receipt_sha256=sha(path),not_complete_scalar_OP=True,
        physical_optical_dense_equivalent_not_in_NN_subtotal=True,independent_server_formula_replay=False,raw_derivation=value)


def complete(lgvq_root):
    if sha(MAIN)!=MAIN_SHA or sha(TEMP)!=TEMP_SHA:raise ValueError('Sealed prior indices changed')
    suite=read(lgvq_root/'suite_status.json')
    expected=[f'{target}_repeat{i:02d}' for target in ('temporal','spatial') for i in (1,2,3)]
    if suite['status']!='complete' or suite.get('mps_released') is not True or suite['completed']!=expected:
        raise ValueError('All six LGVQ costs and actual private MPS release required')
    original=read(MAIN);main=copy.deepcopy(original);temp=read(TEMP);oldtemp=copy.deepcopy(temp)
    byid={r['id']:r for r in main['nine_metric_rows']};pair,common=t12_pair()
    byid['08'].update(ours=pair['ours'],qwen=pair['qwen'],common_tokens=common,
        status='matched_three_fresh_round_timing_energy_independently_audited',
        ops_status='Ours actual600 original NN dense-MAC derived; Qwen complete NN OP derivation not yet certified; unknown is not zero')
    byid['08']['full_test_reference']['qwen']=pair['qwen']['full2304_reference_PSNR']
    attach_t12_student_op(byid['08']['ours'])
    for target,rid in (('temporal','01a'),('spatial','01b')):
        value,common=lgvq_result(lgvq_root,target,byid[rid]['qwen'])
        byid[rid].update(ours=value,common_tokens=common,status='matched_three_fresh_round_timing_energy_independently_audited',
            ops_status='Qwen original NN dense-MAC retained; Ours saved-shape NN derivation pending; physical optical dense-equivalent unknown, not zero')
    for old in original['nine_metric_rows']:
        for system in ('ours','qwen'):
            if old[system] is not None and byid[old['id']][system]!=old[system]:raise ValueError('Original fourteen qualified objects must remain exactly unchanged')
    if main['shared_qwen_temporal100']!=original['shared_qwen_temporal100']:raise ValueError('Shared Qwen100 anchor changed')
    receipt=dict(previous_main=rel(MAIN),previous_main_sha256=MAIN_SHA,previous_temporal=rel(TEMP),previous_temporal_sha256=TEMP_SHA,
        source=rel(__file__),source_sha256=sha(__file__),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        original14_objects_unchanged=True,original14_model_or_GPU_replays=0,new_cost_model_or_GPU_replays=0,
        GPU_equivalence_not_CPU=True,legacy_cpu_contract_fields_are_path_SHA_aliases_only=True,
        complete_scalar_or_physical_optical_OP_claim=False)
    main.update(status='COMPLETE_TIMING_ENERGY_18_OF_18_OPERATION_SCOPE_EXPLICIT',qualified_model_task_items=18,
        completion_receipt=receipt,physical_optical_dense_equivalent_OP=None)
    ours=copy.deepcopy(byid['01a']['ours']);anchor=temp['shared_qwen_temporal100']['aggregate']
    close(anchor['sample_ms'],byid['01a']['qwen']['sample_ms']);close(anchor['sample_j'],byid['01a']['qwen']['sample_j'])
    temp.update(status='COMPLETE_TIMING_ENERGY_WITH_QUALIFIED_LIGHTGEN_TEMPORAL',lightgen_ours_temporal=ours,
                completion_receipt=receipt,relative_to_lightgen=[])
    for row in [*temp['rows'],dict(temp['supplementary_rental_qwen_anchor'],model='qwen',cap=100,supplementary=True)]:
        a=row['aggregate'];temp['relative_to_lightgen'].append(dict(model=row['model'],cap=row['cap'],platform=row['platform'],
            supplementary=row.get('supplementary',False),time_ratio=a['sample_ms']/ours['sample_ms'],
            energy_ratio=a['sample_j']/ours['sample_j'],cross_platform_not_a_causal_model_speedup=row['platform']['power_limit_w']!=250))
    if any(temp[k]!=v for k,v in oldtemp.items() if k!='status'):raise ValueError('Original temporal matrix objects changed')
    return main,temp


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--main-output',type=Path,required=True)
    p.add_argument('--temporal-output',type=Path,required=True);p.add_argument('--lgvq-root',type=Path,required=True)
    args=p.parse_args()
    if args.main_output.exists() or args.temporal_output.exists():raise FileExistsError('New immutable indices required')
    main,temp=complete(args.lgvq_root.resolve())
    for path,value in ((args.main_output,main),(args.temporal_output,temp)):
        with path.open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2)
    print(json.dumps(dict(status=main['status'],qualified_items=18,original14_objects_unchanged=True,
        main_sha256=sha(args.main_output),temporal_sha256=sha(args.temporal_output))))


if __name__=='__main__':main()
