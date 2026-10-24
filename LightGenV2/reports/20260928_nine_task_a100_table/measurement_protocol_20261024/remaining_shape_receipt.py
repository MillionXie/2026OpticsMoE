"""Bind three existing-shape NN subtotals to their exact qualified cost triples."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def independent_pair(local,server,commit_key):
    value=json.loads(local.read_text());other=json.loads(server.read_text())
    # Only the checkout provenance can differ after an unrelated peer commit.
    # Formula bytes, graph evidence, all600 records and cost references must match.
    if (commit_key not in value or commit_key not in other
            or {k:v for k,v in value.items() if k!=commit_key}!={k:v for k,v in other.items() if k!=commit_key}):
        raise ValueError('Independent same-scope shape formulas differ')
    return value


def match_cost(value,seconds,energy,report_shas,report_key='report_sha256'):
    refs=value['cost_references']
    if (value['calls']!=600 or len(refs)!=3
            or [r[report_key] for r in refs]!=report_shas
            or abs(sum(r['seconds200'] for r in refs)-seconds)>1e-8
            or abs(sum(r['joules200'] for r in refs)-energy)>1e-7):
        raise ValueError('Original shape/cost triple differs')


def qwen_subtotal(value,seconds,energy,report_shas):
    match_cost(value,seconds,energy,report_shas)
    repeats=value['repeat_results'];op=value['MAC_OP600']
    if (value['GPU_forwards']!=0 or value['new_neural_forwards']!=0
            or value['new_timing_or_power_used'] or value['complete_scalar_OP_claim']
            or [r['repeat'] for r in repeats]!=[1,2,3]
            or any(len(r['records'])!=200 or r['MAC200']!=sum(v['MAC'] for v in r['records']) for r in repeats)
            or any(v['MAC']!=sum(v['MAC_terms'].values()) for r in repeats for v in r['records'])
            or op!=2*sum(r['MAC200'] for r in repeats)):
        raise ValueError('Original600 dense-NN MAC scope differs')
    return dict(OP600=op,TOP_per_input=op/600/1e12,TOPS=op/1e12/seconds,
        TOPS_per_W=op/1e12/energy,not_complete_scalar_OP=True)


def image_student_subtotal(value,seconds,energy,report_shas):
    match_cost(value,seconds,energy,report_shas,'summary_sha256')
    mac=value['actual_NN_MAC600'];serial=value['charged_serial_NN_MAC600']
    residual=value['parallel_residual_NN_MAC600'];position=value['native_position_MAC600_not_in_selected_timing_spans']
    records=value['per_call_first_repeat']
    if (not value['no_GPU_or_neural_forwards'] or not value['no_new_time_or_power']
            or value['complete_scalar_OP_claim'] or value['native_mergers_executed_per_query']!=1
            or value['auxiliary_deepstack_mergers_executed']!=0 or value['offline_gallery_inference_included']
            or value['physical_optical_dense_equivalent_OP'] is not None
            or len(records)!=200 or mac!=serial+residual+position
            or any(value[key]!=3*sum(r[record_key] for r in records) for key,record_key in [
                ('actual_NN_MAC600','actual_MAC'),('charged_serial_NN_MAC600','selected_serial_MAC'),
                ('parallel_residual_NN_MAC600','parallel_residual_MAC'),
                ('native_position_MAC600_not_in_selected_timing_spans','native_position_MAC_not_in_selected_spans')])):
        raise ValueError('Original one-merger student NN partition differs')
    return dict(OP600=2*mac,TOP_per_input=2*mac/600/1e12,TOPS=2*mac/1e12/seconds,
        TOPS_per_W=2*mac/1e12/energy,charged_serial_OP600=2*serial,
        parallel_residual_OP600=2*residual,native_position_OP600_not_in_selected_timing_spans=2*position,
        physical_optical_dense_equivalent_OP=None,not_complete_scalar_OP=True,
        numerator_scope='all actual NN dense MAC, not only selected serial NN; optical propagation/FFT separate')


def attach(row,kind):
    settings={
        'spatial_qwen':('lgvq_spatial_qwen_saved_shape_MAC','qwen','derivation_commit',
            'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/lgvq_spatial_input_a100_20261024_01'),
        'image_student':('abo_image_student_saved_shape_MAC','ours','source_commit',
            'LightGenV2/tasks/t08_abo_image_text_retrieval/runs/simulation/a100_image_student_input_inclusive_20261025_04'),
        't07_qwen':('t07_qwen_saved_shape_MAC','qwen','derivation_commit',
            'LightGenV2/tasks/t07_abo_image_retrieval/runs/simulation/a100_input_inclusive_20261024_qwen01')}
    stem,model,commit_key,root=settings[kind]
    local=HERE/(stem+'_01.json');server=HERE/(stem+'_server_01.json')
    value=independent_pair(local,server,commit_key);cost=row[model]
    if kind=='t07_qwen':
        report_shas=[r['report_sha256'] for r in cost['repeats']];metrics=cost['aggregate']
    else:
        report_shas=[r['report_sha256'] for r in cost['raw_references']];metrics=cost
    seconds=metrics['sample_ms']*.6;energy=metrics['sample_j']*600
    for repeat,ref in enumerate(value['cost_references'],1):
        folder=REPO/root/f'{model}_repeat{repeat:02d}'
        if ref.get('run',ref.get('path'))!=folder.relative_to(REPO).as_posix():
            raise ValueError('Fixed qualified repeat enrollment differs')
        files=[('summary.json','summary_sha256'),('calls.csv','calls_sha256'),('stages.csv','stage_occurrences_sha256')] if kind=='image_student' else [
            ('report.json','report_sha256'),('calls.csv','calls_sha256')]
        if kind=='spatial_qwen':files.append(('input_audit.json','input_audit_sha256'))
        for filename,key in files:
            if sha(folder/filename)!=ref[key]:raise ValueError('Original measured or shape raw changed')
    if kind=='image_student':
        if sha(REPO/value['metadata_path'])!=value['metadata_sha256']:raise ValueError('Original checkpoint shape metadata changed')
        result=image_student_subtotal(value,seconds,energy,report_shas)
    else:
        result=qwen_subtotal(value,seconds,energy,report_shas)
        if kind=='t07_qwen' and (value['model_LM_head_executed'] or value['offline_gallery_inference_included']):
            raise ValueError('Unexecuted LM head/offline gallery mixed into query cost')
        if kind=='spatial_qwen':result['equivalent_TOP_per_input_frame']=result['TOP_per_input']/4
    result.update(local_receipt=local.relative_to(REPO).as_posix(),local_receipt_sha256=sha(local),
        independent_server_receipt=server.relative_to(REPO).as_posix(),independent_server_receipt_sha256=sha(server),
        same_formula_and_raw_both_ends=True,checkout_commit_may_differ_without_changing_any_formula_or_raw=True,
        raw_derivation=value)
    cost['derived_NN_dense_MAC_subtotal']=result
    row['ops_status']={
        'spatial_qwen':'Qwen original spatial full-native/pooled-vocabulary/head NN subtotal independently derived from saved600 shapes; Ours pending; explicit non-MAC',
        'image_student':'Both NN subtotals verified; Ours one real main merger, actual/selected/residual/position separated; physical optical dense-equivalent/non-MAC separate',
        't07_qwen':'Qwen original dynamic native graph +1600x64 query search NN subtotal independently derived; Ours/physical optical pending; explicit non-MAC'}[kind]
