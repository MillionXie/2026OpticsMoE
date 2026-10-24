"""Exact cost binding for T07/LSP/SALICON original compact NN subtotals."""
import json
from pathlib import Path
from remaining_shape_receipt import independent_pair,match_cost,sha

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]


def subtotal(value,seconds,energy,report_shas):
    match_cost(value,seconds,energy,report_shas,'summary_sha256')
    if 'actual_NN_MAC_OP600' in value:
        op=value['actual_NN_MAC_OP600'];serial=value['charged_serial_NN_OP600'];residual=value['parallel_residual_NN_OP600']
    else:
        op=2*value['actual_NN_MAC600'];serial=2*value['charged_serial_NN_MAC600'];residual=2*value['parallel_residual_NN_MAC600']
    terms=value['per_query_MAC']
    if (op!=serial+residual or op!=1200*terms['actual_NN_MAC']
            or serial!=1200*terms['charged_serial_NN_MAC'] or residual!=1200*terms['parallel_residual_NN_MAC']
            or terms['charged_serial_NN_MAC']!=sum(terms['serial_NN_terms'].values())
            or terms['parallel_residual_NN_MAC']!=sum(terms['parallel_NN_terms'].values())
            or value['other_executed_digital_dense_MAC600']!=600*sum(terms['other_executed_digital_dense_MAC'].values())):
        raise ValueError('Actual NN/serial/residual/other-digital partition differs')
    repeats=value.get('repeat_results')
    if repeats is not None and ([r['repeat'] for r in repeats]!=[1,2,3]
            or any(len(r['records'])!=200 or r['MAC200']!=sum(v['MAC'] for v in r['records']) for r in repeats)
            or op!=2*sum(r['MAC200'] for r in repeats)):
        raise ValueError('Original all600 repeated NN shapes differ')
    return dict(OP600=op,TOP_per_input=op/600/1e12,TOPS=op/1e12/seconds,TOPS_per_W=op/1e12/energy,
        charged_serial_NN_OP600=serial,parallel_residual_NN_OP600=residual,
        other_executed_digital_dense_OP600_not_in_NN_subtotal=2*value['other_executed_digital_dense_MAC600'],
        physical_optical_dense_equivalent_OP=None,not_complete_scalar_OP=True,
        numerator_scope='all actual NN Linear/Conv/search dense MAC, not only selected serial NN; digital router, optical and FFT separate')


def attach(row,task):
    dense=task!='T07'
    stem='dense_student_saved_shape_MAC' if dense else 't07_student_saved_shape_MAC'
    local=HERE/(stem+'_01.json');server=HERE/(stem+'_server_01.json');all_values=independent_pair(local,server,'source_commit')
    if (all_values['model_or_GPU_forwards']!=0 or all_values['new_time_or_power'] or all_values['complete_scalar_OP_claim']
            or all_values['physical_optical_dense_equivalent_OP'] is not None or not all_values['FFT_not_physical_optical_OP']):
        raise ValueError('Original zero-forward NN-only scope differs')
    if dense:
        from dense_student_saved_shape_mac import GROUPS
        directory,run,_,_=GROUPS[task];value=next(v for v in all_values['groups'] if v['task']==task)
        if value['native_mergers_executed_per_query']!=1 or value['auxiliary_deepstack_mergers_executed']!=0:
            raise ValueError('Original one-merger student differs')
        report_shas=[r['summary_sha256'] for r in sorted((r for r in row['raw_audit']['repeats'] if r['system']=='ours'),key=lambda r:r['repeat'])]
        seconds=row['ours']['sample_ms']*.6;energy=row['ours']['sample_j']*600
        metadata=REPO/value['metadata_path']
    else:
        directory='t07_abo_image_retrieval';run='a100_input_inclusive_20261024_01';value=all_values
        if not value['one_main_merger'] or not value['no_late_RGB_bypass'] or not value['no_RoPE_or_Language_transformer_or_LM_head']:
            raise ValueError('Original compact8225 graph differs')
        report_shas=[r['raw_summary_sha256'] for r in row['ours']['repeats']]
        seconds=row['ours']['aggregate']['total_ms']*.6;energy=row['ours']['aggregate']['phase_energy_j']*600
        metadata=REPO/'LightGenV2/tasks'/directory/'runs/simulation/a100_standalone_saved_shape_metadata_20261025_01/metadata.json'
    if sha(metadata)!=value['metadata_sha256']:raise ValueError('Original tensor shape metadata changed')
    for repeat,ref in enumerate(value['cost_references'],1):
        folder=REPO/'LightGenV2/tasks'/directory/'runs/simulation'/run/f'ours_repeat{repeat:02d}'
        if ref['run']!=folder.relative_to(REPO).as_posix():raise ValueError('Qualified original repeat enrollment differs')
        for filename,key in [('summary.json','summary_sha256'),('samples.csv','samples_sha256'),('stages.csv','stages_sha256'),('identity.json','identity_sha256')]:
            if sha(folder/filename)!=ref[key]:raise ValueError('Original raw input/stages/cost changed')
    result=subtotal(value,seconds,energy,report_shas)
    result.update(local_receipt=local.relative_to(REPO).as_posix(),local_receipt_sha256=sha(local),
        independent_server_receipt=server.relative_to(REPO).as_posix(),independent_server_receipt_sha256=sha(server),
        same_formula_and_raw_both_ends=True,raw_derivation=value)
    row['ours']['derived_NN_dense_MAC_subtotal']=result
    row['ops_status']='Both NN dense-MAC subtotals independently derived/verified from original graph and saved600 shapes; actual/selected/residual/other-digital/physical optical/non-MAC separate; physical optical dense-equivalent still unknown'
