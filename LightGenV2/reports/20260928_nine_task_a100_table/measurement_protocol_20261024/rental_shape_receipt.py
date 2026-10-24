"""Bind original DS/Kimi offline dense-MAC work to their sealed cost triples.

This is a local original-shape derivation, not a second server replay, complete
scalar-operation count, hardware instruction counter, or new rental inference.
"""
import hashlib
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def matched(group,seconds,energy,report_shas):
    repeats=group['repeat_results'];refs=group['cost_references']
    if (group['calls']!=600 or len(repeats)!=3 or len(refs)!=3
            or [r['repeat'] for r in repeats]!=[1,2,3]
            or any(len(r['records'])!=200 for r in repeats)
            or any(v['MAC']!=sum(v['MAC_terms'].values())
                   or v['logical_unpadded_MAC']+v['FA2_zero_padded_value_dense_MAC']!=v['MAC']
                   for r in repeats for v in r['records'])
            or any(r['MAC200']!=sum(v['MAC'] for v in r['records']) for r in repeats)
            or group['MAC_OP600']!=2*sum(r['MAC200'] for r in repeats)
            or group['FA2_zero_padding_MAC600']!=sum(v['FA2_zero_padded_value_dense_MAC'] for r in repeats for v in r['records'])
            or abs(sum(r['seconds200'] for r in refs)-seconds)>1e-8
            or abs(sum(r['joules200'] for r in refs)-energy)>1e-7
            or [r['report_sha256'] for r in refs]!=report_shas):
        raise ValueError('Original DS/Kimi dense-MAC work or exact cost triple differs')
    op=group['MAC_OP600'];pad=group['FA2_zero_padding_MAC600']*2
    return dict(OP600=op,TOP_per_input=op/600/1e12,TOPS=op/1e12/seconds,
        TOPS_per_W=op/1e12/energy,equivalent_TOP_per_input_frame=op/2400/1e12,
        logical_unpadded_OP600=op-pad,zero_padded_value_dense_OP600=pad,
        not_complete_scalar_OP=True,not_hardware_instruction_count=True)


def attach(cost,model,cap):
    from rental_original_shape_mac import HEADS,METADATA,ROOT,SOURCE
    local=HERE/'rental_original_network_shape_MAC_01.json';value=json.loads(local.read_text())
    if (value['status']!='ORIGINAL_RUNTIME_AND_SEALED_INPUT_SHAPES_DENSE_MAC_DERIVED'
            or value['GPU_forwards']!=0 or value['new_neural_forwards']!=0
            or value['new_timing_or_power_used'] or not value['no_rental_connection']
            or value['complete_scalar_OP_or_hardware_instruction_count_claim']):
        raise ValueError('Original zero-forward derivation scope differs')
    blob=subprocess.check_output(['git','show',value['derivation_commit']+':'+
        (HERE/'rental_original_shape_mac.py').relative_to(REPO).as_posix()],cwd=REPO)
    if hashlib.sha256(blob).hexdigest()!=value['derivation_source_sha256']:
        raise ValueError('Published original derivation source differs')
    for key,(_,pin) in SOURCE.items():
        if value['source_facts']['public_sources'][key]['sha256']!=pin:
            raise ValueError('Original runtime source pin differs')
    group=next(g for g in value['groups'] if (g['model'],g['cap'])==(model,cap))
    if (group['config_sha256']!=sha(METADATA/model/'config.json')
            or group['head_sha256']!=HEADS[model]):raise ValueError('Original graph assets differ')
    for repeat,ref in enumerate(group['cost_references'],1):
        folder=ROOT/f'{model}_cap{cap}_repeat{repeat:02d}'
        if ref['run']!=folder.relative_to(REPO).as_posix():raise ValueError('Fixed qualified run map differs')
        for filename,key in [('report.json','report_sha256'),('calls.csv','calls_sha256'),('input_audit.json','input_audit_sha256')]:
            if sha(folder/filename)!=ref[key]:raise ValueError('Original formula/cost raw changed')
    result=matched(group,cost['seconds_total'],cost['sample_j']*600,cost['report_sha256'])
    result.update(local_receipt=local.relative_to(REPO).as_posix(),local_receipt_sha256=sha(local),
        independent_server_formula_replay=False,
        source='original runtime SHA-bound graph + all600 sealed input shapes; active routed/shared experts and last-vector full vocabulary/quality head',
        attention_scope=value['DS_attention'] if model=='deepseek' else value['Kimi_grids'],
        convention=value['convention'],non_MAC_not_in_subtotal=value['non_MAC_not_in_subtotal'],
        raw_derivation=group)
    cost['derived_NN_dense_MAC_subtotal']=result
