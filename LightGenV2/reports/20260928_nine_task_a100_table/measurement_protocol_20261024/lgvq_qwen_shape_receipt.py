"""Bind offline original-Qwen MAC numerators to the exact qualified cost triple."""
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def matched(group,seconds,energy,report_shas):
    op=group['MAC_OP600'];refs=group['cost_references']
    if (group['calls']!=600 or op!=2*sum(r['MAC200'] for r in group['repeat_results'])
            or any(r['MAC200']!=sum(v['MAC'] for v in r['records']) for r in group['repeat_results'])
            or abs(sum(r['seconds200'] for r in refs)-seconds)>1e-8
            or abs(sum(r['joules200'] for r in refs)-energy)>1e-7
            or [r['report_sha256'] for r in refs]!=report_shas):
        raise ValueError('Original video MAC/cost triple differs')
    return dict(OP600=op,TOP_per_input=op/600/1e12,TOPS=op/1e12/seconds,TOPS_per_W=op/1e12/energy,
        equivalent_TOP_per_input_frame=op/2400/1e12,not_complete_scalar_OP=True)


def attach(cost,group_name):
    local=HERE/'lgvq_qwen_network_shape_MAC_02.json';server=HERE/'lgvq_qwen_network_shape_MAC_server_02.json'
    value=json.loads(local.read_text());independent=json.loads(server.read_text())
    if (value['GPU_forwards']!=0 or value['new_neural_forwards']!=0 or value['new_timing_or_power_used']
            or value['complete_scalar_OP_claim']):raise ValueError('Original zero-forward dense-MAC scope differs')
    groups={g['group']:g for g in value['groups']};others={g['group']:g for g in independent['groups']}
    group=groups[group_name];lab=group_name.startswith('laboratory')
    if lab and group!=others.get(group_name):raise ValueError('Independent original lab formula agreement differs')
    # Recheck original byte-bound processor/config/cost evidence without redoing inference.
    for ref in group['cost_references']:
        folder=REPO/ref['path']
        for filename,key in [('report.json','report_sha256'),('calls.csv','calls_sha256'),('input_audit.json','input_audit_sha256')]:
            if sha(folder/filename)!=ref[key]:raise ValueError('Original formula/cost raw changed')
    report_shas=cost.get('report_sha256') or [r['report_sha256'] for r in cost['raw_references']]
    result=matched(group,cost.get('seconds_total',cost['sample_ms']*.6),cost['sample_j']*600,report_shas)
    result.update(local_receipt=local.relative_to(REPO).as_posix(),local_receipt_sha256=sha(local),
        independent_lab_replay_receipt=server.relative_to(REPO).as_posix(),independent_lab_replay_sha256=sha(server),
        this_group_independently_replayed_on_lab=lab,
        source='original native graph + existing all600 video processor shapes; includes pooled full vocabulary and trained five-row head',
        raw_derivation=group)
    cost['derived_NN_dense_MAC_subtotal']=result
