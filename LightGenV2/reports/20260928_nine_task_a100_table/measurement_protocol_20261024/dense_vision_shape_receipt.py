"""Exact original LSP/SALICON Qwen dense-MAC and measured cost binding."""
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def matched(group,seconds,energy):
    refs=group['cost_references'];repeats=group['repeat_results'];op=group['MAC_OP600']
    if (group['calls']!=600 or len(refs)!=3 or [r['repeat'] for r in repeats]!=[1,2,3]
            or any(len(r['records'])!=200 or r['MAC200']!=sum(v['MAC'] for v in r['records']) for r in repeats)
            or any(v['MAC']!=sum(v['MAC_terms'].values()) for r in repeats for v in r['records'])
            or op!=2*sum(r['MAC200'] for r in repeats)
            or abs(sum(r['seconds200'] for r in refs)-seconds)>1e-8
            or abs(sum(r['joules200'] for r in refs)-energy)>1e-7
            or not group['all_native24_vision_blocks_and_four_mergers']
            or group['language_or_vocabulary_projection_executed']):
        raise ValueError('Original full-Vision MAC/qualified cost scope differs')
    return dict(OP600=op,TOP_per_input=op/600/1e12,TOPS=op/1e12/seconds,
        TOPS_per_W=op/1e12/energy,not_complete_scalar_OP=True)


def attach(row,task):
    from dense_vision_qwen_saved_shape_mac import GROUPS
    local=HERE/'dense_vision_qwen_saved_shape_MAC_01.json';server=HERE/'dense_vision_qwen_saved_shape_MAC_server_01.json'
    value=json.loads(local.read_text())
    if (value!=json.loads(server.read_text()) or value['GPU_forwards']!=0 or value['new_neural_forwards']!=0
            or value['new_timing_or_power_used'] or value['complete_scalar_OP_claim']):
        raise ValueError('Independent zero-forward full-Vision formula agreement differs')
    group=next(g for g in value['groups'] if g['task']==task);directory,run,_=GROUPS[task]
    for repeat,ref in enumerate(group['cost_references'],1):
        folder=REPO/'LightGenV2/tasks'/directory/'runs/simulation'/run/f'qwen_repeat{repeat:02d}'
        if ref['run']!=folder.relative_to(REPO).as_posix():raise ValueError('Original fixed qualified repeat differs')
        for name,key in [('identity.json','identity_sha256'),('samples.csv','samples_sha256'),('summary.json','summary_sha256')]:
            if sha(folder/name)!=ref[key]:raise ValueError('Original native/head inputs or cost changed')
    cost=row['qwen'];result=matched(group,cost['sample_ms']*.6,cost['sample_j']*600)
    result.update(local_receipt=local.relative_to(REPO).as_posix(),local_receipt_sha256=sha(local),
        independent_server_receipt=server.relative_to(REPO).as_posix(),independent_server_receipt_sha256=sha(server),
        convention=value['convention'],non_MAC_not_in_subtotal=value['non_MAC_not_in_subtotal'],raw_derivation=group)
    cost['derived_NN_dense_MAC_subtotal']=result
    row['ops_status']='Qwen original full24-Vision/all4mergers + trained decoder dense-MAC independently derived on both CPUs from saved600 shapes; no Language/vocabulary; Ours/physical optical separate pending'
