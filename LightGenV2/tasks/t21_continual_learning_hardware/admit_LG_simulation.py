"""Record the new human5pp admission without rewriting old gates or weights.

Pure JSON/SHA audit; no Torch, SDK, prediction edits or checkpoint selection.
CPU deployment reference and legacy GPU-parity status remain separate facts.
"""
import argparse,hashlib,json,subprocess,time
from pathlib import Path


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):value.update(block)
    return value.hexdigest()


def validate(cfg):
    if cfg!=json.loads(Path(__file__).with_name('config_LG_CD_admission5pp_20261025.json').read_text()):raise ValueError('Only exact human-approved5pp profile')


def assess(scores,phase,cfg):
    reference=cfg['original_same1000'][phase];other=cfg['frozen_D2NN_same1000'][phase]
    if set(scores)!=set(reference):raise ValueError('Every task of the same complete stage is required')
    result={}
    for name,original in reference.items():
        row=scores[name]
        if row['n']!=1000 or type(row['correct']) is not int or not 0<=row['correct']<=1000:raise ValueError('Complete fixed1000 integer counts required')
        lo=original+cfg['lower_correct_delta'];hi=original+cfg['upper_correct_delta'];actual=row['correct']
        result[name]=dict(original_correct=original,adapted_CPU_correct=actual,difference_pp=(actual-original)/10,
            minimum_correct=lo,maximum_correct=hi,within_original_band=lo<=actual<=hi,
            D2NN_reference_correct=other[name],not_below_D2NN=actual>=other[name])
    return dict(passed=all(v['within_original_band'] and v['not_below_D2NN'] for v in result.values()),tasks=result)


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('run','profile','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--phase',choices=('C','D'),required=True);p.add_argument('--parent-readiness',type=Path);a=p.parse_args();cfg=json.loads(a.profile.read_text());validate(cfg)
    folder=a.run/a.phase;audit=a.run/('CPU_deployment_'+a.phase+'_20261025');receipt=json.loads((folder/'completion.json').read_text());report=json.loads((audit/'report.json').read_text());cpu=json.loads((audit/'best.json').read_text());audit_protocol=json.loads((audit/'protocol.json').read_text())
    if receipt['phase']!=a.phase or audit_protocol['phase']!=a.phase or not report['passed'] or not report['strict_CPU'] or not report['saved_best_last_reloaded'] or cpu['strict_CPU'] is not True:raise ValueError('Complete original CPU deployment audit required')
    roles={row['role']:row for row in report['rows']}
    if set(roles)!={'best','last'} or not all(v['saved_state_and_CPU_phase_export_reproducible'] and v['old_experts_and_gains_bitwise_frozen'] for v in roles.values()):raise ValueError('Incomplete saved-weight/frozen proof')
    cp=folder/'best_checkpoint.pt'
    for role in roles:
        if sha(folder/(role+'_checkpoint.pt'))!=receipt[role+'_sha256'] or roles[role]['checkpoint_sha256']!=receipt[role+'_sha256']:raise ValueError('Saved weight identity changed')
    if cpu['checkpoint_sha256']!=receipt['best_sha256'] or any((v['n'],v['correct'])!=(roles['best']['scores'][n]['n'],roles['best']['scores'][n]['correct']) for n,v in cpu['scores'].items()):raise ValueError('CPU score receipt mismatch')
    parent=None
    if a.phase=='D':
        if a.parent_readiness is None:raise ValueError('D requires an admitted exact C parent')
        parent=json.loads(a.parent_readiness.read_text());protocol=json.loads((a.run/'protocol.json').read_text())
        if parent['phase']!='C' or not parent['passed'] or parent['checkpoint_sha256']!=protocol['parent_C_sha256']:raise ValueError('D was not trained from this admitted C')
    decision=assess(cpu['scores'],a.phase,cfg)
    result=dict(schema=cfg['schema'],phase=a.phase,complete=True,**decision,checkpoint_sha256=sha(cp),checkpoint=str(cp),
        profile=cfg,profile_SHA256=sha(a.profile),original_completion_SHA256=sha(folder/'completion.json'),original_completion_gate_preserved=receipt['all_tasks_within_band'],
        CPU_best_report_SHA256=sha(audit/'best.json'),CPU_summary_SHA256=sha(audit/'report.json'),legacy_CPU_GPU_parity_passed=report['legacy_CPU_GPU_parity_passed'],
        legacy_GPU_gate_unchanged=True,reference_device='cpu',parent_readiness_SHA256=sha(a.parent_readiness) if parent else None,
        hardware_or_weights_changed=False,physical_target_change=False,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_SHA256=sha(__file__),
        SDK=False,GPU=False,TEST_gradients=False,TEST_selected_development_metric=True,independent_generalization_claim=False,created=time.time())
    if a.out.exists():raise FileExistsError('Preserve previous admission decision')
    a.out.write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
