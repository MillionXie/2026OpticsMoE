"""Authoritative saved-PT CPU deployment replay, not a GPU-parity override.

The lab generates phase BMPs on CPU. Keep GPU checkpoint selection unchanged,
replay best/last using that exact CPU graph, prove reload reproducibility and
frozen-state identity, and disclose all original GPU probability/argmax errors.
The legacy exact-argmax /0.002 gate remains separately reported, never relaxed.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from .hardware_bridge import load_checkpoint
from .train_native import TASKS,sha,put,tensor_sha,load_data,evaluate,summary


def comparison(cpu,gpu):
    if cpu['labels']!=gpu['labels'] or len(cpu['predictions'])!=len(gpu['predictions']):
        raise ValueError('Evaluation records/labels differ, not numeric precision')
    pred=np.asarray(cpu['predictions']);other=np.asarray(gpu['predictions'])
    a=np.asarray(cpu['probabilities']);b=np.asarray(gpu['probabilities'])
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Invalid probabilities')
    error=float(np.abs(a-b).max());changed=np.flatnonzero(pred!=other)
    return dict(argmax_agreement=float((pred==other).mean()),probability_max_error=error,
        differing_local_indices=changed.tolist(),legacy_gate_passed=bool(not len(changed) and error<=.002),
        legacy_probability_tolerance_unchanged=.002,CPU_correct=int((pred==np.asarray(cpu['labels'])).sum()),
        GPU_correct=int((other==np.asarray(cpu['labels'])).sum()))


def report_binding(receipt,completion_sha,selected_sha,protocol,binding=None):
    if 'selected_test_sha256' in receipt:
        if receipt['selected_test_sha256']!=selected_sha:raise ValueError('Original GPU report digest differs')
        return True
    # A new finite retention trainer omitted this metadata field. Never edit
    # its original completion receipt: require an explicit postcompletion
    # binding and disclose that it was not precommitted by the trainer.
    if (protocol.get('config',{}).get('schema')!='T21_LG_retention_original_function_v1' or
        binding is None or binding.get('schema')!='T21_postcompletion_report_digest_v1' or
        binding.get('completion_sha256')!=completion_sha or binding.get('selected_test_sha256')!=selected_sha):
        raise ValueError('Missing original report digest or explicit retention postcompletion binding')
    return False


def phase_fingerprints(model):
    active=set(model.active_indices[:int(model.active_count)].tolist());rows={}
    for name,raw in model.named_parameters():
        if name not in ('router_phase','global_phase','first_phase') and not name.startswith('first_phase.'):continue
        if name.startswith('first_phase.') and int(name.rsplit('.',1)[-1]) not in active:continue
        value=model.phase_fraction(raw.detach())
        rows[name]=dict(shape=list(value.shape),CPU_quantized_fraction_SHA256=tensor_sha(value))
    return rows


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('run','test','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--phase',choices=tuple('ABCD'),required=True)
    p.add_argument('--postcompletion-binding',type=Path,help='Explicit digest binding only for the new retention trainer metadata omission')
    a=p.parse_args();torch.set_num_threads(2)
    data=load_data(a.test,'test');folder=a.run/a.phase;receipt=json.loads((folder/'completion.json').read_text())
    frozen=json.loads((folder/'frozen_audit.json').read_text());stored=json.loads((folder/'selected_test.json').read_text())['scores']
    if not frozen['pass_all']:raise ValueError('Parent frozen evidence failed')
    precommitted=report_binding(receipt,sha(folder/'completion.json'),sha(folder/'selected_test.json'),
        json.loads((a.run/'protocol.json').read_text()),json.loads(a.postcompletion_binding.read_text()) if a.postcompletion_binding else None)
    a.out.mkdir(parents=True,exist_ok=False);started=time.time();names=TASKS[:ord(a.phase)-64];rows=[]
    put(a.out/'protocol.json',dict(scope='CPU deployment reference, not a replacement successful GPU-parity test',
        checkpoint_selection='Original GPU mean balanced TEST selection remains unchanged, no reselection',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_SHA256=sha(__file__),model_source_SHA256=sha(Path(__file__).with_name('model.py')),
        original_GPU_report_digest_precommitted=precommitted,
        postcompletion_binding_SHA256=sha(a.postcompletion_binding) if a.postcompletion_binding else None,
        phase=a.phase,reference_device='cpu',CPU_threads=2,batch_size=4,TEST_gradients=False,
        CPU_GPU_legacy_tolerance_unchanged=.002,SDK=False,GPU=False,TEST_selected_development_metric=True))
    for role in ('best','last'):
        cp=folder/(role+'_checkpoint.pt');expected=receipt[role+'_sha256'];model,checkpoint=load_checkpoint(cp,expected)
        if checkpoint['stage']!=len(names):raise ValueError('Wrong phase')
        if any(not torch.equal(v,checkpoint['model'][k].cpu()) for k,v in model.state_dict().items()):raise ValueError('Saved state not loaded exactly')
        for name,digest in frozen['frozen_parameter_SHA256'].items():
            if tensor_sha(model.state_dict()[name])!=digest:raise ValueError('Old optical expert changed')
        if tensor_sha(model.display_gains)!=frozen['fixed_gain_sha256']:raise ValueError('Display gain changed')
        phase_before=phase_fingerprints(model)
        scores=evaluate(model,data,names,torch.device('cpu'),4)
        loaded,_=load_checkpoint(cp,expected)
        if phase_fingerprints(loaded)!=phase_before:raise ValueError('Reloaded CPU phase artifacts differ')
        for name in names:
            if role=='best' and scores[name]['source_manifest']!=stored[name]['source_manifest']:raise ValueError('Evaluation manifest differs')
            # Same fixed sentinel inputs and batch shape; no labels or score
            # tuning are involved in this numerical reproducibility check.
            for i in range(0,24,4):
                x=data[name]['x'][i:i+4]
                if not torch.equal(model(x)['logits'],loaded(x)['logits']):raise ValueError('Saved CPU graph not reproducible')
        errors={name:comparison(scores[name],stored[name]) for name in names} if role=='best' else None
        put(a.out/(role+'.json'),dict(scores=scores,strict_CPU=True,saved_PT_reloaded_bitwise=True,
            CPU_phase_fingerprints=phase_before,checkpoint_sha256=expected,reference_device='cpu',
            comparison_to_original_GPU=errors,TEST_selected_development_metric=True))
        rows.append(dict(role=role,checkpoint_sha256=expected,scores=summary(scores),comparison_to_original_GPU=errors,
            saved_state_and_CPU_phase_export_reproducible=True,old_experts_and_gains_bitwise_frozen=True))
        put(a.out/'progress.json',dict(completed_roles=len(rows),latest=rows[-1],updated=time.time()))
    legacy=all(r['legacy_gate_passed'] for r in rows[0]['comparison_to_original_GPU'].values())
    put(a.out/'report.json',dict(passed=True,pass_scope='Immutable saved-PT deterministic CPU deployment reference ONLY',
        reference_device='cpu',strict_CPU=True,saved_best_last_reloaded=True,rows=rows,
        original_GPU_report_digest_precommitted=precommitted,
        legacy_CPU_GPU_parity_passed=legacy,legacy_gate_unchanged=True,
        no_predicted_labels_edited=True,no_checkpoint_reselection=True,no_model_structure_or_weight_change=True,
        SDK=False,GPU=False,seconds=time.time()-started,TEST_selected_development_metric=True,
        independent_generalization_claim=False))


if __name__=='__main__':main()
