"""Saved best/last complete original-graph CPU audit, never score reselection."""
import argparse,json,time,subprocess
from pathlib import Path
import numpy as np
import torch
from .unified_native import build_unified,CONTRACT,FRONTEND_SHA256
from .train_unified_static import load_arrays,cache_frontend,batch_amplitude,evaluate
from LightGenV2.tasks.t21_continual_learning_hardware.train_native import sha,put,tensor_sha


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('run','data','frontend','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2);receipt=json.loads((a.run/'completion.json').read_text());protocol=json.loads((a.run/'protocol.json').read_text());stored=json.loads((a.run/'selected_test.json').read_text())
    if not receipt['passed'] or protocol['frontend_sha256']!=FRONTEND_SHA256 or protocol['selection_uses_TEST'] is not False:raise ValueError('Complete locked original run required')
    a.out.mkdir(parents=True,exist_ok=False);started=time.monotonic();data=load_arrays(a.data,'test');features=cache_frontend(data,a.frontend);rows=[]
    if data['ids'].tolist()!=stored['ids'] or data['labels'].tolist()!=stored['labels'] or data['domains'].tolist()!=stored['domains']:raise ValueError('Original full evaluation identity differs')
    for role in ('best','last'):
        path=a.run/(role+'_checkpoint.pt')
        if sha(path)!=receipt[role+'_sha256']:raise ValueError('Locked checkpoint changed')
        saved=torch.load(path,map_location='cpu',weights_only=False)
        if saved['contract']!=CONTRACT or saved['frontend_sha256']!=FRONTEND_SHA256:raise ValueError('Changed input/frontend contract')
        model=build_unified(saved['architecture']);model.load_state_dict(saved['model'],strict=True);model.eval();reload=build_unified(saved['architecture']);reload.load_state_dict(saved['model'],strict=True);reload.eval()
        if any(not torch.equal(v,saved['model'][n]) for n,v in model.state_dict().items()):raise ValueError('Saved graph not loaded exactly')
        phase={n:tensor_sha(model.phase_fraction(v)) for n,v in model.named_parameters() if 'phase' in n}
        if tensor_sha(model.display_gains)!=protocol['fixed_gain_sha256']:raise ValueError('TRAIN gain changed')
        scores=evaluate(model,data,features,torch.device('cpu'),4)
        for start in range(0,24,4):
            amp=batch_amplitude(data,features,np.arange(start,start+4),torch.device('cpu'))
            if not torch.equal(model(amp)['logits'],reload(amp)['logits']):raise ValueError('Saved original CPU graph is not reproducible')
        differences=None
        if role=='best':
            pred=np.asarray(scores['predictions']);gpu=np.asarray(stored['predictions']);error=float(np.abs(np.asarray(scores['probabilities'])-np.asarray(stored['probabilities'])).max());changed=np.flatnonzero(pred!=gpu)
            differences=dict(argmax_disagreements=len(changed),differing_local_indices=changed.tolist(),maximum_probability_error=error,legacy_probability_tolerance=.002,legacy_parity_passed=not len(changed) and error<=.002)
        put(a.out/(role+'.json'),dict(result=scores,checkpoint_sha256=sha(path),CPU_phase_SHA256=phase,strict_CPU_graph_reloaded=True,comparison_to_original_GPU=differences))
        rows.append(dict(role=role,checkpoint_sha256=sha(path),scores=scores['scores'],comparison_to_GPU=differences,CPU_phase_SHA256=phase));put(a.out/'progress.json',dict(completed_roles=len(rows),updated=time.time()))
    put(a.out/'report.json',dict(passed=True,strict_CPU=True,rows=rows,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256=sha(__file__),
        selected_original_PT_unchanged=True,no_checkpoint_reselection=True,no_probability_or_label_editing=True,CPU_GPU_legacy_gate_unchanged=True,SDK=False,TEST_gradients=False,historical_TEST_pool_reuse=True,seconds=time.monotonic()-started))


if __name__=='__main__':main()
