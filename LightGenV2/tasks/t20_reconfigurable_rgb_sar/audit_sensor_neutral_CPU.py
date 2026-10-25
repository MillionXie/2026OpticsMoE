"""Locked v2 best/last exact native graph CPU replay, no re-selection."""
import argparse,json,time,subprocess
from pathlib import Path
import numpy as np
import torch
from .unified_native import build_unified,FRONTEND_SHA256
from .sensor_neutral import CONTRACT,frontend,amplitude
from .train_sensor_neutral import infer
from LightGenV2.tasks.t21_continual_learning_hardware.train_native import sha,put,tensor_sha
from .train_unified_static import load_arrays
from .static_four import build_static_four,geometry,CONTRACT as FOUR_CONTRACT
from .diagnose_frontend_transfer import checked_features


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(__doc__)
    for n in ('run','data','frontend','out'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2);receipt=json.loads((a.run/'completion.json').read_text());protocol=json.loads((a.run/'protocol.json').read_text());stored=json.loads((a.run/'selected_test.json').read_text());contract=receipt['contract'];four=contract==FOUR_CONTRACT;assert receipt['passed'] and contract in (CONTRACT,FOUR_CONTRACT) and protocol['contract']==contract and not protocol['selection_uses_TEST'];front_sha=protocol['frontend_sha256'] if four else FRONTEND_SHA256;factory=build_static_four if four else build_unified;a.out.mkdir(parents=True,exist_ok=False);started=time.monotonic();data=load_arrays(a.data,'test');front=checked_features(data,a.frontend,front_sha) if four else frontend(data,a.frontend);rows=[]
    if data['ids'].tolist()!=stored['ids'] or data['labels'].tolist()!=stored['labels'] or data['domains'].tolist()!=stored['domains']:raise ValueError('Full TEST identity changed')
    for role in ('best','last'):
        path=a.run/(role+'_checkpoint.pt');assert sha(path)==receipt[role+'_sha256'];cp=torch.load(path,map_location='cpu',weights_only=False);assert cp['contract']==contract and cp['frontend_sha256']==front_sha;model=factory(cp['architecture']);model.load_state_dict(cp['model'],strict=True);model.eval();reload=factory(cp['architecture']);reload.load_state_dict(cp['model'],strict=True);reload.eval()
        if four:
            assert geometry(model)==protocol['geometry']
            assert receipt['frontend_SHA256']==front_sha and all(tensor_sha(model.state_dict()[n])==v for n,v in protocol['frozen_state_sha256'].items())
        if any(not torch.equal(v,cp['model'][n]) for n,v in model.state_dict().items()):raise ValueError('State reload differs')
        if tensor_sha(model.display_gains)!=protocol['fixed_gain_sha256']:raise ValueError('Frozen TRAIN gain differs')
        phases={n:tensor_sha(model.phase_fraction(v)) for n,v in model.named_parameters() if 'phase' in n};report=infer(model,data,front,torch.device('cpu'))
        for start in range(0,24,4):
            x=amplitude(data,front,np.arange(start,start+4),torch.device('cpu'));torch.testing.assert_close(model(x)['logits'],reload(x)['logits'],atol=0,rtol=0)
        parity=None
        if role=='best':
            changed=np.flatnonzero(np.asarray(report['predictions'])!=np.asarray(stored['predictions']));error=float(np.abs(np.asarray(report['probabilities'])-np.asarray(stored['probabilities'])).max());parity=dict(argmax_disagreements=len(changed),differing_indices=changed.tolist(),maximum_probability_error=error,legacy_probability_tolerance=.002,legacy_parity_passed=not len(changed) and error<=.002)
        row=dict(role=role,checkpoint_SHA256=sha(path),scores=report['scores'],CPU_phase_SHA256=phases,comparison_to_GPU=parity);rows.append(row);put(a.out/(role+'.json'),dict(**row,full_result=report));put(a.out/'progress.json',dict(completed_roles=len(rows),updated=time.time()))
    put(a.out/'report.json',dict(passed=True,strict_CPU=True,rows=rows,original_PT_unchanged=True,no_checkpoint_reselection=True,SDK=False,TEST_gradients=False,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),seconds=time.monotonic()-started))


if __name__=='__main__':main()
