"""Full CPU replay of the locked shared-domain-frontend optical control."""
import argparse,json,subprocess,time
from pathlib import Path
import torch
import numpy as np
from .domain_frontend_probe import features,CONTRACT,FRONTEND_SHA
from .unified_native import build_unified
from .static_four import build_static_four,CONTRACT as FOUR_CONTRACT,geometry
from .train_unified_static import load_arrays
from .sensor_neutral import amplitude
from .train_sensor_neutral import infer
from LightGenV2.tasks.t21_continual_learning_hardware.train_native import sha,put,tensor_sha


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(__doc__)
    for n in ('run','data','frontend','out'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2);receipt=json.loads((a.run/'completion.json').read_text());protocol=json.loads((a.run/'protocol.json').read_text());stored=json.loads((a.run/'selected_test.json').read_text());contract=receipt['contract'];assert receipt['passed'] and contract in (CONTRACT,FOUR_CONTRACT) and protocol['contract']==contract and not receipt['selection_uses_TEST'];factory=build_static_four if contract==FOUR_CONTRACT else build_unified;data=load_arrays(a.data,'test');front=features(data,a.frontend);a.out.mkdir(parents=True,exist_ok=False);rows=[];started=time.monotonic()
    assert stored['ids']==data['ids'].tolist() and stored['labels']==data['labels'].tolist() and stored['domains']==data['domains'].tolist()
    for role in ('best','last'):
        path=a.run/(role+'_checkpoint.pt');assert sha(path)==receipt[role+'_sha256'];cp=torch.load(path,map_location='cpu',weights_only=False);assert cp['contract']==contract and cp['frontend_sha256']==FRONTEND_SHA;model=factory(cp['architecture']);model.load_state_dict(cp['model'],strict=True);model.eval();reload=factory(cp['architecture']);reload.load_state_dict(cp['model'],strict=True);reload.eval()
        if contract==FOUR_CONTRACT:assert geometry(model)==protocol['geometry']
        assert all(torch.equal(v,cp['model'][n]) for n,v in model.state_dict().items());assert tensor_sha(model.display_gains)==protocol['fixed_gain_sha256'];assert all(tensor_sha(model.state_dict()[n])==v for n,v in protocol['frozen_state_sha256'].items());report=infer(model,data,front,torch.device('cpu'));phases={n:tensor_sha(model.phase_fraction(v)) for n,v in model.named_parameters() if 'phase' in n}
        for start in range(0,24,4):
            x=amplitude(data,front,np.arange(start,start+4),torch.device('cpu'));torch.testing.assert_close(model(x)['logits'],reload(x)['logits'],atol=0,rtol=0)
        parity=None
        if role=='best':
            changed=np.flatnonzero(np.asarray(report['predictions'])!=np.asarray(stored['predictions']));error=float(np.abs(np.asarray(report['probabilities'])-np.asarray(stored['probabilities'])).max());parity=dict(argmax_disagreements=len(changed),differing_indices=changed.tolist(),maximum_probability_error=error,legacy_probability_tolerance=.002,legacy_parity_passed=not len(changed) and error<=.002)
        row=dict(role=role,checkpoint_SHA256=sha(path),scores=report['scores'],CPU_phase_SHA256=phases,comparison_to_GPU=parity);rows.append(row);put(a.out/(role+'.json'),dict(**row,full_result=report));put(a.out/'progress.json',dict(completed_roles=len(rows),updated=time.time()))
    put(a.out/'report.json',dict(passed=True,strict_CPU=True,rows=rows,no_checkpoint_reselection=True,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),SDK=False,TEST_gradients=False,seconds=time.monotonic()-started))


if __name__=='__main__':main()
