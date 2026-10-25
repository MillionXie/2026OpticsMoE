"""Export physical-mask-equivalent PT away from float sigmoid bin boundaries.

No optimization or PT selection. Preserve CPU canonical8bit phase codes exactly,
all nonphase tensors bitwise, the original graph and frozen frontend. Exported
raw logits sit at quantizer bin centers; source weights are never overwritten.
"""
import argparse,json,copy,subprocess,time
from pathlib import Path
import torch
import numpy as np
from .unified_native import build_unified
from .domain_frontend_probe import features,FRONTEND_SHA,CONTRACT
from .train_sensor_neutral import infer
from .train_unified_static import load_arrays
from LightGenV2.tasks.t21_continual_learning_hardware.train_native import put,sha,tensor_sha,gpu_gate


def phase_codes(raw):return torch.floor(raw.sigmoid()*256).clamp(0,255).to(torch.uint8)


def canonicalize(state):
    result={n:v.detach().cpu().clone() for n,v in state.items()};proof={}
    for name,value in result.items():
        if 'phase' not in name:continue
        code=phase_codes(value);center=torch.logit((code.float()+.5)/256).to(value.dtype)
        if not torch.equal(code,phase_codes(center)):raise RuntimeError('Physical phase code changed')
        result[name]=center;proof[name]=dict(original_raw_SHA256=tensor_sha(value),canonical_raw_SHA256=tensor_sha(center),physical_phase_code_SHA256=tensor_sha(code))
    if any(not torch.equal(result[n],v.cpu()) for n,v in state.items() if 'phase' not in n):raise RuntimeError('Nonphase state changed')
    return result,proof


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(__doc__)
    for n in ('run','data','frontend','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--gpu-uuid',required=True);a=p.parse_args();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;receipt=json.loads((a.run/'completion.json').read_text());audit=json.loads((a.run.parent/(a.run.name+'_CPU')/'best.json').read_text());path=a.run/'best_checkpoint.pt';parent_sha=sha(path)
    assert parent_sha==receipt['best_sha256']==audit['checkpoint_SHA256'] and receipt['contract']==CONTRACT and receipt['frontend_SHA256']==FRONTEND_SHA
    cp=torch.load(path,map_location='cpu',weights_only=False);state,proof=canonicalize(cp['model']);export=dict(cp,model=state,canonical_export=True,original_checkpoint_SHA256=parent_sha,export_rule='CPU phase code unchanged; raw logit moved to center of existing8bit bin; no training')
    gpu=gpu_gate(dict(allowed_GPU_UUIDs=[a.gpu_uuid],per_process_GPU_memory_fraction=.2),a.gpu_uuid);device=torch.device('cuda:0');model=build_unified(cp['architecture']).to(device);model.load_state_dict(state,strict=True);model.eval()
    for name,param in model.named_parameters():
        if 'phase' in name and not torch.equal(phase_codes(param).cpu(),phase_codes(cp['model'][name])):raise RuntimeError('GPU phase code differs from canonical physical pattern')
    a.out.mkdir(parents=True,exist_ok=False);started=time.monotonic();target=a.out/'model.pt';torch.save(export,target);digest=sha(target);data=load_arrays(a.data,'test');front=features(data,a.frontend);cuda=infer(model,data,front,device);put(a.out/'GPU_report.json',cuda);del model;torch.cuda.empty_cache();model=build_unified(cp['architecture']);model.load_state_dict(torch.load(target,map_location='cpu',weights_only=False)['model'],strict=True);model.eval();cpu=infer(model,data,front,torch.device('cpu'));put(a.out/'CPU_report.json',cpu)
    original=audit['full_result'];assert original['ids']==cpu['ids']==cuda['ids'] and original['labels']==cpu['labels']==cuda['labels'];old_error=float(np.abs(np.asarray(cpu['probabilities'])-np.asarray(original['probabilities'])).max());error=float(np.abs(np.asarray(cpu['probabilities'])-np.asarray(cuda['probabilities'])).max());old_changes=int((np.asarray(cpu['predictions'])!=np.asarray(original['predictions'])).sum());new_changes=int((np.asarray(cpu['predictions'])!=np.asarray(cuda['predictions'])).sum());passed=old_error<=1e-7 and not old_changes and error<=.002 and not new_changes
    if sha(path)!=parent_sha or sha(target)!=digest:raise RuntimeError('Immutable PT changed')
    manifest=dict(passed=passed,case=a.run.name,original_run=str(a.run),original_checkpoint_SHA256=parent_sha,export_checkpoint_SHA256=digest,frontend_SHA256=FRONTEND_SHA,original_graph_unchanged=True,physical_phase_codes_preserved=True,all_nonphase_tensors_preserved=True,CPU_against_original=dict(max_probability_error=old_error,argmax_disagreements=old_changes),GPU_against_CPU=dict(max_probability_error=error,argmax_disagreements=new_changes,legacy_tolerance=.002),scores=cpu['scores'],phase_proof=proof,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256=sha(__file__),gpu=gpu,TEST_gradients=False,checkpoint_reselection=False,probability_editing=False,SDK=False,seconds=time.monotonic()-started);put(a.out/'manifest.json',manifest)
    if not passed:raise RuntimeError('Physical-equivalent export did not meet original numerical gate')


if __name__=='__main__':main()
