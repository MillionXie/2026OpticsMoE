"""Deterministic physical-code-equivalent export for the existing static-four graph.

Reuses the published quantizer-bin-centre representation, not optimization.
All CPU physical phase codes and all nonphase tensors must be preserved.
Full CPU replay must match the source CPU report, with no score-based selection.
"""
import argparse,json,subprocess,time
from pathlib import Path
import numpy as np
import torch
from .export_canonical_phase import canonicalize,phase_codes
from .static_four import build_static_four,geometry,CONTRACT
from .diagnose_frontend_transfer import checked_features
from .train_unified_static import load_arrays
from .train_sensor_neutral import infer
from LightGenV2.tasks.t21_continual_learning_hardware.train_native import sha,put,tensor_sha,gpu_gate


def compare(reference,actual,tolerance):
    for key in ('ids','labels','domains'):
        if reference[key]!=actual[key]:raise ValueError('Replay identities changed')
    delta=np.flatnonzero(np.asarray(reference['predictions'])!=np.asarray(actual['predictions']))
    error=float(np.abs(np.asarray(reference['probabilities'])-np.asarray(actual['probabilities'])).max())
    return dict(argmax_disagreements=len(delta),differing_indices=delta.tolist(),maximum_probability_error=error,tolerance=tolerance,passed=not len(delta) and error<=tolerance)


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(__doc__)
    for name in ('run','data','frontend','out'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--gpu-uuid',required=True);a=parser.parse_args()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    receipt=json.loads((a.run/'completion.json').read_text());protocol=json.loads((a.run/'protocol.json').read_text());audit=json.loads((a.run.parent/(a.run.name+'_CPU')/'best.json').read_text());path=a.run/'best_checkpoint.pt';original_sha=sha(path)
    assert receipt['contract']==protocol['contract']==CONTRACT and receipt['best_sha256']==audit['checkpoint_SHA256']==original_sha
    assert not protocol['selection_uses_TEST'] and not protocol['TEST_gradients'] and not protocol['TEST_teacher']
    assert sha(a.frontend)==receipt['frontend_SHA256']==protocol['frontend_sha256'] and sha(a.data/'manifest.json')==protocol['data_manifest_sha256']
    cp=torch.load(path,map_location='cpu',weights_only=False);assert cp['contract']==CONTRACT and cp['frontend_sha256']==protocol['frontend_sha256']
    for name,value in protocol['frozen_state_sha256'].items():assert tensor_sha(cp['model'][name])==value
    state,proof=canonicalize(cp['model']);model=build_static_four(cp['architecture']);model.load_state_dict(state,strict=True);model.eval();assert geometry(model)==protocol['geometry']
    assert tensor_sha(model.display_gains)==protocol['fixed_gain_sha256']
    sources={str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('export_canonical_phase.py'),Path(__file__).with_name('static_four.py'),Path(__file__).with_name('hardware_aware.py'),Path(__file__).with_name('train_sensor_neutral.py'),Path(__file__).with_name('sensor_neutral.py'),Path(__file__).with_name('diagnose_frontend_transfer.py'),Path('LightGenV2/tasks/t21_continual_learning_hardware/model.py')]}
    a.out.mkdir(parents=True,exist_ok=False);started=time.monotonic();target=a.out/'model.pt'
    torch.save(dict(cp,model=state,canonical_export=True,original_checkpoint_SHA256=original_sha,export_rule='Preserve CPU physical 8bit phase codes and all nonphase tensors; no training or selection'),target);export_sha=sha(target)
    put(a.out/'status.json',dict(state='GPU_replay',updated=time.time()))
    gpu=gpu_gate(dict(allowed_GPU_UUIDs=['GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d'],per_process_GPU_memory_fraction=.2),a.gpu_uuid);device=torch.device('cuda:0');model=model.to(device)
    for name,param in model.named_parameters():
        if 'phase' in name:assert torch.equal(phase_codes(param).cpu(),phase_codes(cp['model'][name]))
    data=load_arrays(a.data,'test');front=checked_features(data,a.frontend,protocol['frontend_sha256']);cuda=infer(model,data,front,device);put(a.out/'GPU_report.json',cuda)
    del model;torch.cuda.empty_cache();put(a.out/'status.json',dict(state='CPU_full_replay',updated=time.time()))
    model=build_static_four(cp['architecture']);model.load_state_dict(torch.load(target,map_location='cpu',weights_only=False)['model'],strict=True);model.eval();cpu=infer(model,data,front,torch.device('cpu'));put(a.out/'CPU_report.json',cpu)
    original_comparison=compare(audit['full_result'],cpu,0.);parity=compare(cpu,cuda,.002)
    assert sha(path)==original_sha and sha(target)==export_sha and all(sha(n)==value for n,value in sources.items())
    manifest=dict(passed=original_comparison['passed'] and parity['passed'],contract=CONTRACT,source_run=str(a.run),source_checkpoint_SHA256=original_sha,export_checkpoint_SHA256=export_sha,frontend_SHA256=protocol['frontend_sha256'],geometry=geometry(model),physical_phase_codes_preserved=True,nonphase_tensors_bitwise_preserved=True,raw_phase_representation_changed_only=True,original_CPU_comparison=original_comparison,GPU_against_CPU=parity,scores=cpu['scores'],phase_proof=proof,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256=sources,gpu=gpu,CPU_threads=2,training=False,checkpoint_reselection=False,TEST_gradients=False,probability_editing=False,SDK=False,seconds=time.monotonic()-started)
    put(a.out/'manifest.json',manifest);put(a.out/'status.json',dict(state='complete' if manifest['passed'] else 'numerical_gate_failed',updated=time.time()))
    if not manifest['passed']:raise RuntimeError('Equivalent export did not meet unchanged numerical gate')


if __name__=='__main__':main()
