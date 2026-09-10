"""Fixed-checkpoint ablation: replace only the final global phase by noise."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import transformers

from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.datasets import prepare_lsp
from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_router import training as base
from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_router.protocol import build_periodic_test_protocol, persist_protocol
from .modeling import architecture_label, build_student, load_vision_backbone, sha256_file
from .refine import checked_fusion
from .run import PROFILES, _seed
from .settings import load_settings, save_resolved_config

TASK = Path(__file__).resolve().parent
GLOBAL_KEY = 'hybrid.optical_branch.core.global_phase.phase.raw_phase'


def uniform_phase_raw(shape, seed):
    u = torch.rand(shape, generator=torch.Generator().manual_seed(seed)).clamp(1e-6, 1-1e-6)
    return torch.logit(u)


def run(args):
    out = args.run_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        (out/name).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str)+'\n', encoding='utf-8')
    write('status.json', {'status': 'waiting' if args.wait_for_report else 'initializing', 'command': sys.argv})
    model = None
    try:
        if args.wait_for_report:
            deadline = time.monotonic()+args.wait_hours*3600
            while not args.wait_for_report.is_file():
                if time.monotonic()>deadline: raise TimeoutError('Training report not ready within waiting limit')
                time.sleep(30)
        blob = args.checkpoint.read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        payload = torch.load(io.BytesIO(blob), map_location='cpu', weights_only=False)
        if args.wait_for_report:
            final = json.loads(args.wait_for_report.read_text())
            if final['checkpoint_sha256'] != digest: raise RuntimeError('Training final report/checkpoint SHA mismatch')
        (out/'best_checkpoint.pt').write_bytes(blob)
        settings = load_settings(TASK/'configs'/PROFILES[args.profile])
        if payload['checkpoint_architecture'] != architecture_label(settings): raise RuntimeError('Wrong architecture/alpha profile')
        if payload['router_contract_sha256'] != settings.router_contract_sha256: raise RuntimeError('Wrong router contract')
        if payload['weight_variant'] != 'ema': raise RuntimeError('Expected selected EMA checkpoint')
        settings.data_root = args.data_root.resolve()
        settings.cache_dir = args.cache_dir.resolve()
        settings.local_files_only = True
        settings.download = False
        settings.output_dir = out
        settings.inference_batch_size = 24
        settings.num_workers = 4
        settings.visualization_sample_count = 0
        # Same metric/loss reporting as the earlier low-alpha evaluation.
        settings.coordinate_loss_weight = .1
        _seed(42)
        bundle = build_periodic_test_protocol(prepare_lsp(settings, persist=False))
        persist_protocol(bundle, out)
        if args.wait_for_report:
            import os
            device_id = os.environ['CUDA_VISIBLE_DEVICES']
            while True:
                used = int(subprocess.check_output(['nvidia-smi','-i',device_id,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
                if used < 200: break
                if time.monotonic()>deadline: raise TimeoutError('Selected GPU remained occupied')
                time.sleep(30)
        device = torch.device('cuda:0')
        loaded = load_vision_backbone(settings, device)
        model = build_student(loaded, settings)
        model.core.load_state_dict(payload['core'], strict=True)
        model.head.load_state_dict(payload['head'], strict=True)
        model.requires_grad_(False).eval()
        if settings.phase_parameterization != 'sigmoid': raise RuntimeError('Noise inverse mapping requires sigmoid phase')
        global_raw = dict(model.core.named_parameters())[GLOBAL_KEY]
        if tuple(global_raw.shape) != (478,478): raise RuntimeError('Expected existing 478x478 active global phase')
        original = global_raw.detach().clone()
        manifest = {'source_checkpoint': str(args.checkpoint), 'checkpoint_sha256': digest,
                    'source_epoch': payload['epoch'], 'command': sys.argv,
                    'git_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=TASK,text=True).strip(),
                    'profile': args.profile, 'fusion': checked_fusion(model,settings),
                    'intervention': 'replace only global raw phase; physical phase iid uniform [0,2pi), fixed per seed for every image',
                    'seeds': args.seeds, 'training': False, 'samples_per_condition': len(bundle.test),
                    'data_manifest_sha256': sha256_file(out/'pose_protocol_split.csv'),
                    'environment': {'torch':torch.__version__,'transformers':transformers.__version__,'gpu':torch.cuda.get_device_name()}}
        write('run_manifest.json',manifest)
        save_resolved_config(settings)
        loader = base._loader(bundle.test,settings,training=False)
        def evaluate(label):
            write('status.json',{'status':'evaluating','condition':label})
            metrics,_ = base.evaluate_model(model,'student',loader,loaded.processor,device,settings,
                       phase=label,epoch=payload['epoch'],save_outputs=True,tta=False)
            print('RESULT',label,metrics['pck_at_0.2_torso'],flush=True)
            return metrics
        baseline = evaluate('trained_global')
        rows=[]
        for seed in args.seeds:
            with torch.no_grad(): global_raw.copy_(uniform_phase_raw(global_raw.shape,seed).to(device))
            physical=2*math.pi*global_raw.detach().cpu().sigmoid()
            mask_file=out/f'global_uniform_seed{seed}.npy'
            np.save(mask_file,physical.numpy())
            metrics=evaluate(f'global_uniform_seed{seed}')
            rows.append({'seed':seed,'phase_sha256':sha256_file(mask_file),'metrics':metrics})
        with torch.no_grad(): global_raw.copy_(original)
        for key,value in model.core.state_dict().items():
            if not torch.equal(value.cpu(),payload['core'][key].cpu()): raise RuntimeError(f'Unexpected core state change: {key}')
        for key,value in model.head.state_dict().items():
            if not torch.equal(value.cpu(),payload['head'][key].cpu()): raise RuntimeError(f'Unexpected head state change: {key}')
        keys=['pck_at_0.2_torso','pckh_at_0.5_head','normalized_mean_error_torso']
        summary={k:{'mean':float(np.mean([r['metrics'][k] for r in rows])),
                    'std_population':float(np.std([r['metrics'][k] for r in rows])),
                    'min':min(r['metrics'][k] for r in rows),'max':max(r['metrics'][k] for r in rows)} for k in keys}
        write('final_report.json',{'baseline':baseline,'noise_runs':rows,'noise_summary':summary,
              'pck_drop_percentage_points':100*(baseline['pck_at_0.2_torso']-summary[keys[0]]['mean']),
              'fusion':manifest['fusion'],'source_checkpoint_sha256':digest,
              'only_global_phase_changed':True,'all_state_restored_exactly':True,'retraining':False})
        write('status.json',{'status':'complete'})
    except Exception as error:
        write('status.json',{'status':'failed','error':repr(error)})
        raise
    finally:
        if model is not None:model.restore_native()


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',choices=['main_dc20_no_shift_warmstart','alpha50'],required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--cache-dir',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44,45,46])
    p.add_argument('--wait-for-report',type=Path)
    p.add_argument('--wait-hours',type=float,default=12)
    run(p.parse_args())
