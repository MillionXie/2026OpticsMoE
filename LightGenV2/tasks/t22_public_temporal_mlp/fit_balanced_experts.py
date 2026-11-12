"""Fit linear experts on TRAIN to preserve a teacher under a balanced router."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import torch

if __package__:
    from .data import batch, groups, load_inputs, sha256
    from .model import load_lightgen
    from .run import ROOT, ensure_idle_gpu, evaluate, save_checkpoint, write
else:
    from data import batch, groups, load_inputs, sha256
    from model import load_lightgen
    from run import ROOT, ensure_idle_gpu, evaluate, save_checkpoint, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher', type=Path, required=True)
    parser.add_argument('--router', type=Path, required=True)
    parser.add_argument('--paths', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--ridge', type=float, default=1e-4)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--allow-shared-gpu', action='store_true')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.ridge <= 0:
        raise ValueError('A positive ridge coefficient is required')
    if args.device.startswith('cuda') and not args.allow_shared_gpu:
        ensure_idle_gpu()
    torch.set_num_threads(4)
    teacher, teacher_metadata = load_lightgen(args.teacher, args.device)
    model, _ = load_lightgen(args.teacher, args.device)
    router_source, _ = load_lightgen(args.router, 'cpu')
    if teacher.cfg != router_source.cfg or teacher.cfg.expert_mode != 'linear':
        raise ValueError('Matching linear-expert architectures are required')
    model.frame_router.load_state_dict(router_source.frame_router.state_dict())
    model.eval()
    payload = load_inputs(json.loads(args.paths.read_text('utf-8')))
    if payload['identity']['manifest']['sha256'] != teacher_metadata['dataset']['manifest']['sha256']:
        raise ValueError('TRAIN manifest differs from teacher')
    indices, valid = groups(payload, 'train')
    width = model.cfg.width
    gram = torch.zeros(4*width, 4*width, device=args.device, dtype=torch.float64)
    cross = torch.zeros(4*width, width, device=args.device, dtype=torch.float64)
    target_energy = 0.
    rows = 0
    started = time.time()
    with torch.inference_mode():
        for start in range(0, len(indices), 4):
            item = batch(payload, indices[start:start+4], valid[start:start+4], args.device)
            tokens = teacher.visual_input_norm(teacher.vision_adapter(item['vision']))
            tokens = tokens[item['valid']]
            target_routes = teacher.frame_router(tokens)
            target = teacher.frame_experts(tokens, target_routes['weights']).reshape(-1, width)
            weights = model.frame_router(tokens)['weights']
            design = (tokens[..., None, :]*weights[..., None, :, None]).flatten(-2)
            design = design.reshape(-1, 4*width)
            gram += (design.T @ design).double()
            cross += (design.T @ target).double()
            target_energy += float(target.double().square().sum())
            rows += len(design)
    if rows != 2250*4*49:
        raise ValueError('TRAIN or padding population differs')
    gram = gram.cpu()
    cross = cross.cpu()
    prior = torch.cat([expert.weight.detach().cpu().T for expert in teacher.frame_experts.experts]).double()
    strength = args.ridge*gram.trace()/len(gram)
    solution = torch.linalg.solve(gram+strength*torch.eye(len(gram),dtype=gram.dtype),cross+strength*prior)
    residual_energy = target_energy-2*float((solution*cross).sum())+float((solution*(gram@solution)).sum())
    with torch.no_grad():
        for index, expert in enumerate(model.frame_experts.experts):
            expert.weight.copy_(solution[index*width:(index+1)*width].T.to(expert.weight))
    args.out.mkdir(parents=True)
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    metadata = dict(teacher_metadata)
    metadata.pop('metrics',None)
    metadata.pop('epoch',None)
    metadata.update(
        role='PUBLIC_SURROGATE_DEVELOPMENT',
        method='TRAIN-only ridge distillation of linear experts with a balanced visual router',
        source_commit=os.environ.get('LIGHTGEN_SOURCE_COMMIT',commit),
        source_SHA256={path.name:sha256(path) for path in ROOT.glob('*.py')},
        initialization={'teacher_sha256':sha256(args.teacher),'router_sha256':sha256(args.router)},
        training_rows=rows,
        training_videos=2250,
        test_used_for_gradient=False,
        test_used_for_selection=True,
        ridge_coefficient=args.ridge,
        ridge_diagonal_strength=float(strength),
        TRAIN_relative_mixture_MSE=max(residual_energy,0.)/target_energy,
        python=platform.python_version(),
        torch=str(torch.__version__),
        device=args.device,
        command=list(__import__('sys').argv),
        gpu_name=torch.cuda.get_device_name(0) if args.device.startswith('cuda') else None,
        nonexclusive_GPU=args.allow_shared_gpu,
    )
    metrics = evaluate(model,payload,args.device,4,args.out)
    save_checkpoint(args.out/'LightGen.pt',model.checkpoint(**metadata,metrics=metrics))
    write(args.out/'metadata.json',metadata)
    write(args.out/'status.json',{'state':'completed','elapsed_seconds':time.time()-started,**metrics})
    print(json.dumps({'TRAIN_relative_mixture_MSE':metadata['TRAIN_relative_mixture_MSE'],
                      'srcc':metrics['srcc'],'routing':metrics['frame_routing']}),flush=True)


if __name__ == '__main__':
    main()
