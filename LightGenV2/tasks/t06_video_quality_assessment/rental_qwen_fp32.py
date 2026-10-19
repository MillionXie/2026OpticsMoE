"""Optional frozen FP32 Qwen/visual fallback on the already rented card, never Qwen B1.

Run only after the two rental models finish and when the original-platform
CPU admission expired without measurements. No new hardware, rental or training.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from . import rental_precision_recheck as shared
from . import lab_fp32_recheck as lab
from . import rental_qwen_device_control as original


def main():
    import torch
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=('qwen','clip','yolo'),default='qwen')
    p.add_argument('--stage',choices=('util','native-cost','paced-cost'),required=True)
    p.add_argument('--gpu-uuid',required=True);p.add_argument('--deadline',type=float,required=True)
    args=p.parse_args()
    if not 0<args.deadline-time.time()<=7200 or os.environ.get('CUDA_VISIBLE_DEVICES')!=args.gpu_uuid:
        raise ValueError('Finite already-authorized rental GPU window')
    args.baseline=args.model;args.precision='fp32';args.batch=3;args.power_limit_w=400
    if args.model=='clip':args.batch=10
    args.output=shared.ROOT/f'{args.model}_fp32_{args.stage}'
    args.model_path=original.MODEL_DIR if args.model=='qwen' else Path('/data/models/openai-clip/ViT-B-32.pt' if args.model=='clip' else '/data/models/yolo11s.pt')
    previous=lab.TASK/'runs/simulation/lgvq_temporal_reproduction_a100_20261013'/args.model
    args.checkpoint=original.OUT/'original_best_checkpoint.pt' if args.model=='qwen' else previous/'checkpoints/temporal/best_checkpoint.pt'
    args.manifest=Path('/data/lgvq/original_manifest.csv');args.reproduction_receipt=None
    args.remapped_test_rows=[r for r in shared.old.rows() if r['split']=='test']
    if not shared.REPO.is_relative_to('/data') or args.output.exists():raise RuntimeError('New canonical data only')
    if args.model=='qwen':
        receipt=json.loads((args.model_path/'asset_receipt.json').read_text())
        if receipt['revision']!=original.REVISION or shared.old.sha(args.checkpoint)!=original.HEAD_SHA:
            raise RuntimeError('Original official revision and frozen head required')
        for name,digest in receipt['hashes'].items():
            if shared.old.sha(args.model_path/name)!=digest:raise RuntimeError('Original official Qwen asset changed')
        if shared.old.sha(args.model_path/'model.safetensors')!=original.WEIGHT_SHA:
            raise RuntimeError('Original official Qwen weight SHA required')
    else:
        args.reproduction_receipt=previous/'reproduction_receipt.json'
        lab.frozen.validate_reproduction_receipt(args.reproduction_receipt,args.model,args.checkpoint)
        if shared.old.sha(args.model_path)!=lab.frozen.BACKBONE_SHA[args.model]:raise RuntimeError('Original official visual asset SHA required')
        if args.model=='clip':
            import clip,inspect
            expected={'__init__.py':'d7930eb290ebed432116795c91473200d098451a94c0aedd4ae0f934f97955e8',
                      'model.py':'429489552b2610903b82c9c3215cbb50ca8552e60f0b17dfc4233c1ea0f8422c',
                      'clip.py':'c28fdadfc140ce89c525cd6c29c0dc71c462e8fa45315e6524f080a4dafcdf75'}
            for name,digest in expected.items():
                if shared.old.sha(Path(clip.__file__).parent/name)!=digest:raise RuntimeError('Original OpenAI CLIP implementation changed')
        else:
            import ultralytics
            if ultralytics.__version__!='8.4.159':raise RuntimeError('Original Ultralytics version required')
    args.utilization_source=str(shared.ROOT/f'{args.model}_fp32_util'/('paced' if args.stage=='paced-cost' else 'native')/'report.json')
    args.output.mkdir();guard=shared.Guard(args)
    identity=dict(model=args.model,batch=args.batch,precision='fp32',gpu_uuid=args.gpu_uuid,
        sources={**shared.sources(),str(Path(lab.__file__).relative_to(shared.REPO)):shared.old.sha(Path(lab.__file__)),
                 str(Path(__file__).relative_to(shared.REPO)):shared.old.sha(Path(__file__))},
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=shared.REPO,text=True).strip(),
        fallback_platform_note='Original40GB admission unavailable; new FP32-only80GB execution, not the excluded rental nativeB1 comparison')
    try:
        guard.check();torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest')
        model,rows,prepare,transfer,compute,first,assets=lab.runtime(args,guard,identity)
        shared.old.write(args.output/'identity.json',identity)
        if args.stage=='util':
            cpu,_=prepare(rows[:args.batch]);inputs=transfer(cpu);reference,_=compute(inputs);reference=reference.cpu()
            for paced in (False,True):
                report=shared.measure_util(args,compute,inputs,args.output/('paced' if paced else 'native'),guard,paced)
                assert all(torch.allclose(torch.tensor(r['output']),reference,atol=.0002,rtol=0) for r in report['repeats'])
        else:
            util=json.loads(Path(args.utilization_source).read_text())
            if util['precision']!='fp32' or util['model']!=args.model or util['batch']!=args.batch:raise RuntimeError('Own independent util identity')
            identity['independent_utilization']=dict(path=args.utilization_source,sha256=shared.old.sha(Path(args.utilization_source)),value=util['mean_gpu_busy_percent'])
            duty=util['frozen_duty'] if args.stage=='paced-cost' else 1
            shared.evaluate(args,compute,prepare,transfer,first,args.output/'evaluation',guard,duty,identity,rows)
        if lab.costs.model_hashes(args.model_path)!=assets:raise RuntimeError('Official Qwen assets changed')
        guard.check();shared.old.write(args.output/'status.json',dict(status='complete',time=time.time()))
    except BaseException as error:
        shared.old.write(args.output/'status.json',dict(status='failed',error=repr(error),time=time.time()));raise
    finally:shared.old.write(args.output/'identity.json',identity);guard.close(args.output/'process_audit.json')


if __name__=='__main__':main()
