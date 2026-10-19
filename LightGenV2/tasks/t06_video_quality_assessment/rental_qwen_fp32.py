"""Optional frozen FP32 Qwen3 fallback on the already rented card, never B1.

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
    p.add_argument('--stage',choices=('util','native-cost','paced-cost'),required=True)
    p.add_argument('--gpu-uuid',required=True);p.add_argument('--deadline',type=float,required=True)
    args=p.parse_args()
    if not 0<args.deadline-time.time()<=7200 or os.environ.get('CUDA_VISIBLE_DEVICES')!=args.gpu_uuid:
        raise ValueError('Finite already-authorized rental GPU window')
    args.model='qwen';args.baseline='qwen';args.precision='fp32';args.batch=3;args.power_limit_w=400
    args.output=shared.ROOT/f'qwen_fp32_{args.stage}'
    args.model_path=original.MODEL_DIR
    args.checkpoint=original.OUT/'original_best_checkpoint.pt'
    args.manifest=Path('/data/lgvq/original_manifest.csv');args.reproduction_receipt=None
    args.remapped_test_rows=[r for r in shared.old.rows() if r['split']=='test']
    if not shared.REPO.is_relative_to('/data') or args.output.exists():raise RuntimeError('New canonical data only')
    receipt=json.loads((args.model_path/'asset_receipt.json').read_text())
    if receipt['revision']!=original.REVISION or shared.old.sha(args.checkpoint)!=original.HEAD_SHA:
        raise RuntimeError('Original official revision and frozen head required')
    for name,digest in receipt['hashes'].items():
        if shared.old.sha(args.model_path/name)!=digest:raise RuntimeError('Original official Qwen asset changed')
    if shared.old.sha(args.model_path/'model.safetensors')!=original.WEIGHT_SHA:
        raise RuntimeError('Original official Qwen weight SHA required')
    args.utilization_source=str(shared.ROOT/'qwen_fp32_util'/('paced' if args.stage=='paced-cost' else 'native')/'report.json')
    args.output.mkdir();guard=shared.Guard(args)
    identity=dict(model='qwen',batch=3,precision='fp32',gpu_uuid=args.gpu_uuid,
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
            cpu,_=prepare(rows[:3]);inputs=transfer(cpu);reference,_=compute(inputs);reference=reference.cpu()
            for paced in (False,True):
                report=shared.measure_util(args,compute,inputs,args.output/('paced' if paced else 'native'),guard,paced)
                assert all(torch.allclose(torch.tensor(r['output']),reference,atol=.0002,rtol=0) for r in report['repeats'])
        else:
            util=json.loads(Path(args.utilization_source).read_text())
            if util['precision']!='fp32' or util['model']!='qwen' or util['batch']!=3:raise RuntimeError('Own independent util identity')
            identity['independent_utilization']=dict(path=args.utilization_source,sha256=shared.old.sha(Path(args.utilization_source)),value=util['mean_gpu_busy_percent'])
            duty=util['frozen_duty'] if args.stage=='paced-cost' else 1
            shared.evaluate(args,compute,prepare,transfer,first,args.output/'evaluation',guard,duty,identity,rows)
        if lab.costs.model_hashes(args.model_path)!=assets:raise RuntimeError('Official Qwen assets changed')
        guard.check();shared.old.write(args.output/'status.json',dict(status='complete',time=time.time()))
    except BaseException as error:
        shared.old.write(args.output/'status.json',dict(status='failed',error=repr(error),time=time.time()));raise
    finally:shared.old.write(args.output/'identity.json',identity);guard.close(args.output/'process_audit.json')


if __name__=='__main__':main()
