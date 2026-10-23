"""Re-evaluate completed last checkpoints on CPU; watch existing runs without restarting them."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from .models import build_model
from .task_problems import Problem
from .train_easy100 import TASK, ROOT, git, require_published_source, write_json


def check_last_epoch(checkpoint, manifest):
    if manifest['status'] != 'complete' or manifest['epochs'] != 30:
        raise ValueError('Only a completed original 30-epoch run is eligible')
    if checkpoint['epoch'] != manifest['epochs']:
        raise ValueError('Last checkpoint is not the final authorized epoch')
    if checkpoint['manifest']['task'] != manifest['task']:
        raise ValueError('Checkpoint task identity mismatch')


def evaluate_one(run_dir):
    manifest = json.loads((run_dir/'run_manifest.json').read_text())
    checkpoint_path = run_dir/'last_checkpoint.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    check_last_epoch(checkpoint, manifest)
    output = run_dir/'last_epoch_review_20261023'
    if (output/'result.json').exists():
        return json.loads((output/'result.json').read_text())
    output.mkdir(exist_ok=True)
    write_json(output/'status.json', {'status':'evaluating', 'pid':os.getpid(), 'epoch':30})
    model = (build_model(manifest['task'], backbone=manifest['backbone'])
             if 'backbone' in manifest else build_model(manifest['task'])).eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    task = manifest['task']
    if task in {'abo_image_to_text','abo_text_to_image'}:
        from .easy100_data import Easy100
        from .train_easy100 import evaluate
        command = manifest['command']
        dataset = Easy100(Path(command[command.index('--data-root')+1]))
        if dataset.identity != manifest['data_identity']:
            raise ValueError('Original data identity differs')
        loader = DataLoader(dataset.images('test'), batch_size=32, num_workers=0)
        records = evaluate(model,loader,dataset.title_tokens,dataset.title_skus,
                           task.removeprefix('abo_'),'cpu')
        metrics = {k:v for k,v in records.items() if isinstance(v,(int,float))}
    else:
        problem = Problem(task)
        if problem.identity != manifest['data_identity']:
            raise ValueError('Original data identity differs')
        # MOS scale is checkpoint state; never reconfigure or refit it here.
        data = problem.final_test if problem.final_test is not None else problem.test
        loader = DataLoader(data,batch_size=problem.batch_size,num_workers=0)
        metrics, records = problem.evaluate(model,loader,'cpu')
    digest = hashlib.sha256()
    with checkpoint_path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
    result = dict(status='complete',task=task,selected_epoch=30,selection='fixed_last_epoch',
        selection_uses_test=False,checkpoint='last_checkpoint.pt',checkpoint_sha256=digest.hexdigest(),
        metrics=metrics,original_training_commit=manifest['git_commit'],evaluation_commit=git('rev-parse','HEAD'),
        original_selection=manifest['selection'],historical_test_monitoring=True,
        note='User changed reporting rule after observing historical test curves; no epoch selected by test metrics.',
        device='cpu',precision='FP32_TF32_disabled',finished_unix=time.time())
    write_json(output/'fixed_weight_predictions.json', records)
    write_json(output/'result.json',result)
    write_json(output/'status.json', {'status':'complete','pid':os.getpid(),'epoch':30})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suites',nargs='+',required=True)
    parser.add_argument('--watch',action='store_true')
    args=parser.parse_args()
    require_published_source()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    summary_path=TASK/'runs/control/last_epoch_review_20261023.json'
    results,errors={},{}
    while True:
        pending=[]
        for suite_name in args.suites:
            if Path(suite_name).name != suite_name:raise ValueError('Invalid suite')
            queue=json.loads((TASK/'runs/simulation'/suite_name/'queue_status.json').read_text())
            for job in queue['jobs']:
                if job.get('phase','formal')!='formal' or job['status'] in {'failed','blocked_by_failed_smoke'}:continue
                name=job['name']
                if name in results or name in errors:continue
                if job['status']!='complete':pending.append(name);continue
                try:
                    print('Evaluating final epoch:',name,flush=True)
                    results[name]=evaluate_one(TASK/'runs/simulation'/name)
                    print(name,results[name]['metrics'],flush=True)
                except Exception as exc:
                    errors[name]=repr(exc)
                    print('Evaluation failed:',name,repr(exc),flush=True)
                write_json(summary_path,dict(status='running',pid=os.getpid(),results=results,errors=errors,pending=pending))
        status='complete' if not pending and not errors else ('failed' if errors else 'waiting_for_training')
        write_json(summary_path,dict(status=status,pid=os.getpid(),results=results,errors=errors,pending=pending))
        if not pending or not args.watch:break
        time.sleep(30)


if __name__=='__main__':main()
