"""Fixed-last, single-seed30 training with explicit backbone/freeze profiles."""
import argparse
import hashlib
import itertools
import math
import os
from pathlib import Path
import platform
import random
import signal
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .models import build_model, parameter_counts
from .task_problems import Problem
from .train_easy100 import TASK, ROOT, PROFILE, wait_for_gpu, git, require_published_source, write_json, save_checkpoint


def run(args):
    publication_began=time.monotonic()
    while True:
        try:
            require_published_source()
            break
        except RuntimeError as exc:
            if time.monotonic()-publication_began>600:raise
            print('Waiting for clean, published source before data/CUDA:',str(exc),flush=True)
            time.sleep(5)
    if args.seed != 42 or args.epochs != (1 if args.smoke_batches else 30):
        raise ValueError('Requires seed42 and30epochs (one epoch for smoke)')
    run_dir = args.run_dir.resolve()
    category = 'smoke' if args.smoke_batches else 'simulation'
    if not run_dir.is_relative_to((TASK / 'runs' / category).resolve()):
        raise ValueError('Run path outside task')
    torch.set_num_threads(4)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    problem = Problem(args.task)
    if args.preflight:
        print({'task': args.task, 'identity': problem.identity, 'train': len(problem.train), 'test': len(problem.test)})
        return
    # Original data checks can outlast a GPU idle snapshot. Wait without a
    # CUDA context when another user arrives before actual admission.
    wait_for_gpu(args)
    run_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device('cuda:0')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model = build_model(args.task, backbone=args.backbone).to(device)
    problem.configure_model(model, device)
    train_data = problem.train
    if args.smoke_batches and problem.batch_sampler is None:
        train_data = Subset(train_data, range(min(len(train_data), args.smoke_batches*problem.batch_size)))
    options = dict(batch_size=problem.batch_size, num_workers=2, pin_memory=True,
                   multiprocessing_context='spawn', persistent_workers=True)
    if problem.batch_sampler is None:
        train = DataLoader(train_data, shuffle=True, generator=torch.Generator().manual_seed(42), **options)
    else:
        sampler = (list(itertools.islice(iter(problem.batch_sampler), args.smoke_batches))
                   if args.smoke_batches else problem.batch_sampler)
        train = DataLoader(train_data,batch_sampler=sampler,**{k:v for k,v in options.items() if k!='batch_size'})
    test = DataLoader(problem.test, shuffle=False, **options)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=.001, weight_decay=.0001)
    frozen = args.backbone == 'resnet18_imagenet_frozen'
    frozen_hash = None
    if frozen:
        from .frozen_resnet18 import encoder_state_sha256, assert_encoder_unchanged
        frozen_hash = encoder_state_sha256(model.encoder)
        assert_encoder_unchanged(model.encoder, frozen_hash)
        encoder_ids = {id(p) for p in model.encoder.parameters()}
        if any(id(p) in encoder_ids for p in trainable):
            raise RuntimeError('Frozen encoder entered the optimizer')
    manifest = {'task': args.task, 'backbone': args.backbone, 'status': 'running', 'seed': 42, 'epochs': args.epochs,
        'learning_rate': .001, 'schedule': 'constant', 'warmup_epochs': 0, 'effective_batch': 32,
        'microbatch': problem.batch_size, 'precision': 'FP32_TF32_disabled', 'optimizer': 'AdamW_wd0.0001_clip1',
        'git_commit': git('rev-parse','HEAD'), 'data_identity': problem.identity,
        'selection': 'fixed last epoch; no metric-based selection', 'parameter_counts': parameter_counts(model),
        'pid': os.getpid(), 'gpu_uuid': args.gpu_uuid, 'gpu': torch.cuda.get_device_name(),
        'command': [sys.executable,'-m',__spec__.name,*sys.argv[1:]],
        'python': platform.python_version(), 'torch': torch.__version__, 'started_unix': time.time(),
        'source_sha256': {name: hashlib.sha256((TASK/name).read_bytes()).hexdigest()
                          for name in ('train_tasks.py','task_problems.py','models.py','lgvq_data.py','other_data.py','editor_dataset.py','train_easy100.py')},
        'smoke_batches': args.smoke_batches, 'metric': problem.metric}
    manifest.update(initialization='official_ImageNet1K_V1' if frozen else 'random_no_pretraining',
                    training_scope='decoder_only' if frozen else 'full_network',
                    optimizer_parameters=sum(p.numel() for p in trainable))
    if frozen:
        manifest.update(pretrained_provenance=model.encoder.pretrained_provenance,
                        encoder_state_sha256_before=frozen_hash, frozen_encoder_audits=[],
                        source_sha256={**manifest['source_sha256'], **{
                            name:hashlib.sha256((TASK/name).read_bytes()).hexdigest()
                            for name in ('frozen_resnet18.py', 'frozen_visual_models.py')}})
    write_json(run_dir/'run_manifest.json', manifest)
    history, selected_epoch = [], None
    began=time.monotonic()
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM, interrupt)
    try:
        for epoch in range(1,args.epochs+1):
            model.train()
            if hasattr(train.batch_sampler, 'set_epoch'):
                train.batch_sampler.set_epoch(epoch)
            optimizer.zero_grad(set_to_none=True)
            total, count = 0., 0
            accumulation = max(1,32//problem.batch_size)
            for step,batch in enumerate(train):
                n = len(batch['sample_id'])
                group_start = (step//accumulation)*accumulation*problem.batch_size
                group_samples = n if accumulation==1 else min(32,len(train_data)-group_start)
                loss=problem.loss(model,batch,device)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                (loss*n/group_samples).backward()
                if (step+1)%accumulation==0 or step+1==len(train):
                    torch.nn.utils.clip_grad_norm_(trainable,1.)
                    optimizer.step(); optimizer.zero_grad(set_to_none=True)
                total+=loss.item()*n; count+=n
                if (step+1)%25==0:
                    write_json(run_dir/'progress.json',{'status':'training','epoch':epoch,'step':step+1,
                        'train_loss':total/count,'elapsed_seconds':time.monotonic()-began})
            evaluated = epoch==1 or epoch%5==0 or epoch==args.epochs
            metrics, records = problem.evaluate(model,test,device) if evaluated else ({},None)
            entry={'epoch':epoch,'train_loss':total/count,'learning_rate':.001,
                   'elapsed_seconds':time.monotonic()-began,**metrics}
            if frozen:
                audit=assert_encoder_unchanged(model.encoder,frozen_hash)
                manifest['frozen_encoder_audits'].append({'epoch':epoch,**audit})
                write_json(run_dir/'run_manifest.json',manifest)
            payload={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'manifest':manifest,'metrics':entry}
            save_checkpoint(run_dir/'last_checkpoint.pt',payload)
            if epoch == args.epochs:
                selected_epoch=epoch
                save_checkpoint(run_dir/'best_checkpoint.pt',payload)
                write_json(run_dir/'best_predictions.json',records if records is not None else
                           {'status':'selected_by_TRAIN_loss_not_evaluated_this_epoch','selected_epoch':epoch})
            history.append(entry)
            write_json(run_dir/'history.json',history)
            write_json(run_dir/'progress.json',{'status':'training',**entry,'selected_epoch':selected_epoch})
            print(entry,flush=True)
        selected=torch.load(run_dir/'last_checkpoint.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(selected['model'])
        if frozen:
            manifest['frozen_encoder_final_audit']=assert_encoder_unchanged(model.encoder,frozen_hash)
        metrics,records=problem.evaluate(model,test,device)
        selection_metrics=metrics
        final_loader=None
        if problem.final_test is not None:
            final_loader=DataLoader(problem.final_test,shuffle=False,**options)
            metrics,records=problem.evaluate(model,final_loader,device)
        write_json(run_dir/'fixed_weight_predictions.json',records)
        write_json(run_dir/'result.json',{'status':'complete','selected_epoch':selected_epoch,'metrics':metrics,
            'selection_metrics':selection_metrics,'selection_uses_test':False,
            'selection':'fixed_last_epoch','checkpoint':'last_checkpoint.pt',
            'checkpoint_sha256':hashlib.sha256((run_dir/'last_checkpoint.pt').read_bytes()).hexdigest(),
            'training_scope':manifest['training_scope'],
            'frozen_encoder_audit':manifest.get('frozen_encoder_final_audit'),
            'development_metric':problem.task!='abo_conditional_edit','elapsed_seconds':time.monotonic()-began})
        manifest['status']='complete'
    except BaseException as exc:
        manifest.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=repr(exc))
        raise
    finally:
        manifest['finished_unix']=time.time()
        write_json(run_dir/'run_manifest.json',manifest)
        for loader in (train,test,locals().get('final_loader')):
            if loader is None:continue
            iterator=getattr(loader,'_iterator',None)
            if iterator is not None: iterator._shutdown_workers()
        del model,optimizer
        torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',required=True)
    p.add_argument('--backbone',choices=['resnet20','resnet32','resnet18_imagenet_frozen'],default='resnet20')
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--gpu-uuid',required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--smoke-batches',type=int,default=0)
    p.add_argument('--preflight',action='store_true')
    p.add_argument('--queue-status-path',type=Path)
    p.add_argument('--shared-owner-pids',type=int,nargs='*',default=[])
    run(p.parse_args())


if __name__=='__main__': main()
