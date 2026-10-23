"""Fixed-last, single-seed30 training with explicit backbone/freeze profiles."""
import argparse
import hashlib
import itertools
import io
import json
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

EDITOR_COUNTS = {'train':20736,'val':2304,'test':2304}


class CachedIndexBatches:
    """Read deterministic pairs in the retained original sampler's order."""
    batch_sampler = None

    def __init__(self, dataset, indices, limit=0):
        self.dataset, self.indices, self.limit = dataset, indices, limit

    def __len__(self):
        return min(len(self.indices), self.limit) if self.limit else len(self.indices)

    def __iter__(self):
        from torch.utils.data import default_collate
        batches = iter(self.indices)
        if self.limit:
            batches = itertools.islice(batches, self.limit)
        for indices in batches:
            if hasattr(self.dataset,'get_batch'):
                yield self.dataset.get_batch(indices)
            else:
                yield default_collate([self.dataset[int(index)] for index in indices])


def load_editor_parent(path, problem, model, optimizer):
    """Read one atomic checkpoint version; no metric-based selection."""
    path = path.resolve(strict=True)
    if path.name != 'last_checkpoint.pt' or not path.is_relative_to((TASK/'runs/simulation').resolve()):
        raise ValueError('Editor resume requires a task-local simulation last checkpoint')
    content = path.read_bytes()
    parent = torch.load(io.BytesIO(content), map_location='cpu', weights_only=False)
    manifest = parent['manifest']
    expected = dict(task='abo_conditional_edit', backbone='resnet18_imagenet_frozen', seed=42,
                    epochs=30, learning_rate=.001, effective_batch=32, microbatch=8,
                    text_encoder='byte_gru', training_scope='frozen_visual_trainable_text_and_decoder')
    if any(manifest.get(key) != value for key,value in expected.items()):
        raise ValueError('Editor parent training protocol differs')
    if manifest['data_identity'] != problem.identity or not 1 <= parent['epoch'] < 30:
        raise ValueError('Editor parent data or continuation epoch differs')
    model.load_state_dict(parent['model'], strict=True)
    from .frozen_resnet18 import assert_encoder_unchanged
    from .continue_lgvq50 import structured_sha256
    assert_encoder_unchanged(model.encoder,manifest['encoder_state_sha256_before'])
    expected_optimizer_sha = structured_sha256(parent['optimizer'])
    optimizer.load_state_dict(parent['optimizer'])
    if structured_sha256(optimizer.state_dict()) != expected_optimizer_sha:
        raise ValueError('Editor AdamW moments or parameter groups did not restore exactly')
    if any(group['lr'] != .001 or group['weight_decay'] != .0001 for group in optimizer.param_groups):
        raise ValueError('Editor optimizer does not use the fixed approved settings')
    expected_steps = math.ceil(len(problem.train)/32)*parent['epoch']
    if (len(optimizer.state) != sum(len(group['params']) for group in optimizer.param_groups)
            or any(int(state['step']) != expected_steps for state in optimizer.state.values())):
        raise ValueError('Editor AdamW state does not match all completed parent epochs')
    return parent, {'source_run':str(path.parent), 'checkpoint':'last_checkpoint.pt',
                    'checkpoint_sha256':hashlib.sha256(content).hexdigest(),
                    'completed_epoch':parent['epoch'], 'model_and_AdamW_restored':True,
                    'optimizer_state_sha256':expected_optimizer_sha,'optimizer_steps':expected_steps,
                    'selection':'last_completed_epoch_for_engineering_resume'}


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
    small_text = getattr(args,'text_encoder','minilm') == 'byte_gru'
    if small_text and args.backbone != 'resnet18_imagenet_frozen':
        raise ValueError('Small text profile requires frozen pretrained ResNet18')
    model_options = {'backbone':args.backbone}
    if small_text:model_options['text_encoder']='byte_gru'
    model = build_model(args.task, **model_options).to(device)
    if small_text and parameter_counts(model)['total'] >= 15000000:
        raise ValueError('Complete model must have fewer than15000000 parameters')
    problem.configure_model(model, device)
    train_data = problem.train
    resume_path = getattr(args, 'resume_from', None)
    cache_root = getattr(args, 'editor_cache_root', None)
    if bool(resume_path) != bool(cache_root):
        raise ValueError('Cached editor continuation needs both parent last and cache root')
    if resume_path:
        if args.task != 'abo_conditional_edit' or not small_text:
            raise ValueError('Engineering resume is restricted to the deterministic small-text editor')
        from .performance_cache import CachedEditorPairs
        cache_root = cache_root.resolve(strict=True)
        if not cache_root.is_relative_to((TASK/'runs').resolve()):
            raise ValueError('Editor cache must remain inside task runs')
        datasets = {split:CachedEditorPairs(cache_root/split, source_identity=problem.identity,verify_integrity=False)
                    for split in ('train','val','test')}
        if any(len(ds)!=EDITOR_COUNTS[split] or ds.manifest.get('dataset_split')!=split
               for split,ds in datasets.items()):
            raise RuntimeError('Editor cache split/count binding differs')
        if any(not ds.manifest.get('all_rows_exactly_verified') for ds in datasets.values()):
            raise RuntimeError('Editor cache has not passed full exact input/target verification')
        for dataset in datasets.values():dataset.preload()
        problem.train,problem.test,problem.final_test = (datasets[key] for key in ('train','val','test'))
        train_data = problem.train
    if args.smoke_batches and problem.batch_sampler is None and not resume_path:
        train_data = Subset(train_data, range(min(len(train_data), args.smoke_batches*problem.batch_size)))
    options = dict(batch_size=problem.batch_size, num_workers=2, pin_memory=True,
                   multiprocessing_context='spawn', persistent_workers=True)
    if resume_path:
        train = None
    elif problem.batch_sampler is None:
        train = DataLoader(train_data, shuffle=True, generator=torch.Generator().manual_seed(42), **options)
    else:
        sampler = (list(itertools.islice(iter(problem.batch_sampler), args.smoke_batches))
                   if args.smoke_batches else problem.batch_sampler)
        train = DataLoader(train_data,batch_sampler=sampler,**{k:v for k,v in options.items() if k!='batch_size'})
    test_data = problem.test
    if resume_path and args.smoke_batches:
        problem.final_test = None  # Engineering smoke does not observe TEST predictions.
    if resume_path:
        test_count=min(16,len(test_data)) if args.smoke_batches else len(test_data)
        test=CachedIndexBatches(test_data,DataLoader(range(test_count),batch_size=8,shuffle=False,num_workers=0))
    else:
        test = DataLoader(test_data, shuffle=False, **options)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=.001, weight_decay=.0001)
    parent, continuation, index_loader = None, None, None
    start_epoch, stop_epoch = 1,args.epochs
    if resume_path:
        from .continue_lgvq50 import replay_index_loader
        parent,continuation = load_editor_parent(resume_path,problem,model,optimizer)
        start_epoch = parent['epoch']+1
        stop_epoch = start_epoch if args.smoke_batches else 30
        if any(isinstance(module,torch.nn.Dropout) and module.p for module in model.modules()):
            raise ValueError('Old checkpoint has no model RNG state; stochastic heads cannot resume this way')
        index_loader,replay_audit = replay_index_loader(len(train_data),8,parent['epoch'])
        continuation['sampler_replay'] = replay_audit
        train = CachedIndexBatches(train_data,index_loader,args.smoke_batches)
    frozen = args.backbone == 'resnet18_imagenet_frozen'
    frozen_hash = None
    text_frozen_hash = None
    if frozen:
        from .frozen_resnet18 import encoder_state_sha256, assert_encoder_unchanged
        frozen_hash = encoder_state_sha256(model.encoder)
        assert_encoder_unchanged(model.encoder, frozen_hash)
        encoder_ids = {id(p) for p in model.encoder.parameters()}
        if any(id(p) in encoder_ids for p in trainable):
            raise RuntimeError('Frozen encoder entered the optimizer')
        if hasattr(model,'text_encoder') and not small_text:
            text_frozen_hash=encoder_state_sha256(model.text_encoder)
            assert_encoder_unchanged(model.text_encoder,text_frozen_hash)
            text_ids={id(p) for p in model.text_encoder.parameters()}
            if any(id(p) in text_ids for p in trainable):
                raise RuntimeError('Frozen text encoder entered the optimizer')
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
    manifest['max_physical_gpus']=getattr(args,'max_physical_gpus',3)
    manifest['max_jobs_per_gpu']=getattr(args,'max_jobs_per_gpu',2)
    manifest.update(initialization='official_ImageNet1K_V1' if frozen else 'random_no_pretraining',
                    training_scope='decoder_only' if frozen else 'full_network',
                    optimizer_parameters=sum(p.numel() for p in trainable))
    if small_text:
        manifest.update(text_encoder='byte_gru',text_initialization='random_seed42',
                        training_scope='frozen_visual_trainable_text_and_decoder',
                        full_model_parameter_limit=15000000,
                        text_state_sha256_before=encoder_state_sha256(model.text_encoder))
    if frozen:
        manifest.update(pretrained_provenance=model.encoder.pretrained_provenance,
                        encoder_state_sha256_before=frozen_hash, frozen_encoder_audits=[],
                        source_sha256={**manifest['source_sha256'], **{
                            name:hashlib.sha256((TASK/name).read_bytes()).hexdigest()
                            for name in ('frozen_resnet18.py', 'frozen_visual_models.py')}})
        if small_text:
            manifest['source_sha256']['frozen_multimodal_models.py']=hashlib.sha256((TASK/'frozen_multimodal_models.py').read_bytes()).hexdigest()
        if text_frozen_hash:
            manifest['initialization']='official_pretrained_visual_and_text'
            manifest.update(text_pretrained_provenance=model.text_encoder.pretrained_provenance,
                            text_encoder_state_sha256_before=text_frozen_hash, frozen_text_encoder_audits=[])
            manifest['source_sha256'].update({name:hashlib.sha256((TASK/name).read_bytes()).hexdigest()
                for name in ('frozen_text.py','frozen_multimodal_models.py')})
    if continuation:
        manifest.update(engineering_continuation=continuation,epochs=stop_epoch,
                        first_epoch=start_epoch,editor_cache_root=str(cache_root),
                        cache_manifests={split:hashlib.sha256((cache_root/split/'manifest.json').read_bytes()).hexdigest()
                                         for split in ('train','val','test')},
                        cache_reader_provenance={split:ds.reader_provenance for split,ds in datasets.items()},
                        cache_runtime='CPU_RAM_uint8_stores_and_vectorized_exact_batch_reconstruction',
                        source_sha256={**manifest['source_sha256'],**{
                            name:hashlib.sha256((TASK/name).read_bytes()).hexdigest()
                            for name in ('performance_cache.py','continue_lgvq50.py')}})
        # Complete parent epochs remain traceable; never rewrite its manifest.
        manifest['frozen_encoder_audits'] = list(parent['manifest']['frozen_encoder_audits'])
    write_json(run_dir/'run_manifest.json', manifest)
    history, selected_epoch = [], None
    if continuation and not args.smoke_batches:
        parent_history = json.loads((resume_path.parent/'history.json').read_text())
        history = [entry for entry in parent_history if entry['epoch'] <= parent['epoch']]
        if [entry['epoch'] for entry in history] != list(range(1,start_epoch)):
            raise ValueError('Editor parent epoch history is incomplete')
    elapsed_offset = history[-1]['elapsed_seconds'] if history else 0.
    began=time.monotonic()
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM, interrupt)
    try:
        for epoch in range(start_epoch,stop_epoch+1):
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
                        'train_loss':total/count,'elapsed_seconds':elapsed_offset+time.monotonic()-began,
                        'session_elapsed_seconds':time.monotonic()-began})
            evaluated = epoch==1 or epoch%5==0 or epoch==stop_epoch
            metrics, records = problem.evaluate(model,test,device) if evaluated else ({},None)
            entry={'epoch':epoch,'train_loss':total/count,'learning_rate':.001,
                   'elapsed_seconds':elapsed_offset+time.monotonic()-began,
                   'session_elapsed_seconds':time.monotonic()-began,**metrics}
            if frozen:
                audit=assert_encoder_unchanged(model.encoder,frozen_hash)
                manifest['frozen_encoder_audits'].append({'epoch':epoch,**audit})
                if text_frozen_hash:
                    text_audit=assert_encoder_unchanged(model.text_encoder,text_frozen_hash)
                    manifest['frozen_text_encoder_audits'].append({'epoch':epoch,**text_audit})
                write_json(run_dir/'run_manifest.json',manifest)
            payload={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'manifest':manifest,'metrics':entry}
            save_checkpoint(run_dir/'last_checkpoint.pt',payload)
            if epoch == stop_epoch:
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
        metrics,records=problem.evaluate(model,test,device)
        selection_metrics=metrics
        final_loader=None
        if problem.final_test is not None:
            if resume_path:
                final_loader=CachedIndexBatches(problem.final_test,
                    DataLoader(range(len(problem.final_test)),batch_size=8,shuffle=False,num_workers=0))
            else:
                final_loader=DataLoader(problem.final_test,shuffle=False,**options)
            metrics,records=problem.evaluate(model,final_loader,device)
        if frozen:
            manifest['frozen_encoder_final_audit']=assert_encoder_unchanged(model.encoder,frozen_hash)
            if text_frozen_hash:
                manifest['frozen_text_encoder_final_audit']=assert_encoder_unchanged(model.text_encoder,text_frozen_hash)
            if small_text:
                manifest['text_state_sha256_after']=encoder_state_sha256(model.text_encoder)
                manifest['trainable_text_changed']=manifest['text_state_sha256_after']!=manifest['text_state_sha256_before']
        if continuation and not args.smoke_batches:
            if hashlib.sha256(resume_path.read_bytes()).hexdigest()!=continuation['checkpoint_sha256']:
                raise RuntimeError('Engineering parent checkpoint changed during formal continuation')
        write_json(run_dir/'fixed_weight_predictions.json',records)
        write_json(run_dir/'result.json',{'status':'complete','selected_epoch':selected_epoch,'metrics':metrics,
            'selection_metrics':selection_metrics,'selection_uses_test':False,
            'selection':'fixed_last_epoch','checkpoint':'last_checkpoint.pt',
            'checkpoint_sha256':hashlib.sha256((run_dir/'last_checkpoint.pt').read_bytes()).hexdigest(),
            'training_scope':manifest['training_scope'],
            'frozen_encoder_audit':manifest.get('frozen_encoder_final_audit'),
            'frozen_text_encoder_audit':manifest.get('frozen_text_encoder_final_audit'),
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
        if index_loader is not None:
            from .continue_lgvq50 import shutdown_loader
            shutdown_loader(index_loader)
        del model,optimizer
        torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',required=True)
    p.add_argument('--backbone',choices=['resnet20','resnet32','resnet18_imagenet_frozen'],default='resnet20')
    p.add_argument('--text-encoder',choices=['minilm','byte_gru'],default='minilm')
    p.add_argument('--max-physical-gpus',type=int,choices=[3,4,6],default=3)
    p.add_argument('--max-jobs-per-gpu',type=int,choices=[2,3],default=2)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--gpu-uuid',required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--smoke-batches',type=int,default=0)
    p.add_argument('--preflight',action='store_true')
    p.add_argument('--queue-status-path',type=Path)
    p.add_argument('--shared-owner-pids',type=int,nargs='*',default=[])
    p.add_argument('--resume-from',type=Path,help='Cached editor only: last completed epoch checkpoint')
    p.add_argument('--editor-cache-root',type=Path,help='Exactly verified deterministic train/val/test pair cache')
    run(p.parse_args())


if __name__=='__main__': main()
