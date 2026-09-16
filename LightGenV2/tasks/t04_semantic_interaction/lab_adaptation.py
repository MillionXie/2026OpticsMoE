"""Offline measured-CCD ablation and downstream-only head adaptation; no hardware calls."""
import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import time


def stratified_split(rows, seed=20260916):
    """Fixed 200/50 per operation, selected without looking at correctness."""
    if len(rows) != 1000 or len({r['sample_id'] for r in rows}) != 1000:
        raise ValueError('Require 1000 unique sample IDs')
    train, held = [], []
    rng = random.Random(seed)
    for task in ('add', 'replace', 'move', 'remove'):
        indices = [i for i, r in enumerate(rows) if r['task'] == task]
        if len(indices) != 250:
            raise ValueError('Require 250 samples per operation')
        rng.shuffle(indices)
        train.extend(indices[:200]); held.extend(indices[200:])
    return sorted(train), sorted(held)


def trainable_readout_name(name):
    # summarize() is used BEFORE the vision optical stages. Changing it would
    # invalidate captured inputs! It and the task classifier remain frozen.
    return name.startswith(('post_film.', 'coordinate_projection.', 'editor.', 'decoder.'))


def selection_key(metrics):
    m = metrics['overall']
    return tuple(m[k] for k in ('scene_exact_match', 'changed_cell_accuracy', 'edit_grid_iou', 'object_f1'))


def validate_resume_config(previous, current):
    for key in ('session', 'seed', 'batch_size'):
        if previous[key] != getattr(current, key):
            raise ValueError('Resume must preserve '+key)
    if current.epochs <= previous['epochs']:
        raise ValueError('Resume target must exceed previous epochs')


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def state_digest(module, names=None):
    import torch
    h = hashlib.sha256()
    for name, value in module.state_dict().items():
        if names is None or name in names:
            h.update((name+str(value.dtype)+str(tuple(value.shape))).encode())
            h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def cache_features(a, out):
    import torch
    from .lab_bench import open_session, checked_data, measured_prefix, audit
    from .lab_runtime import load_model, batch_for, replay, STAGES
    from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.metrics import MetricAccumulator
    root, session, state, config, release = open_session(a)
    if state['measured_stages'] != list(STAGES) or len(state['fields']) != 1000:
        raise ValueError('Require all 6000 actual captures')
    for stage in STAGES:
        a.stage = stage; audit(a)
        print('AUDITED', stage, flush=True)
    model, cfg = load_model(root, a.device)
    model.requires_grad_(False).eval()
    data = checked_data(root, release, cfg)
    accumulator, removed = MetricAccumulator(), MetricAccumulator()
    rows, removed_rows, spatial, condition, labels, hashes = [], [], [], [], [], []
    captured = {}
    def save_inputs(module, args):
        captured['spatial'] = args[0].detach().cpu().clone()
        captured['condition'] = args[1].detach().cpu().clone()
    hook = model.shared_readout.register_forward_pre_hook(save_inputs)
    cores = (model.language_core, model.vision_core)
    calls = [0]
    def count_optics(module, args):
        calls[0] += 1
    counters = []
    for core in cores:
        for prop in (core.optical_branch.core.router.propagator, core.optical_branch.core.propagator):
            counters.append(prop.register_forward_pre_hook(count_optics))
    try:
        for i, item in enumerate(state['fields']):
            record = data.records[item['index']]
            if record['sample_id'] != item['sample_id'] or digest(root/'data'/record['relative_dir']/'source.png') != item['source_sha256']:
                raise ValueError('Sample/source identity mismatch')
            batch = batch_for(data, item['index'], a.device)
            measured, frame_hashes = measured_prefix(session, config, item, STAGES)
            calls[0] = 0
            output, tap = replay(model, batch, measured)
            if calls[0] != 6 or tuple(tap.detectors) != STAGES:
                raise RuntimeError('All six measured boundaries must execute')
            values, _, _ = accumulator.update(output, batch)
            rows.extend(values); hashes.append(frame_hashes)
            spatial.append(captured['spatial']); condition.append(captured['condition'])
            labels.append({k: (batch[k].detach().cpu().clone() if torch.is_tensor(batch[k]) else list(batch[k]))
                           for k in ('source_grid','target_grid','edit_grid','preserve_grid','task_index','task','instruction','sample_id')})
            for core in cores: core.set_fusion_ablation('remove_optical')
            calls[0] = 0
            with torch.inference_mode():
                no_opt = model(batch['source_image'], batch['prompt_hidden'])
            if calls[0] != 0:
                raise RuntimeError('Remove-optical unexpectedly called propagation')
            values, _, _ = removed.update(no_opt, batch); removed_rows.extend(values)
            for core in cores: core.set_fusion_ablation('none')
            if (i+1) % 25 == 0:
                print('CACHE_AND_ABLATION', i+1, '/ 1000', flush=True)
    finally:
        hook.remove()
        for handle in counters: handle.remove()
        for core in cores: core.set_fusion_ablation('none')
    original = json.loads((session/'results.json').read_text(encoding='utf-8'))
    for group, values in accumulator.compute().items():
        for key, value in values.items():
            if abs(value-original['metrics'][group][key]) > 1e-7:
                raise RuntimeError('Measured baseline did not reproduce: '+group+'/'+key)
    train, held = stratified_split(rows, a.seed)
    dump(out/'split.json', dict(seed=a.seed, adaptation_indices=train, holdout_indices=held,
         adaptation_ids=[rows[i]['sample_id'] for i in train], holdout_ids=[rows[i]['sample_id'] for i in held],
         selection='best holdout scene exact, then changed-cell, IoU, F1; holdout is used for model selection',
         full1000_is_mixed_adaptation_and_holdout=True))
    dump(out/'baseline_measured.json', dict(metrics=accumulator.compute(), rows=rows))
    dump(out/'same_checkpoint_remove_optical.json', dict(metrics=removed.compute(), rows=removed_rows,
         normal=accumulator.compute(), retrained=False, propagations_per_sample=0,
         protocol='both language and vision optical feature paths off; electronic coefficient restored to 1; same checkpoint',
         checkpoint_sha256=release['checkpoint_sha256']))
    print('BASELINE_MEASURED', json.dumps(accumulator.compute()), flush=True)
    print('REMOVE_OPTICAL', json.dumps(removed.compute()), flush=True)
    cache = dict(spatial=torch.cat(spatial), condition=torch.cat(condition), labels=labels, rows=rows,
                 ccd_sha256=hashes, checkpoint_sha256=release['checkpoint_sha256'],
                 session_sha256=digest(session/'session.json'), baseline_result_sha256=digest(session/'results.json'))
    torch.save(cache, out/'readout_inputs.pt')
    return model, cfg, cache, train, held


def batch_cached(cache, indices, device):
    import torch
    batch = {}
    for key in cache['labels'][0]:
        values = [cache['labels'][i][key] for i in indices]
        batch[key] = torch.cat(values).to(device) if torch.is_tensor(values[0]) else sum(values, [])
    return cache['spatial'][indices].to(device), cache['condition'][indices].to(device), batch


def predict_head(head, spatial, condition):
    output = head(spatial, condition)
    zero = output['category_logits'].new_zeros(())
    output.update(ccd_operating_loss=zero, router_balance_loss=zero)
    return output


def evaluate_head(head, cache, train, held, device, batch_size):
    import torch
    from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.metrics import MetricAccumulator
    accumulators = {name: MetricAccumulator() for name in ('adaptation800','holdout200','all1000')}
    train_set = set(train); rows = []
    head.eval()
    with torch.inference_mode():
        for start in range(0, len(cache['labels']), batch_size):
            ids = list(range(start, min(start+batch_size, len(cache['labels']))))
            x, c, batch = batch_cached(cache, ids, device)
            output = predict_head(head, x, c)
            # Reuse the exact original accumulator; no metric reimplementation.
            values, _, _ = accumulators['all1000'].update(output, batch); rows.extend(values)
            for name, is_train in [('adaptation800', True), ('holdout200', False)]:
                positions = [j for j, i in enumerate(ids) if (i in train_set) == is_train]
                if not positions: continue
                sub_batch = {k: (v[positions] if torch.is_tensor(v) else [v[j] for j in positions]) for k,v in batch.items()}
                sub_output = {k:v[positions] for k,v in output.items() if k in ('category_logits','edit_logits','task_logits')}
                accumulators[name].update(sub_output, sub_batch)
    return {k:v.compute() for k,v in accumulators.items()}, rows


def charts(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    history = [json.loads(line) for line in (out/'epochs.jsonl').read_text().splitlines()]
    metrics = list(history[0]['metrics']['all1000']['overall'])
    metrics.remove('samples')
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    for ax, key in zip(axes.flat, metrics):
        for split in ('adaptation800','holdout200','all1000'):
            ax.plot([r['epoch'] for r in history], [r['metrics'][split]['overall'][key] for r in history], label=split)
        ax.set_title(key); ax.set_xlabel('Epoch'); ax.grid(alpha=.2)
    axes.flat[-1].axis('off'); axes.flat[0].legend()
    fig.suptitle('Measured CCD: downstream electronic readout adaptation (holdout used for selection)')
    fig.tight_layout(); fig.savefig(out/'all_metrics.png', dpi=160); plt.close(fig)


def run(a):
    import torch
    from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.objectives import editing_objective
    out = Path(a.output).resolve()
    if out.exists(): raise FileExistsError('Use a new output directory; never overwrite results')
    out.mkdir(parents=True)
    random.seed(a.seed); torch.manual_seed(a.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    if str(a.device).startswith('cuda') and not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    package = Path(__file__).resolve().parents[3]/'OFFLINE_SOURCE.json'
    provenance = json.loads(package.read_text()) if package.exists() else {'source_commit':a.source_commit}
    if not provenance.get('source_commit'): raise ValueError('Source commit required')
    dump(out/'config.json', dict(**vars(a), source=provenance, torch_version=torch.__version__,
         device_name=torch.cuda.get_device_name() if str(a.device).startswith('cuda') else 'cpu',
         protocol='frozen six measured optical passes; downstream shared readout only; no hardware',
         selection='holdout200 scene exact then changed-cell/IoU/F1; not independent final test'))
    resume = Path(a.resume_from).resolve() if a.resume_from else None
    last = None
    if resume:
        from .lab_runtime import load_model
        from .lab_bench import open_session
        previous = json.loads((resume/'config.json').read_text())
        validate_resume_config(previous, a)
        if json.loads((resume/'status.json').read_text())['status'] != 'complete':
            raise ValueError('Resume only a completed run')
        root, session, _, _, release = open_session(a)
        model, cfg = load_model(root,a.device); model.requires_grad_(False).eval()
        cache_path = resume/'readout_inputs.pt'
        if not cache_path.exists():
            cache_path = Path(json.loads((resume/'resume_provenance.json').read_text())['cache_path'])
        cache = torch.load(cache_path,map_location='cpu',weights_only=False)
        if cache['checkpoint_sha256'] != release['checkpoint_sha256'] or cache['session_sha256'] != digest(session/'session.json') or cache['baseline_result_sha256'] != digest(session/'results.json'):
            raise ValueError('Resume cache/session/checkpoint identity changed')
        split = json.loads((resume/'split.json').read_text())
        train,held = stratified_split(cache['rows'],a.seed)
        if train != split['adaptation_indices'] or held != split['holdout_indices']:
            raise ValueError('Resume split changed')
        last = torch.load(resume/'last_checkpoint.pt',map_location='cpu',weights_only=False)
        if last['epoch'] != previous['epochs'] or last['base_checkpoint_sha256'] != cache['checkpoint_sha256'] or last['session_sha256'] != cache['session_sha256']:
            raise ValueError('Resume last checkpoint mismatch')
        for name in ('split.json','baseline_measured.json','same_checkpoint_remove_optical.json','epochs.jsonl','epochs.csv','best_checkpoint.pt','best_predictions.json'):
            shutil.copy2(resume/name,out/name)
        dump(out/'resume_provenance.json',dict(previous_run=str(resume),previous_epoch=last['epoch'],
             last_sha256=digest(resume/'last_checkpoint.pt'),cache_path=str(cache_path),cache_sha256=digest(cache_path),
             learning_rate_policy='hold constant at saved final optimizer LR; no reheating or cosine reset',
             optimizer_restored=True,original_best_preserved=True))
        dump(out/'status.json',dict(status='resuming',epoch=last['epoch']))
    else:
        dump(out/'status.json', dict(status='extracting_and_ablating'))
        model, cfg, cache, train, held = cache_features(a, out)
    head = copy.deepcopy(model.shared_readout).to(a.device)
    head.requires_grad_(False)
    for name, parameter in head.named_parameters(): parameter.requires_grad_(trainable_readout_name(name))
    names = [name for name,p in head.named_parameters() if p.requires_grad]
    frozen = [name for name in head.state_dict() if not trainable_readout_name(name)]
    frozen_digest = state_digest(head, frozen)
    if last:
        head.load_state_dict(last['shared_readout'],strict=True)
        if last['trainable_names'] != names or state_digest(head,frozen) != frozen_digest:
            raise ValueError('Resume trainable scope/frozen parameters mismatch')
    original_model_digest = state_digest(model)
    dump(out/'trainable_parameters.json', dict(names=names, count=sum(p.numel() for p in head.parameters() if p.requires_grad),
         frozen_summary_and_task_head=frozen, original_model_sha256=original_model_digest))
    optimizer = torch.optim.AdamW([p for p in head.parameters() if p.requires_grad], lr=a.lr, weight_decay=1e-4)
    best_key = None; best_epoch = 0; best_metrics = None
    start_epoch = 0
    if last:
        optimizer.load_state_dict(last['optimizer'])
        resume_lr = optimizer.param_groups[0]['lr']
        best = torch.load(out/'best_checkpoint.pt',map_location='cpu',weights_only=False)
        best_epoch,best_metrics = best['epoch'],best['metrics']
        best_key = selection_key(best_metrics['holdout200'])
        metrics,_ = evaluate_head(head,cache,train,held,a.device,a.batch_size)
        error = max(abs(v-last['metrics'][s][g][k]) for s,groups in metrics.items() for g,values in groups.items() for k,v in values.items())
        if error>1e-7:raise RuntimeError('Resume metrics mismatch: '+str(error))
        print('RESUME_VERIFIED',last['epoch'],'metric_error',error,'constant_lr',resume_lr,flush=True)
        start_epoch = last['epoch']+1
    history_file = (out/'epochs.jsonl').open('a' if resume else 'x', encoding='utf-8', buffering=1)
    csv_file = (out/'epochs.csv').open('a' if resume else 'x', encoding='utf-8', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=['epoch','split','operation','metric','value','training_loss','lr'])
    if not resume:writer.writeheader()
    try:
        for epoch in range(start_epoch,a.epochs+1):
            lr = resume_lr if resume else a.lr * (0.01 + 0.99 * (1+math.cos(math.pi*max(0,epoch-1)/max(1,a.epochs-1)))/2)
            for group in optimizer.param_groups: group['lr'] = lr
            train_loss = 0.0
            if epoch:
                head.train(); order = train.copy(); random.Random(a.seed+epoch).shuffle(order)
                for start in range(0,len(order),a.batch_size):
                    ids = order[start:start+a.batch_size]
                    x,c,batch = batch_cached(cache,ids,a.device)
                    optimizer.zero_grad(set_to_none=True)
                    losses = editing_objective(predict_head(head,x,c),batch,cfg)
                    if not torch.isfinite(losses['total']): raise FloatingPointError('Nonfinite loss')
                    losses['total'].backward()
                    torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
                    optimizer.step(); train_loss += float(losses['total'].detach())*len(ids)/len(train)
            metrics, rows = evaluate_head(head,cache,train,held,a.device,a.batch_size if epoch else 1)
            if epoch == 0:
                baseline = json.loads((out/'baseline_measured.json').read_text())['metrics']
                for g, values in baseline.items():
                    for k,v in values.items():
                        if abs(v-metrics['all1000'][g][k])>1e-7: raise RuntimeError('Cached epoch0 differs from replay')
            if state_digest(head,frozen) != frozen_digest: raise RuntimeError('Upstream-used head parameters changed')
            record = dict(epoch=epoch,training_loss=train_loss,lr=lr,metrics=metrics)
            history_file.write(json.dumps(record)+'\n')
            for split, groups in metrics.items():
                for operation, values in groups.items():
                    for metric,value in values.items(): writer.writerow(dict(epoch=epoch,split=split,operation=operation,metric=metric,value=value,training_loss=train_loss,lr=lr))
            csv_file.flush()
            print('EPOCH', json.dumps(record), flush=True)
            payload = dict(epoch=epoch, shared_readout={k:v.detach().cpu() for k,v in head.state_dict().items()},
                 optimizer=optimizer.state_dict(),metrics=metrics,base_checkpoint_sha256=cache['checkpoint_sha256'],
                 session_sha256=cache['session_sha256'],trainable_names=names,source=provenance,
                 application='Load original pinned model, then model.shared_readout.load_state_dict(payload[shared_readout]); all other weights unchanged')
            torch.save(payload,out/'last_checkpoint.pt')
            key = selection_key(metrics['holdout200'])
            if best_key is None or key > best_key:
                best_key,best_epoch,best_metrics = key,epoch,metrics
                torch.save(payload,out/'best_checkpoint.pt')
                dump(out/'best_predictions.json',dict(epoch=epoch,rows=rows))
            dump(out/'status.json',dict(status='training',epoch=epoch,total_epochs=a.epochs,best_epoch=best_epoch,
                 best_holdout=best_metrics['holdout200']['overall']))
    finally:
        history_file.close(); csv_file.close()
    if state_digest(model) != original_model_digest: raise RuntimeError('Base model changed')
    dump(out/'summary.json',dict(status='complete',best_epoch=best_epoch,best_metrics=best_metrics,
         last_epoch=a.epochs,base_model_unchanged=True,frozen_summary_unchanged=True,
         selection='holdout200 scene exact then changed-cell/IoU/F1; holdout used for checkpoint selection',
         all1000_is_not_independent_test=True))
    charts(out)
    dump(out/'status.json',dict(status='complete',epoch=a.epochs,best_epoch=best_epoch))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',required=True); p.add_argument('--session',default='test1000_02')
    p.add_argument('--config',default='LAB.local.json'); p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda'); p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--batch-size',type=int,default=32); p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--seed',type=int,default=20260916); p.add_argument('--source-commit',default='')
    p.add_argument('--resume-from',default='',help='Completed prior run; restore last optimizer, split and global best; keep final LR')
    a=p.parse_args()
    if a.epochs<1 or a.batch_size<1 or a.lr<=0: p.error('Invalid optimization settings')
    a.config=str(Path(a.project)/a.config) if not Path(a.config).is_absolute() else a.config
    run(a)


if __name__=='__main__': main()
