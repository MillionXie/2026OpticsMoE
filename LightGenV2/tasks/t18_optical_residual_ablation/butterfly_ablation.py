"""Full60 optical ablation with bounded paired validation-only fit adaptation."""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import fishnet_ablation as f
import butterfly_fit_policy as fit_policy
from model import coherent_modulation
from optical_reference.optics import DetectorArray

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
torch, np, b, r = f.torch, f.np, f.b, f.r


def source_identity(profile):
    # Explicit repository paths permit runtime and published Git-blob audits.
    files = [b.TASK / path for path in f.t.source_identity()['historical']]
    files += [HERE / name for name in ('run.py', 'model.py', 'campaign.py', 'prepare_mango.py',
        'fishnet_ablation.py', 'fishnet_selection.py', 'gpu_budget.py', 'prepare_fishnet.py',
        'fishnet_profile.json', 'butterfly_ablation.py', 'butterfly_fit_policy.py', 'prepare_butterfly.py')]
    files.append(profile)
    return {str(p.resolve().relative_to(ROOT)).replace('\\', '/'): r.sha(p) for p in files}


def graph_config(depth, profile):
    assert depth in profile['depths'] == [2, 4, 8, 16]
    cycles = depth // 2
    graph = copy.deepcopy(r.CONFIGS['moe_L6_oeo_relu_softsign'])
    graph['model']['num_cycles'] = cycles
    graph['expert_bank']['d2nn'].update(num_layers=cycles, nonlinear_schedule=[True] * cycles)
    graph['global_fc']['num_masks'] = cycles
    graph['experiment']['baseline_layers'] = depth
    graph['experiment']['expected_parameters']['moe'] = 10000 + 439848 * cycles
    graph['variant'].update(id=f'moe_L{depth}_oeo_relu_softsign', depth=depth)
    graph['detector'].update(profile['detector'])
    return graph


class MultiClassMoE(b.OpticalMoE):
    def __init__(self, graph, classes):
        torch.nn.Module.__init__(self)
        self.net = b.RelayFanoutMoE(graph, num_classes=classes)


def install_residual(model, rho):
    assert rho in (0., .3)
    class ResidualPhase(f.t.PhaseLayer):
        def forward(self, field):
            assert not self._dropout_enabled()
            return coherent_modulation(field, self.get_phase(), self.residual_rho)
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, f.t.PhaseLayer) and ('.expert_bank.' in name or '.global_fcs.' in name):
            module.__class__ = ResidualPhase
            module.residual_rho = float(rho)
            targets.append(name)
    assert len(targets) == model.net.num_cycles * 10
    assert all('router' not in name for name in targets)
    model.net.expert_bank.vectorize_homogeneous_d2nn = False
    return targets


def build(depth, profile, cfg, residual=True, device='cuda'):
    model = MultiClassMoE(graph_config(depth, profile), len(cfg['classes']))
    model.net.expert_bank.vectorize_homogeneous_d2nn = False
    assert model.masks.shape == (60, 500, 500)
    assert float(model.masks.sum(0).max()) == 1
    assert torch.all(model.masks.sum((-1, -2)) == 32 * 32)
    with torch.no_grad():
        for name, p in model.named_parameters():
            assert name.endswith('raw_phase')
            p.uniform_(-cfg['phase_init_raw_uniform'], cfg['phase_init_raw_uniform'])
    if residual:
        install_residual(model, cfg['residual_rho'])
    assert sum(p.numel() for p in model.parameters()) == 10000 + 439848 * (depth // 2)
    assert len(list(model.parameters())) == 1 + 5 * depth
    assert not any(isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)) for m in model.modules())
    return model.to(device)


def config(profile, manifest, rho, candidate):
    cfg = copy.deepcopy(profile)
    cfg.update(classes=manifest['classes'], residual_rho=rho, candidate=candidate,
        residual_scope='expert_and_global_only', residual_formula='((1-rho)*exp(i*phi)+rho)*U',
        data_sha256=manifest['cache_sha256'], selection_split='val')
    if candidate != 'base':
        cfg.update(copy.deepcopy(profile['restart_candidates'][candidate]))
    return cfg


def load_data(path, split, manifest):
    # Opening NPZ only materializes the requested train or val arrays during fit.
    with np.load(path, allow_pickle=False) as z:
        x = z[split + '_images'].copy()
        y = z[split + '_labels'].copy()
        ids = z[split + '_ids'].copy()
    assert x.dtype == np.uint8 and x.shape == (len(y), 150, 150, 3)
    assert np.bincount(y, minlength=60).tolist() == manifest['supports'][split]
    assert ids.tolist() == manifest['split_ids'][split]
    return torch.from_numpy(x.transpose(0, 3, 1, 2).copy()), torch.from_numpy(y).long(), ids


def encode(images, theta=None, device='cuda'):
    return b.encode(images.to(device=device, dtype=torch.float32) / 255., theta)


def forward(model, amplitude):
    out = model(amplitude)
    energies = out['energies']
    prob = (energies + 1e-12) / (energies.sum(1, keepdim=True) + energies.shape[1] * 1e-12)
    intensity = out['intensity']
    capture = (intensity * model.masks.sum(0)).sum((-1, -2)) / intensity.sum((-1, -2)).clamp_min(1e-12)
    return prob, capture


def metrics(y, p):
    classes = p.shape[1]
    pred = p.argmax(1)
    support = np.bincount(y, minlength=classes)
    assert np.all(support > 0) and np.isfinite(p).all()
    assert np.allclose(p.sum(1), 1, atol=1e-5)
    cm = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    recall = np.diag(cm) / support
    precision = np.diag(cm) / np.maximum(cm.sum(0), 1)
    f1 = 2 * recall * precision / np.maximum(recall + precision, 1e-12)
    nll = -np.log(np.maximum(p[np.arange(len(y)), y], 1e-12))
    return dict(n=len(y), support=support.tolist(), accuracy=float((pred == y).mean()),
        balanced_accuracy=float(recall.mean()), macro_f1=float(f1.mean()), recall=recall.tolist(),
        confusion_matrix=cm.tolist(), nll=float(nll.mean()),
        balanced_nll=float(np.mean([nll[y == k].mean() for k in range(classes)])))


@torch.no_grad()
def evaluate(model, data, batch=1):
    assert batch == 1
    model.eval()
    probs, captures = [], []
    for idx in torch.arange(len(data[1])).split(batch):
        p, capture = forward(model, encode(data[0][idx]))
        probs.append(p.cpu().numpy()); captures.append(capture.cpu().numpy())
    p = np.concatenate(probs).astype(np.float64)
    y = data[1].numpy()
    result = metrics(y, p)
    result['detector_capture'] = float(np.concatenate(captures).mean())
    rows = [dict(sample_id=str(sid), label_true=int(label), label_pred=int(q.argmax()),
        **{f'score{k}': float(q[k]) for k in range(p.shape[1])}) for sid, label, q in zip(data[2], y, p)]
    return result, rows


def objective(model, prob, capture, y, weights, cfg):
    target = torch.nn.functional.one_hot(y, prob.shape[1]) * (1 - cfg['label_smoothing']) + cfg['label_smoothing'] / prob.shape[1]
    nll = (-(target * prob.clamp_min(1e-12).log()).sum(1) * weights[y]).mean()
    return nll - cfg['capture_weight'] * capture.clamp_min(1e-12).log().mean() + cfg['phase_smooth_weight'] * b.phase_smoothness(model)


def smoke(a, profile, manifest, cfg):
    r.setseed(cfg['seed'])
    model = build(a.depth, profile, cfg, residual=False)
    data = load_data(a.data, 'train', manifest)
    initial = f.tensor_identity(model)
    inputs = [encode(data[0][i:i + 1]) for i in range(2)]
    with torch.no_grad():
        base = torch.cat([forward(model, x)[0] for x in inputs])
    install_residual(model, 0.)
    with torch.no_grad():
        zero = torch.cat([forward(model, x)[0] for x in inputs])
    assert torch.equal(base, zero) and initial == f.tensor_identity(model)
    targets = install_residual(model, .3)
    losses, probabilities = [], []
    weights = torch.ones(60, device='cuda')
    for i, x in enumerate(inputs):
        p, capture = forward(model, x)
        probabilities.append(p.detach())
        losses.append(objective(model, p, capture, data[1][i:i + 1].cuda(), weights, cfg) / 2)
    sum(losses).backward()
    full = {name: p.grad.detach().clone() for name, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    for i, x in enumerate(inputs):
        p, capture = forward(model, x)
        (objective(model, p, capture, data[1][i:i + 1].cuda(), weights, cfg) / 2).backward()
    gradients = {}
    for name, p in model.named_parameters():
        assert torch.isfinite(p.grad).all() and float(p.grad.norm()) > 0
        gradients[name] = dict(norm=float(p.grad.norm()),
            relative_l2=float((p.grad - full[name]).norm() / full[name].norm().clamp_min(1e-20)),
            cosine=float(torch.nn.functional.cosine_similarity(p.grad.flatten(), full[name].flatten(), dim=0)))
    with torch.no_grad():
        repeat = torch.cat([forward(model, x)[0] for x in inputs])
    error = float((repeat - torch.cat(probabilities)).abs().max())
    assert repeat.shape == (2, 60) and error <= 1e-6
    assert max(x['relative_l2'] for x in gradients.values()) <= .02
    assert min(x['cosine'] for x in gradients.values()) >= .999
    r.save(a.out / 'smoke.json', dict(passed=True, depth=a.depth, rho0_exact_identity=True,
        initialization_unchanged=True, router_has_residual=False, targets=len(targets), gradients=gradients,
        repeated_micro1_probability_error=error, peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024 ** 2,
        scope='Production micro1 graphs: joint mean loss vs separately accumulated sample-weighted loss; no batch2-equivalence claim; train only'))


def train_segment(a, profile, manifest, cfg, src):
    assert a.epochs in (100, 150, 200) and (a.candidate == 'base' or a.epochs == 100)
    data, val = [load_data(a.data, split, manifest) for split in ('train', 'val')]
    r.setseed(cfg['seed'])
    model = build(a.depth, profile, cfg)
    initial_sha = f.tensor_identity(model)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=0.)
    sc = cfg['scheduler']
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=sc['factor'], patience=sc['patience'], min_lr=sc['min_lr'])
    history, orders, transforms, gradients = [], [], [], []
    best, start, elapsed = float('inf'), 1, 0.
    if a.resume:
        ck = torch.load(a.out / 'last_checkpoint.pt', map_location='cpu', weights_only=False)
        assert ck['config'] == cfg and ck['sources'] == src and ck['data_sha256'] == manifest['cache_sha256']
        assert ck['depth'] == a.depth and ck['epoch'] in (100, 150)
        model.load_state_dict(ck['model']); ema.load_state_dict(ck['ema'])
        opt.load_state_dict(ck['optimizer']); scheduler.load_state_dict(ck['scheduler'])
        # Adam states loaded on CPU must follow their corresponding CUDA parameter.
        for state in opt.state.values():
            for key, value in state.items():
                if torch.is_tensor(value) and key != 'step':
                    state[key] = value.cuda()
        history = r.read(a.out / 'history.json'); orders = ck['orders']; transforms = ck['transforms']
        gradients = r.read(a.out / 'gradients.json'); initial_sha = ck['initial_parameter_sha256']
        start, elapsed = ck['epoch'] + 1, ck['seconds']
        best = min(x['val']['balanced_nll'] for x in history)
        assert len(history) == ck['epoch'] and initial_sha == ck['initial_parameter_sha256']
        del ck
    else:
        r.save(a.out / 'initial_validation.json', evaluate(model, val)[0])
    weights = (len(data[1]) / (60 * torch.bincount(data[1], minlength=60).float())).cuda()
    started = time.time()
    for epoch in range(start, a.epochs + 1):
        model.train()
        order = r.epoch_order(cfg['seed'], epoch, len(data[1]))
        theta = b.affine_parameters(len(data[1]), cfg['seed'], epoch, cfg['augmentation'])
        orders.append(r.sha_tensor(order)); transforms.append(r.sha_tensor(theta))
        total = 0.
        for batch_index, idx in enumerate(order.split(cfg['batch_size'])):
            opt.zero_grad(set_to_none=True)
            for micro in idx.split(cfg['microbatch']):
                p, capture = forward(model, encode(data[0][micro], theta[micro]))
                loss = objective(model, p, capture, data[1][micro].cuda(), weights, cfg)
                assert torch.isfinite(loss)
                (loss * len(micro) / len(idx)).backward()
                total += float(loss.detach()) * len(micro)
            if batch_index == 0:
                norms = {name: float(p.grad.norm()) for name, p in model.named_parameters()}
                assert all(np.isfinite(v) and v > 0 for v in norms.values())
                gradients.append(dict(epoch=epoch, norms=norms))
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            with torch.no_grad():
                for ep, mp in zip(ema.parameters(), model.parameters()):
                    ep.lerp_(mp, 1 - cfg['ema_decay'])
        vm, rows = evaluate(ema, val)
        entry = dict(epoch=epoch, train_loss=total / len(data[1]), lr=opt.param_groups[0]['lr'], val=vm)
        if epoch == 1 or epoch % 5 == 0:
            entry['train'] = evaluate(ema, data)[0]
        if vm['balanced_nll'] < best:
            best = vm['balanced_nll']
            r.save_torch(a.out / 'best_checkpoint.pt', dict(model=ema.state_dict(), epoch=epoch, depth=a.depth,
                config=cfg, sources=src, data_sha256=manifest['cache_sha256'], selected_state_kind='ema',
                selection_split='val', validation=vm))
            r.csvwrite(a.out / 'val_predictions.csv', rows)
        history.append(entry); scheduler.step(vm['balanced_nll'])
        seconds = elapsed + time.time() - started
        r.save(a.out / 'history.json', history); r.save(a.out / 'gradients.json', gradients)
        r.save_torch(a.out / 'last_checkpoint.pt', dict(model=model.state_dict(), ema=ema.state_dict(),
            optimizer=opt.state_dict(), scheduler=scheduler.state_dict(), epoch=epoch, depth=a.depth,
            config=cfg, sources=src, data_sha256=manifest['cache_sha256'], orders=orders, transforms=transforms,
            initial_parameter_sha256=initial_sha, seconds=seconds))
        status = dict(state='training', depth=a.depth, rho=a.rho, candidate=a.candidate, epoch=epoch,
            target_epochs=a.epochs, seconds=seconds, val_accuracy=vm['accuracy'], val_balanced_nll=vm['balanced_nll'],
            best_validation_balanced_nll=best, test_evaluations=0)
        r.save(a.out / 'status.json', status); print(json.dumps(status), flush=True)
    ck = torch.load(a.out / 'best_checkpoint.pt', map_location='cpu', weights_only=False)
    selected = min(history, key=lambda x: x['val']['balanced_nll'])
    assert selected['epoch'] == ck['epoch'] and selected['val'] == ck['validation']
    ema.load_state_dict(ck['model'])
    train_metric, rows = evaluate(ema, data)
    r.csvwrite(a.out / 'train_predictions.csv', rows)
    result = dict(depth=a.depth, rho=a.rho, candidate=a.candidate, parameters=sum(p.numel() for p in model.parameters()),
        initial_parameter_sha256=initial_sha, epochs_completed=a.epochs, selected_epoch=ck['epoch'],
        selected_state_kind='ema', selected_validation=ck['validation'], metrics=dict(train=train_metric, val=ck['validation']),
        checkpoint=str((a.out / 'best_checkpoint.pt').resolve()), checkpoint_sha256=r.sha(a.out / 'best_checkpoint.pt'),
        data_sha256=manifest['cache_sha256'], sources=src, orders=orders, transforms=transforms,
        seconds=elapsed + time.time() - started, selection_split='val', test_evaluations=0, test_used_for_selection=False,
        fit_diagnostic=fit_policy.diagnose(history))
    r.save(a.out / f'fit_result_e{a.epochs}.json', result); r.save(a.out / 'fit_result.json', result)


def finalize(a, profile, manifest, cfg, src):
    assert not (a.out / 'result.json').exists(), 'Never repeat a completed test evaluation'
    assert not (a.out / 'test_started.json').exists(), 'Ambiguous test attempt must be audited before retry'
    lock = r.read(a.out / 'selection_lock.json')
    global_lock = r.read(a.suite_root / 'global_selection_lock.json')
    assert global_lock['test_evaluations_before_lock'] == lock['test_evaluations_before_lock'] == 0
    assert lock['config'] == cfg and lock['sources'] == src and lock['data_sha256'] == r.sha(a.data)
    assert lock['checkpoint_sha256'] == r.sha(a.out / 'best_checkpoint.pt')
    assert global_lock['sources'] == src and len(global_lock['selected_arms']) == 8
    assert str(a.out.resolve()) in global_lock['selected_arms']
    assert global_lock['selected_arms'][str(a.out.resolve())]['checkpoint_sha256'] == lock['checkpoint_sha256']
    ck = torch.load(a.out / 'best_checkpoint.pt', map_location='cpu', weights_only=False)
    assert ck['config'] == cfg and ck['sources'] == src and ck['epoch'] == lock['epoch']
    result = r.read(a.out / 'fit_result.json')
    assert result['test_evaluations'] == 0 and result['checkpoint_sha256'] == lock['checkpoint_sha256']
    model = build(a.depth, profile, cfg); model.load_state_dict(ck['model'])
    r.save(a.out / 'test_started.json', dict(time=r.now(), selection_lock_sha256=r.sha(a.out / 'selection_lock.json'),
        global_lock_sha256=r.sha(a.suite_root / 'global_selection_lock.json'), checkpoint_sha256=lock['checkpoint_sha256']))
    test = load_data(a.data, 'test', manifest)
    result['metrics']['test'], rows = evaluate(model, test)
    r.csvwrite(a.out / 'test_predictions.csv', rows)
    result.update(test_evaluations=1, test_previously_observed=False, test_used_for_selection=False, scope=cfg['scope'])
    r.save(a.out / 'result.json', result)


def worker(a, profile):
    assert a.gpu in profile['gpu_uuids'] and os.environ['CUDA_VISIBLE_DEVICES'] == a.gpu
    f.gpu_budget.ALLOWED.update(profile['gpu_uuids'])  # Process-local permission, no old file edits.
    policy = f.gpu_budget.configure(torch, a.gpu, shared=True, memory_gib=profile['memory_gib'])
    torch.set_num_threads(4)
    manifest = r.read(a.data.parent / 'data_manifest.json')
    assert manifest['dataset'] == profile['dataset'] and manifest['source_revision'] == profile['data_revision']
    assert len(manifest['classes']) == 60 and manifest['cache_sha256'] == r.sha(a.data)
    assert manifest['preparation_source_sha256'] == r.sha(HERE / 'prepare_butterfly.py')
    src = source_identity(a.profile)
    cfg = config(profile, manifest, a.rho, a.candidate)
    if a.resume or a.phase == 'finalize':
        metadata = r.read(a.out / 'metadata.json')
        assert metadata['config'] == cfg and metadata['sources'] == src
    else:
        a.out.mkdir(parents=True, exist_ok=False)
        r.save(a.out / 'metadata.json', dict(command=sys.argv, pid=os.getpid(), config=cfg, sources=src,
            graph=graph_config(a.depth, profile), depth=a.depth, data_sha256=r.sha(a.data),
            manifest_sha256=r.sha(a.data.parent / 'data_manifest.json'), gpu_policy=policy,
            git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            environment=f.t.m.environment(), selection_split='val', time=r.now(), training_reads_test=False))
    try:
        if a.phase == 'smoke':
            smoke(a, profile, manifest, cfg)
        elif a.phase == 'fit':
            train_segment(a, profile, manifest, cfg, src)
        else:
            finalize(a, profile, manifest, cfg, src)
        r.save(a.out / 'status.json', dict(state='complete' if a.phase == 'finalize' else a.phase + '_complete',
            test_evaluations=int(a.phase == 'finalize'), time=r.now(), gpu_released_on_exit=True))
    except BaseException:
        r.save(a.out / 'status.json', dict(state='failed', traceback=traceback.format_exc(), time=r.now()))
        raise
    finally:
        gc.collect(); torch.cuda.empty_cache()


def suite(a, profile):
    assert os.environ.get('CUDA_VISIBLE_DEVICES', '') == ''
    a.out.mkdir(parents=True, exist_ok=False)
    src = source_identity(a.profile)
    data = a.data_root / 'butterfly_fixed_split.npz'
    if not (a.data_root / 'data_manifest.json').exists():
        assert a.wait_preparation, 'Prepared data missing; explicit wait required'
        wait_start = time.time()
        r.save(a.out / 'preparation_wait.json', dict(pid=os.getpid(), sources=src, data_root=str(a.data_root),
            command=sys.argv, time=r.now(), git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()))
        r.save(a.out / 'current_processes.json', [])
        while not (a.data_root / 'data_manifest.json').exists():
            receipt = r.read(a.data_root / 'preparation_process.json')
            assert Path('/proc', str(receipt['pid'])).exists(), 'Preparation stopped before manifest; preserve failure and do not train'
            assert source_identity(a.profile) == src, 'Dependency changed during preparation wait'
            assert time.time() - wait_start < 7200, 'Preparation timeout; do not occupy GPUs'
            r.save(a.out / 'status.json', dict(state='waiting_data_preparation', preparation_pid=receipt['pid'],
                gpu_processes=0, test_evaluations=0, seconds=time.time() - wait_start, time=r.now()))
            time.sleep(10)
    manifest = r.read(a.data_root / 'data_manifest.json')
    f.gpu_budget.ALLOWED.update(profile['gpu_uuids'])
    r.save(a.out / 'metadata.json', dict(command=sys.argv, pid=os.getpid(), sources=src, profile=profile,
        data_sha256=r.sha(data), manifest_sha256=r.sha(a.data_root / 'data_manifest.json'), time=r.now(),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()))
    pending, running, logs = [], {}, []
    completed, selected = set(), {}
    def stop_requested(signum, frame):
        raise SystemExit(f'Stop requested: signal {signum}')
    signal.signal(signal.SIGTERM, stop_requested)
    def arm(depth, rho, candidate='base'):
        return a.out / candidate / f'L{depth}_rho{rho:.1f}'
    def launch(job, gpu):
        phase, depth, rho, candidate, epochs, resume = job
        folder = a.out / f'smoke_L{depth}' if phase == 'smoke' else arm(depth, rho, candidate)
        cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--phase', phase, '--depth', str(depth),
            '--rho', str(rho), '--candidate', candidate, '--epochs', str(epochs), '--data', str(data),
            '--profile', str(a.profile), '--out', str(folder), '--gpu', gpu]
        if resume:
            cmd.append('--resume')
        if phase == 'finalize':
            cmd.extend(['--suite-root', str(a.out)])
        assert source_identity(a.profile) == src, 'Training dependency changed; stop before a new child'
        f.gpu_budget.preflight(gpu, shared=True, memory_gib=profile['memory_gib'])
        logfile = a.out / 'logs' / f'{phase}_L{depth}_rho{rho}_{candidate}_e{epochs}.log'
        log = logfile.open('x'); logs.append(log)
        proc = subprocess.Popen(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu), stdout=log, stderr=subprocess.STDOUT)
        receipt = dict(pid=proc.pid, command=cmd, gpu_uuid=gpu, phase=phase, depth=depth, rho=rho, candidate=candidate)
        running[gpu] = (proc, job, receipt)
    def drain():
        while pending or running:
            for gpu in profile['gpu_uuids']:
                if pending and gpu not in running:
                    launch(pending.pop(0), gpu)
            for gpu, (proc, job, _) in list(running.items()):
                if proc.poll() is not None:
                    assert proc.returncode == 0, ('Child failed; preserve runs', job, proc.returncode)
                    completed.add(job); del running[gpu]
            r.save(a.out / 'current_processes.json', [value[2] for value in running.values()])
            r.save(a.out / 'status.json', dict(state='running', completed_jobs=len(completed), pending_jobs=len(pending),
                selected_depths=sorted(selected), processes=[value[2] for value in running.values()], time=r.now()))
            if pending or running:
                time.sleep(5)
    (a.out / 'logs').mkdir()
    try:
        # All four depths must pass before any optimization or test.
        pending.extend(('smoke', d, .3, 'base', 100, False) for d in [16, 8, 4, 2])
        drain()
        pending.extend(('fit', d, rho, 'base', 100, False) for d in [16, 8, 4, 2] for rho in profile['rhos'])
        drain()
        for depth in [16, 8, 4, 2]:
            while True:
                paths = [arm(depth, rho) for rho in profile['rhos']]
                histories = [r.read(p / 'history.json') for p in paths]
                action = fit_policy.pair_action(histories)
                r.save(a.out / f'fit_decision_L{depth}_e{len(histories[0])}.json', action)
                if action['action'] != 'extend_both':
                    break
                pending.extend(('fit', depth, rho, 'base', action['target_epochs'], True) for rho in profile['rhos'])
                drain()
            candidates = [('base', [r.read(p / 'fit_result.json') for p in paths])]
            if action['action'] == 'one_paired_restart':
                candidate = action['candidate']
                pending.extend(('fit', depth, rho, candidate, 100, False) for rho in profile['rhos'])
                drain()
                candidates.append((candidate, [r.read(arm(depth, rho, candidate) / 'fit_result.json') for rho in profile['rhos']]))
            choice = fit_policy.choose_pair(candidates)
            selected[depth] = choice['candidate']
            paths = [arm(depth, rho, choice['candidate']) for rho in profile['rhos']]
            results = [r.read(p / 'fit_result.json') for p in paths]
            metas = [r.read(p / 'metadata.json') for p in paths]
            for key in ('initial_parameter_sha256', 'epochs_completed', 'orders', 'transforms', 'data_sha256', 'parameters', 'sources'):
                assert results[0][key] == results[1][key], ('Pair identity mismatch', key)
            different = {key for key in metas[0]['config'] if metas[0]['config'][key] != metas[1]['config'][key]}
            assert different == {'residual_rho'}
            r.save(a.out / f'pair_lock_L{depth}.json', dict(depth=depth, choice=choice, config_difference=['residual_rho'],
                initial_parameter_sha256=results[0]['initial_parameter_sha256'], epochs_completed=results[0]['epochs_completed'],
                paths=[str(p.resolve()) for p in paths], sources=src, test_evaluations_before_lock=0))
            for path, result, meta in zip(paths, results, metas):
                assert result['test_evaluations'] == 0 and r.sha(path / 'best_checkpoint.pt') == result['checkpoint_sha256']
                r.save(path / 'selection_lock.json', dict(epoch=result['selected_epoch'], checkpoint_sha256=result['checkpoint_sha256'],
                    config=meta['config'], sources=src, data_sha256=manifest['cache_sha256'], selected_validation=result['selected_validation'],
                    selection_split='val', test_evaluations_before_lock=0, time=r.now()))
        selected_arms = {str(arm(d, rho, selected[d]).resolve()): r.read(arm(d, rho, selected[d]) / 'selection_lock.json')
                         for d in profile['depths'] for rho in profile['rhos']}
        assert source_identity(a.profile) == src
        r.save(a.out / 'global_selection_lock.json', dict(selected_arms=selected_arms, selected_candidates=selected,
            sources=src, data_sha256=manifest['cache_sha256'], test_evaluations_before_lock=0, time=r.now()))
        pending.extend(('finalize', d, rho, selected[d], 100, False) for d in profile['depths'] for rho in profile['rhos'])
        drain()
        results = [r.read(arm(d, rho, selected[d]) / 'result.json') for d in profile['depths'] for rho in profile['rhos']]
        assert len(results) == 8 and all(x['test_evaluations'] == 1 for x in results)
        r.save(a.out / 'results.json', results)
        r.save(a.out / 'status.json', dict(state='complete', selected_arms=8, test_evaluations=8,
            gpu_released_on_exit=True, time=r.now()))
    except BaseException:
        for proc, _, _ in running.values():
            if proc.poll() is None:
                proc.terminate(); proc.wait()
        r.save(a.out / 'status.json', dict(state='failed', traceback=traceback.format_exc(), time=r.now()))
        raise
    finally:
        for log in logs:
            log.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['suite', 'smoke', 'fit', 'finalize'], required=True)
    p.add_argument('--profile', type=Path, default=HERE / 'butterfly_profile.json')
    p.add_argument('--data-root', type=Path)
    p.add_argument('--data', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--depth', type=int, choices=[2, 4, 8, 16], default=16)
    p.add_argument('--rho', type=float, choices=[0., .3], default=.3)
    p.add_argument('--candidate', choices=['base', 'regularized', 'optimization'], default='base')
    p.add_argument('--epochs', type=int, choices=[100, 150, 200], default=100)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--wait-preparation', action='store_true')
    p.add_argument('--gpu')
    p.add_argument('--suite-root', type=Path)
    a = p.parse_args(); profile = r.read(a.profile)
    assert profile['microbatch'] == profile['evaluation_batch_size'] == 1 and profile['batch_size'] == 16
    assert profile['initial_epochs'] == profile['restart_epochs'] == 100 and profile['maximum_base_epochs'] == 200
    assert len(profile['gpu_uuids']) == 3 and profile['test_development'] is False
    if a.phase == 'suite':
        assert a.data_root
        try:
            suite(a, profile)
        except BaseException:
            if a.out.exists():
                r.save(a.out / 'status.json', dict(state='failed', traceback=traceback.format_exc(), time=r.now()))
            raise
    else:
        assert a.data
        worker(a, profile)


if __name__ == '__main__':
    main()
