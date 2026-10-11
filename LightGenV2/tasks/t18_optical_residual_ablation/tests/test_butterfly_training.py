"""Native Linux CPU verification; no real test split or GPU is accessed."""
import copy
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

pytestmark = pytest.mark.skipif(sys.platform == 'win32', reason='Native Torch unavailable on this Windows host; verify on Linux without changing shared environment')
TASK = Path(__file__).resolve().parents[1]


def module():
    sys.path.insert(0, str(TASK))
    import butterfly_ablation as a
    return a


def test_all_four_graphs_preserve_optics_and_pair_initialization():
    a = module(); torch = a.torch
    profile = a.r.read(TASK / 'butterfly_profile.json')
    manifest = dict(classes=[str(i) for i in range(60)], cache_sha256='synthetic')
    original = copy.deepcopy(a.r.CONFIGS['moe_L6_oeo_relu_softsign'])
    for depth, count in [(2, 449848), (4, 889696), (8, 1769392), (16, 3528784)]:
        graph = a.graph_config(depth, profile)
        for key in ('optics', 'prompt', 'optical_router', 'nonlinearity', 'global_oeo', 'readout', 'regularization'):
            assert graph[key] == original[key]
        a.r.setseed(17)
        cfg = a.config(profile, manifest, 0., 'base')
        left = a.build(depth, profile, cfg, device='cpu')
        digest = a.f.tensor_identity(left)
        assert sum(p.numel() for p in left.parameters()) == count
        assert len(a.install_residual(left, .3)) == 5 * depth
        assert a.f.tensor_identity(left) == digest
        assert all(not hasattr(m, 'residual_rho') for n, m in left.named_modules() if 'router' in n)
        del left
        a.r.setseed(17)
        right = a.build(depth, profile, dict(cfg, residual_rho=.3), device='cpu')
        assert a.f.tensor_identity(right) == digest
        del right
    assert a.r.CONFIGS['moe_L6_oeo_relu_softsign'] == original
    sources = a.source_identity(TASK / 'butterfly_profile.json')
    assert all((a.ROOT / name).is_file() for name in sources)


def test_real_short_cpu_loop_selects_loss_and_test_is_once_after_global_lock(tmp_path, monkeypatch):
    a = module(); torch = a.torch; np = a.np
    profile = a.r.read(TASK / 'butterfly_profile.json')
    data_path = tmp_path / 'synthetic_data'; data_path.write_bytes(b'train-and-val-fixture')
    manifest = dict(classes=[str(i) for i in range(60)], cache_sha256=a.r.sha(data_path))
    cfg = a.config(profile, manifest, 0., 'base'); src = {'synthetic': 'fixture'}
    out = tmp_path / 'arm'; out.mkdir()
    args = SimpleNamespace(data=data_path, out=out, epochs=100, candidate='base', depth=2, rho=0., resume=False,
                           suite_root=tmp_path)
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.raw_phase = torch.nn.Parameter(torch.zeros(60))
    reads = []
    def load(path, split, manifest):
        reads.append(split)
        return torch.zeros(3, 1), torch.tensor([0, 1, 0]), np.array(['x', 'y', 'z'])
    def forward(model, x):
        return model.raw_phase.softmax(0).expand(len(x), -1), torch.ones(len(x))
    val_calls = 0
    def evaluate(model, data, batch=1):
        nonlocal val_calls
        val_calls += 1
        # Every epoch also has train evaluations, so infer the sequence by call.
        loss = .7 if val_calls >= 4 else 1.
        accuracy = .99 if val_calls >= 5 else .4
        metric = dict(balanced_nll=loss, accuracy=accuracy, balanced_accuracy=.9)
        return metric, [dict(sample_id='x', label_true=0, label_pred=0, score0=1.)]
    monkeypatch.setattr(a, 'build', lambda *args, **kwargs: Toy())
    monkeypatch.setattr(a, 'load_data', load)
    monkeypatch.setattr(a, 'encode', lambda x, theta=None, **kwargs: x)
    monkeypatch.setattr(a, 'forward', forward)
    monkeypatch.setattr(a, 'evaluate', evaluate)
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, *args, **kwargs: self)
    monkeypatch.setattr(a.b, 'phase_smoothness', lambda model: model.raw_phase.square().mean())
    a.train_segment(args, profile, manifest, cfg, src)
    last = torch.load(out / 'last_checkpoint.pt', map_location='cpu', weights_only=False)
    assert float(last['optimizer']['state'][0]['step']) == 100  # One step per 3-sample tail macro.
    assert not torch.equal(last['model']['raw_phase'], last['ema']['raw_phase'])
    result = a.r.read(out / 'fit_result.json')
    assert reads == ['train', 'val'] and result['test_evaluations'] == 0
    history = a.r.read(out / 'history.json')
    assert len(history) == 100 and all('test' not in row for row in history)
    selected = min(history, key=lambda x: x['val']['balanced_nll'])
    assert result['selected_epoch'] == selected['epoch'] == 2
    assert selected['val']['accuracy'] < history[2]['val']['accuracy']
    with pytest.raises(FileNotFoundError):
        a.finalize(args, profile, manifest, cfg, src)
    assert reads == ['train', 'val']
    lock = dict(config=cfg, sources=src, data_sha256=manifest['cache_sha256'], epoch=result['selected_epoch'],
                checkpoint_sha256=result['checkpoint_sha256'], test_evaluations_before_lock=0)
    a.r.save(out / 'selection_lock.json', lock)
    arms = {str(out.resolve()): lock, **{f'dummy_{i}': {} for i in range(7)}}
    a.r.save(tmp_path / 'global_selection_lock.json', dict(test_evaluations_before_lock=0, sources=src, selected_arms=arms))
    a.finalize(args, profile, manifest, cfg, src)
    assert reads == ['train', 'val', 'test'] and a.r.read(out / 'result.json')['test_evaluations'] == 1
    with pytest.raises(AssertionError, match='Never repeat'):
        a.finalize(args, profile, manifest, cfg, src)
    assert reads.count('test') == 1


def test_tail_sample_and_auxiliary_weighting_gives_one_macro_update():
    a = module(); torch = a.torch
    p = torch.tensor([.25], requires_grad=True)
    values = torch.tensor([1., 2., 3.])  # Tail macro has 3, not the nominal 16.
    reference = ((p - values).square() + .2 * (p + values).square() + .04 * p.square()).mean()
    expected, = torch.autograd.grad(reference, p)
    for micro in values.split(1):
        loss = ((p - micro).square() + .2 * (p + micro).square() + .04 * p.square()).mean()
        (loss * len(micro) / len(values)).backward()
    assert torch.allclose(p.grad, expected)


def test_failed_preparation_stops_before_any_gpu_or_test(tmp_path, monkeypatch):
    a = module()
    data = tmp_path / 'data'; data.mkdir()
    a.r.save(data / 'preparation_process.json', dict(pid=999999999))
    out = tmp_path / 'run'
    monkeypatch.setattr(sys, 'argv', ['butterfly_ablation.py', '--phase', 'suite', '--data-root', str(data),
        '--out', str(out), '--wait-preparation', '--profile', str(TASK / 'butterfly_profile.json')])
    monkeypatch.setattr(a.f.gpu_budget, 'preflight', lambda *args, **kwargs: pytest.fail('Preparation must not allocate GPU'))
    with pytest.raises(AssertionError, match='Preparation stopped'):
        a.main()
    assert a.r.read(out / 'status.json')['state'] == 'failed'
    assert a.r.read(out / 'current_processes.json') == []
    assert not (out / 'test_started.json').exists()


def test_cpu_prediction_audit_checks_all60_ids_labels_argmax_and_metrics(tmp_path):
    a = module()
    import audit_butterfly as audit
    labels = a.np.arange(60)
    ids = a.np.array([f'specimen_{i}' for i in range(60)])
    scores = a.np.eye(60) * .9 + .1 / 60
    data = tmp_path / 'data.npz'
    a.np.savez(data, val_labels=labels, val_ids=ids)
    rows = [dict(sample_id=str(sid), label_true=int(label), label_pred=int(label),
                 **{f'score{k}': float(row[k]) for k in range(60)}) for sid, label, row in zip(ids, labels, scores)]
    a.r.csvwrite(tmp_path / 'val_predictions.csv', rows)
    expected = a.metrics(labels, scores)
    manifest = dict(split_ids={'val': ids.tolist()})
    assert audit.prediction_audit(tmp_path, data, manifest, 'val', expected)['records'] == 60
    rows[0]['label_pred'] = 1
    a.r.csvwrite(tmp_path / 'val_predictions.csv', rows)
    with pytest.raises(AssertionError):
        audit.prediction_audit(tmp_path, data, manifest, 'val', expected)
