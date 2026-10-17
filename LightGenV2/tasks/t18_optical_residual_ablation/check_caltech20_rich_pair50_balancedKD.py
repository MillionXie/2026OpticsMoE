"""CPU admission: exact frozen parents and independent temperature-KD gradient."""
import argparse, os
from pathlib import Path
import numpy as np
import torch
import caltech20_rich_pair50_balancedKD as m
SYNTHETIC = m.HERE / 'runs/smoke/caltech20_rich_pair50_balancedKD_synthetic_cpu_20261017'

def synthetic_update(cfg):

    class Masked(torch.nn.Linear):

        def __init__(self):
            super().__init__(4, 20)
            self.mask = None

        def forward(self, x):
            return torch.nn.functional.linear(x if self.mask is None else x * self.mask, self.weight, self.bias)

    class Toy(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.raw_router = torch.nn.Parameter(torch.randn(4))
            self.phase_sets = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(4))])
            self.linear = Masked()
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            z = x * self.raw_router + self.phase_sets[0]
            return (torch.softmax(self.linear(z), 1), torch.sigmoid(z.mean(1)))

        def phase_smoothness(self):
            return self.raw_router.square().mean() + self.phase_sets[0].square().mean()
    torch.manual_seed(17)
    model = Toy().double()
    x = torch.randn(7, 4, dtype=torch.float64)
    y = torch.nn.functional.one_hot(torch.arange(7), 20).double() * 0.95 + 0.05 / 20
    target = torch.softmax(torch.randn(7, 20, dtype=torch.float64), 1)
    mass = torch.tensor([1, 0.5, 1, 0, 0.7, 1, 1], dtype=torch.float64)
    mask = torch.ones(7, 4, dtype=torch.float64)
    weights = torch.ones(20, dtype=torch.float64)
    phase = [p for n, p in model.named_parameters() if n == 'raw_router' or n.startswith('phase_sets.')]
    electronic = [p for n, p in model.named_parameters() if n != 'raw_router' and (not n.startswith('phase_sets.'))]
    opt = torch.optim.AdamW([dict(params=phase, weight_decay=0), dict(params=electronic, weight_decay=0.01)], lr=0.001)
    _, _, g1, _, _ = m.balanced_gradients(model, x, y, target, mass, mask, weights, cfg, 1)
    _, _, g4, _, _ = m.balanced_gradients(model, x, y, target, mass, mask, weights, cfg, 4)
    error = max((float(np.abs(g1[n] - g4[n]).max()) for n in g1))
    assert error < 1e-12, error
    m.balanced_gradients(model, x, y, target, mass, mask, weights, cfg, 4, True, True)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
    opt.step()
    import copy
    old = copy.deepcopy(opt.state_dict())
    before = {n: p.detach().numpy().copy() for n, p in model.named_parameters()}
    _, _, _, info, bound = m.balanced_gradients(model, x, y, target, mass, mask, weights, cfg, 4, True, True, True)
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True))
    opt.step()
    after = opt.state_dict()
    names = [[n for n in before if n == 'raw_router' or n.startswith('phase_sets.')], [n for n in before if n != 'raw_router' and (not n.startswith('phase_sets.'))]]
    grads = {}
    scales = []
    for group, keys in enumerate(names):
        gc = {n: bound['CE_gradients'][n].numpy() for n in keys}
        gk = {n: bound['KD_gradients'][n].numpy() for n in keys}
        cn = np.sqrt(sum((np.square(v).sum() for v in gc.values())))
        kn = np.sqrt(sum((np.square(v).sum() for v in gk.values())))
        eta = min(cfg['KD_scale_cap'], cfg['KD_gradient_fraction'] * cn / (kn + 1e-12))
        scales.append(eta)
        for n in keys:
            grads[n] = gc[n] + eta * gk[n]
    expected_norm = np.sqrt(sum((np.square(g).sum() for g in grads.values())))
    assert abs(expected_norm - norm) < 1e-12
    factor = min(1, 1 / (expected_norm + 1e-06))
    raw_error = 0.0
    moment_error = 0.0
    for group, keys in zip(old['param_groups'], names):
        for pid, n in zip(group['params'], keys):
            prev = old['state'][pid]
            now = after['state'][pid]
            b1, b2 = group['betas']
            grad = grads[n] * factor
            mm = b1 * prev['exp_avg'].numpy() + (1 - b1) * grad
            vv = b2 * prev['exp_avg_sq'].numpy() + (1 - b2) * grad ** 2
            step = int(float(prev['step'])) + 1
            assert step == 2 == int(float(now['step']))
            expected = before[n] * (1 - group['lr'] * group['weight_decay']) - group['lr'] / (1 - b1 ** step) * mm / (np.sqrt(vv) / np.sqrt(1 - b2 ** step) + group['eps'])
            raw_error = max(raw_error, float(np.abs(expected - dict(model.named_parameters())[n].detach().numpy()).max()))
            moment_error = max(moment_error, float(np.abs(mm - now['exp_avg'].numpy()).max()), float(np.abs(vv - now['exp_avg_sq'].numpy()).max()))
    assert raw_error < 1e-12 and moment_error < 1e-12
    return dict(synthetic_models=1, synthetic_forward_calls=model.calls, synthetic_optimizer_steps=2, micro1vs4_maxabs=error, Adam_raw_maxabs=raw_error, Adam_moment_maxabs=moment_error)

def check(out):
    assert out == m.CPU and (not out.exists()) and (os.environ.get('CUDA_VISIBLE_DEVICES') == '') and (m.base.slots()['total'] == 0)
    m.claim(out.with_name(out.name + '.claim.json'), 'own_KD_CPU')
    controls = m.admit()
    cfg = m.config()
    target, valid = m.teacher_targets()
    assert target.shape == (3141, 20) and np.allclose(target.sum(1), 1, atol=2e-07) and (valid.sum() >= 0.95 * len(valid))
    rng = np.random.default_rng(17)
    z = rng.normal(size=(7, 10))
    q = rng.uniform(0.01, 1, (7, 10))
    q /= q.sum(1, keepdims=True)
    zt = torch.tensor(z, dtype=torch.float64, requires_grad=True)
    qt = torch.tensor(q, dtype=torch.float64)
    T = 4.0
    loss = T * T * (qt * (qt.log() - torch.log_softmax(zt / T, 1))).sum(1).mean()
    g = torch.autograd.grad(loss, zt)[0].numpy()
    p = np.exp(z / T - (z / T).max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    error = float(np.abs(g - T * (p - q) / 7).max())
    assert error < 1e-12
    gc = np.array([1.0, -2.0, 0.4])
    gk = np.array([20.0, 3.0, -0.5])
    eta = min(0.1, 0.25 * np.linalg.norm(gc) / (np.linalg.norm(gk) + 1e-12))
    assert np.linalg.norm(eta * gk) <= 0.25 * np.linalg.norm(gc) + 1e-12 and np.isfinite(gc + eta * gk).all()
    toy = m.s.read(SYNTHETIC / 'checks.json')
    assert toy['passed'] and toy['sources'] == m.sources()
    m.engine.verify_sources(toy['sources'], toy['git_commit'])
    parents = {}
    for key in m.CASES:
        last = torch.load(m.BASE / key / 'last.pt', map_location='cpu', weights_only=False)
        assert last['epoch'] == 50 and last['steps'] == 9850 and (last['scheduler']['last_epoch'] == 50) and all((float(v['step']) == 9850 for v in last['optimizer']['state'].values()))
        parents[key] = dict(raw=m.s.digest(last['model']), ema=m.s.digest(last['ema']), moments=m.moments(last['optimizer']))
    out.mkdir()
    m.s.save(out / 'checks.json', dict(m.identity(), passed=True, analytic_temperature_KD_gradient_maxabs=error, parent_states=parents, frozen_controls=controls, teacher_audit_sha256=cfg['teacher_science_sha256'], model_objects=0, model_forward_calls=0, optical_model_objects=0, optical_forward_calls=0, real_optimizer_steps=0, synthetic_proof_sha256=m.s.sha(SYNTHETIC / 'checks.json'), TEST_images_read=0), exclusive=True)

def check_synthetic(out):
    assert out == SYNTHETIC and (not out.exists()) and (os.environ.get('CUDA_VISIBLE_DEVICES') == '')
    m.base.slots()
    m.claim(out.with_name(out.name + '.claim.json'), 'synthetic_balanced_gradient_and_resumed_Adam_algebra')
    torch.set_num_threads(1)
    toy = synthetic_update(m.config())
    out.mkdir()
    m.s.save(out / 'checks.json', dict(m.identity(), passed=True, **toy, optical_model_objects=0, optical_forward_calls=0, real_optimizer_steps=0, TEST_images_read=0), exclusive=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--synthetic', action='store_true')
    a = p.parse_args()
    try:
        check_synthetic(a.out) if a.synthetic else check(a.out)
    except BaseException as e:
        mark = a.out / 'failure.json' if a.out.exists() else a.out.with_name(a.out.name + '.failure.json')
        if not mark.exists():
            m.s.save(mark, dict(error=repr(e), automatic_retry=False), exclusive=True)
        raise
