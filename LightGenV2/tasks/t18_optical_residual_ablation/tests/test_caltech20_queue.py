"""CPU scheduling integration: four slots, adopt L2, stop dispatch on failure."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import caltech20_fit_queue as q


def run(tmp_path, monkeypatch, fail=False):
    data = tmp_path / 'data.npz'; data.write_bytes(b'fixture')
    plan = tmp_path / 'plan.json'; plan.write_text('{}')
    out = tmp_path / 'suite'
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setattr(sys, 'argv', ['queue', '--out', str(out), '--data', str(data),
        '--l2', str(tmp_path / 'completedL2'), '--probe', str(plan)])
    adopted = {k: str(tmp_path / k) for k in ('L2_rho0', 'L2_rho03')}
    monkeypatch.setattr(q, 'prepare', lambda a: (adopted.copy(), {}, {}, {}, {4: 4, 8: 1, 16: 1}))
    monkeypatch.setattr(q, 'idle', lambda gpu: True)
    monkeypatch.setattr(q, 'verify_sources', lambda s: None)
    monkeypatch.setattr(q.time, 'sleep', lambda s: None)
    monkeypatch.setattr(q.subprocess, 'check_output', lambda *a, **kw: 'commit\n')
    running, commands = set(), []
    class Process:
        def __init__(self, args, env, **kwargs):
            gpu = env['CUDA_VISIBLE_DEVICES']
            assert gpu in q.GPUS and gpu not in running
            running.add(gpu); assert len(running) <= 4
            assert args[args.index('--epochs') + 1] == '100'
            assert args[args.index('--depth') + 1] in ('4', '8', '16')
            assert '--full-budget' in args and '--test' not in args
            depth = int(args[args.index('--depth') + 1])
            assert int(args[args.index('--micro') + 1]) == (4 if depth == 4 else 1)
            arm = Path(args[args.index('--out') + 1]); arm.mkdir()
            q.save(arm / 'fit_result.json', dict(epochs_completed=100, stop_reason='maximum_epochs',
                sources={}, test_evaluations=0))
            self.gpu, self.pid, self.done = gpu, 900000 + len(commands), False
            self.code = 1 if fail and not commands else 0
            commands.append(args)
        def poll(self):
            if not self.done: running.remove(self.gpu); self.done = True
            return self.code
    monkeypatch.setattr(q.subprocess, 'Popen', Process)
    if fail:
        with pytest.raises(RuntimeError, match='No automatic retry'): q.main()
    else: q.main()
    assert not running
    return out, commands, adopted


def test_four_slots_adopt_completed_pair_without_retraining(tmp_path, monkeypatch):
    out, commands, adopted = run(tmp_path, monkeypatch)
    assert len(commands) == 6
    arms = q.read(out / 'fit_manifest.json')['arms']
    assert set(arms) == set(q.KEYS)
    assert all(arms[k] == v for k, v in adopted.items())
    assert q.read(out / 'status.json')['test_evaluations'] == 0
    assert q.read(out / 'current_processes.json')['active'] == []


def test_failure_drains_healthy_children_without_dispatch_or_retry(tmp_path, monkeypatch):
    out, commands, _ = run(tmp_path, monkeypatch, fail=True)
    assert len(commands) == 4
    status = q.read(out / 'status.json')
    assert status['state'] == 'fit_failed_no_automatic_retry'
    assert len(status['completed']) == 3 and len(status['failures']) == 1
    assert len(status['pending']) == 2 and status['active'] == 0
    assert not (out / 'fit_manifest.json').exists()
