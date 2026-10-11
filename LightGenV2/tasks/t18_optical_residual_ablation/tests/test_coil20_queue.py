"""CPU integration: bounded three-GPU scheduling and no adoption of tested fits."""
import json
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import coil20_fit_queue as queue


def fixture(tmp_path):
    data = tmp_path/'data.npz'; data.write_bytes(b'fixture')
    queue.save(tmp_path/'data_manifest.json', {'cache_sha256':queue.sha(data)})
    l2 = tmp_path/'L2'; l2.mkdir()
    for name in ['rho0', 'rho03']:
        arm = l2/name; arm.mkdir(); (arm/'best_checkpoint.pt').write_bytes(b'saved')
        queue.save(arm/'fit_result.json', dict(epochs_completed=30, test_evaluations=0,
            data_sha256=queue.sha(data), checkpoint_sha256=queue.sha(arm/'best_checkpoint.pt'),
            accuracy_gate_passed=True, speed_gate_passed=True))
    timing = tmp_path/'timing'; timing.mkdir()
    queue.save(timing/'fit_result.json', dict(epochs_completed=2, depth=16, test_evaluations=0))
    queue.save(timing/'history.json', [{'epoch_seconds':150}, {'epoch_seconds':100}])
    probe = tmp_path/'probe.json'
    queue.save(probe, dict(results=[dict(depth=d, rho=r, microbatch=1 if d==16 else 4,
        expert_batched=True, passed=True) for d in [4,8,16] for r in [0.,.3]]))
    return data, l2, timing, probe


def test_three_slots_and_no_test_worker(tmp_path, monkeypatch):
    data, l2, timing, probe = fixture(tmp_path)
    out = tmp_path/'suite'; running, jobs = set(), []
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(sys, 'argv', ['queue','--out',str(out),'--data',str(data),
        '--l2',str(l2),'--probe',str(probe),'--timing',str(timing)])
    monkeypatch.setattr(queue.time, 'sleep', lambda seconds: None)
    monkeypatch.setattr(queue.subprocess, 'check_output', lambda *args, **kwargs: 'fixture_commit\n')
    class Process:
        def __init__(self, args, env, **kwargs):
            gpu = env['CUDA_VISIBLE_DEVICES']; assert gpu not in running
            running.add(gpu); assert len(running) <= 3
            assert '--full-budget' in args and '--test' not in args
            arm = Path(args[args.index('--out')+1]); arm.mkdir()
            queue.save(arm/'fit_result.json', dict(epochs_completed=30, test_evaluations=0))
            self.gpu, self.pid, self.returncode, self.done = gpu, 900000+len(jobs), 0, False
            jobs.append(args)
        def poll(self):
            if not self.done: running.remove(self.gpu); self.done=True
            return 0
    monkeypatch.setattr(queue.subprocess, 'Popen', Process)
    queue.main()
    assert len(jobs) == 6 and not running
    assert len(queue.read(out/'fit_manifest.json')['arms']) == 8
    assert queue.read(out/'status.json')['state'] == 'fit_complete_awaiting_validation_review'
    assert queue.read(out/'status.json')['test_evaluations'] == 0


def test_previously_tested_fit_rejected_before_launch(tmp_path, monkeypatch):
    data, l2, timing, probe = fixture(tmp_path)
    result = queue.read(l2/'rho0/fit_result.json'); result['test_evaluations']=1
    queue.save(l2/'rho0/fit_result.json', result)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(sys,'argv',['queue','--out',str(tmp_path/'suite'),'--data',str(data),
        '--l2',str(l2),'--probe',str(probe),'--timing',str(timing)])
    monkeypatch.setattr(queue.subprocess,'Popen',lambda *args,**kwargs: pytest.fail('Rejected fit must not launch a child'))
    with pytest.raises(AssertionError): queue.main()
