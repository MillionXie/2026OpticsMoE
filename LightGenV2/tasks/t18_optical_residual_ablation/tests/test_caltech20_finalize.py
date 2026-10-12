"""No optical inference: selection ties, full budget, and once-test barriers."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import caltech20_finalize as f
import audit_caltech20_final as audit


def arm_fixture(tmp_path):
    arm = tmp_path / 'arm'; arm.mkdir()
    history = [dict(epoch=e, val={'balanced_nll': 1. if e >= 99 else 2.}) for e in range(1, 101)]
    cfg = dict(requested_epochs=100, early_stopping=False, microbatch=4, evaluation_batch_size=4)
    f.save(arm / 'history.json', history); f.save(arm / 'config.json', cfg)
    f.save(arm / 'metadata.json', dict(sources={}, git_commit='fixture'))
    (arm / 'best_checkpoint.pt').write_bytes(b'checkpoint')
    fit = dict(test_evaluations=0, test_used_for_selection=False, selection_split='val', candidate='base',
        epochs_completed=100, config=cfg, stop_reason='maximum_epochs', selected_epoch=99,
        metrics={'val': history[98]['val']}, depth=2, sources={},
        history_sha256=f.sha(arm / 'history.json'), checkpoint_sha256=f.sha(arm / 'best_checkpoint.pt'))
    f.save(arm / 'fit_result.json', fit)
    return arm, fit


def test_first_exact_tie_and_no_accuracy_reselection(tmp_path):
    arm, fit = arm_fixture(tmp_path)
    assert f.verify_arm(arm)['selected_epoch'] == 99
    fit['selected_epoch'] = 100; f.save(arm / 'fit_result.json', fit)
    with pytest.raises(AssertionError): f.verify_arm(arm)


def test_partial_epoch_budget_is_not_a_formal_arm(tmp_path):
    arm, fit = arm_fixture(tmp_path)
    fit['epochs_completed'] = 2; f.save(arm / 'fit_result.json', fit)
    with pytest.raises(AssertionError): f.verify_arm(arm)


def test_incomplete_lock_rejected_before_data_access(tmp_path):
    f.save(tmp_path / 'global_selection_lock.json', dict(entries={'L2_rho0': {}}))
    with pytest.raises(AssertionError, match='All eight'):
        f.verify_global(tmp_path, tmp_path / 'missing.npz')


def test_exclusive_durable_claim_survives_failure(tmp_path):
    f.claim_test(tmp_path, 'L16_rho03', 'locked-sha')
    with pytest.raises(FileExistsError): f.claim_test(tmp_path, 'L16_rho03', 'locked-sha')
    assert f.read(tmp_path / 'L16_rho03/test_started.json')['global_selection_lock_sha256'] == 'locked-sha'


def test_worker_claims_before_first_test_materialization(tmp_path, monkeypatch):
    key = 'L16_rho03'; gpu = f.GPUS[0]; data = tmp_path / 'data.npz'
    arm = tmp_path / 'arm'; arm.mkdir(); (arm / 'best_checkpoint.pt').write_bytes(b'checkpoint')
    f.save(tmp_path / 'global_selection_lock.json', {'fixture': True})
    f.save(tmp_path / 'data_manifest.json', {'fixture': True})
    cfg = dict(expert_batched=True, evaluation_batch_size=1)
    entry = dict(arm=str(arm), depth=16, rho=.3, selected_epoch=99, checkpoint_sha256='best')
    lock = dict(entries={key: entry}, data_sha256='data', scope='caltech20')
    fit = dict(sources={}, config=cfg, metrics={}, parameters=3528784)
    checkpoint = dict(selected_state_kind='ema', selection_split='val', epoch=99, config=cfg,
        sources={}, data_sha256='data', model={})
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', gpu)
    monkeypatch.setattr(f, 'verify_global', lambda *a: lock)
    monkeypatch.setattr(f, 'verify_arm', lambda *a: fit)
    model = NS(load_state_dict=lambda *a, **k: None, eval=lambda: None)
    pilot = NS(execution=NS(build=lambda *a, **kw: model),
        torch=NS(set_num_threads=lambda n: None, load=lambda *a, **kw: checkpoint),
        o=NS(f=NS(gpu_budget=NS(ALLOWED=set(), configure=lambda *a, **kw: {}))))
    monkeypatch.setitem(sys.modules, 'medicinal_pilot', pilot)
    # Stop at the first attempted materialization, simulating failure after claim.
    def fail_load(*args):
        assert (tmp_path / key / 'test_started.json').exists()
        raise RuntimeError('simulated materialization failure')
    monkeypatch.setattr(f, 'load_test_arrays', fail_load)
    with pytest.raises(RuntimeError, match='simulated materialization failure'):
        f.worker(NS(out=tmp_path, data=data, key=key, gpu=gpu))
    with pytest.raises(AssertionError, match='No test retry'):
        f.worker(NS(out=tmp_path, data=data, key=key, gpu=gpu))


def test_cpu_final_audit_waits_for_all_tests(tmp_path):
    f.save(tmp_path / 'status.json', dict(state='selection_locked_test_not_started'))
    with pytest.raises(AssertionError): audit.completion_barrier(tmp_path)


def test_test_loader_uses_manifest_size_and_supports(tmp_path, monkeypatch):
    import numpy as np
    # Synthetic data only, no experiment test pixels/labels are touched.
    images = np.zeros((7, 150, 150, 3), dtype=np.uint8)
    labels = np.array([0, 0, 0, 1, 1, 1, 1])
    ids = np.array([f'fixture_{i}' for i in range(7)])
    path = tmp_path / 'fixture.npz'
    np.savez(path, test_images=images, test_labels=labels, test_ids=ids)
    fake_tensor = NS(long=lambda: 'labels')
    monkeypatch.setitem(sys.modules, 'torch', NS(from_numpy=lambda a: fake_tensor))
    manifest = dict(classes=['class_a', 'class_b'], split_ids={'test': ids.tolist()}, supports={'test': [3, 4]})
    assert f.load_test_arrays(path, manifest)[2].tolist() == ids.tolist()
    manifest['supports']['test'] = [4, 3]
    with pytest.raises(AssertionError): f.load_test_arrays(path, manifest)
