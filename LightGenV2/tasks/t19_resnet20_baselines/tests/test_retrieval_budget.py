"""Offline CPU checks of budget/cache evidence; no official data or download."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from LightGenV2.tasks.t19_resnet20_baselines import evaluate_retrieval_budget as budget
from LightGenV2.tasks.t19_resnet20_baselines import frozen_visual_models
from LightGenV2.tasks.t19_resnet20_baselines.frozen_resnet18 import encoder_state_sha256
from LightGenV2.tasks.t19_resnet20_baselines.models import tokenize_text


class TinyFrozenEncoder(nn.Module):
    """Same 512-D frozen interface, including persistent BN buffers."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 512)
        self.bn = nn.BatchNorm1d(512)
        self.requires_grad_(False)
        self.train(False)
        self.pretrained_provenance = {
            'checkpoint_sha256': budget.OFFICIAL_CHECKPOINT_SHA256,
            'encoder_state_sha256': encoder_state_sha256(self), 'offline_fixture': True}

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, images):
        return self.bn(self.linear(images))


class TinyImages:
    def __init__(self, split, length=6):
        self.rows = [{'sample_id': f'{split}_{i}', 'product_id': 'a' if i % 2 else 'b'}
                     for i in range(length)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return {'image': torch.tensor([index / 9., index % 2, .25, -.5]),
                'sample_id': row['sample_id'], 'sku': row['product_id']}


@pytest.fixture
def offline(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(budget, 'FrozenResNet18', TinyFrozenEncoder)
    monkeypatch.setattr(frozen_visual_models, 'FrozenResNet18', TinyFrozenEncoder)
    monkeypatch.setattr(budget, 'git', lambda *args: 'published_offline_fixture')
    monkeypatch.setattr(budget, 'source_identity', lambda: {'offline.py': 'fixture_source'})
    # The existing atomic-rename helpers are shared, already tested code.
    # Windows' sandbox temporary directory denies replace(); keep this suite
    # focused on the new cache, budget and last-reload logic with direct writes.
    monkeypatch.setattr(budget, 'save_checkpoint', lambda path, value: torch.save(value, path))
    monkeypatch.setattr(budget, 'write_json', lambda path, value: path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8'))
    # Full integration below uses ordinary CPU index loading to stay quick;
    # the separate sampler test checks the real two-worker implementation.
    monkeypatch.setattr(budget, 'training_index_loader', lambda n: DataLoader(
        range(n), batch_size=32, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(42)))
    yield
    torch.set_num_threads(previous)


def make_cache(tmp_path, model, split, length=6):
    data = TinyImages(split, length)
    expected = [row['sample_id'] for row in data.rows]
    output = tmp_path / ('cache_' + split)
    cache, receipt = budget.cache_features(model.encoder, data, expected, split=split,
        data_identity={'fixture': True}, output=output)
    restored, again = budget.load_feature_cache(output, data_identity={'fixture': True},
        encoder_hash=encoder_state_sha256(model.encoder))
    assert torch.equal(restored['features'], cache['features'])
    assert not torch.is_inference(cache['features'])
    assert not torch.is_inference(restored['features'])
    assert again == receipt
    assert receipt['all_stored_tensor_values_roundtrip_exactly_verified']
    assert receipt['task_head_forward_count'] == 0
    assert receipt['test_metrics_observed'] is False
    expected_recomputed = length if length <= 32 else 32 + ((length - 1) % 32 + 1)
    assert receipt['encoder_recomputed_samples'] == expected_recomputed
    return restored, receipt, output


@pytest.mark.parametrize('task', budget.TASKS)
def test_actual_cache_to_independent_fixed_budgets(tmp_path, offline, monkeypatch, task):
    base = budget.build_budget_model(task)
    initial = encoder_state_sha256(base)
    initial_encoder = encoder_state_sha256(base.encoder)
    train, train_receipt, _ = make_cache(tmp_path, base, 'train')
    test, test_receipt, _ = make_cache(tmp_path, base, 'test')
    receipts = {'train': train_receipt, 'test': test_receipt}
    tokens, skus = tokenize_text(['alpha', 'beta']), ['a', 'b']
    original_evaluate = budget.evaluate_cached
    observations = []

    def observed_evaluate(model, *args, **kwargs):
        observations.append(encoder_state_sha256(model))
        return original_evaluate(model, *args, **kwargs)

    monkeypatch.setattr(budget, 'evaluate_cached', observed_evaluate)
    iterations = (0,) if task == budget.TASKS[0] else budget.BUDGETS
    before_test = train['features'].clone()
    for epochs in iterations:
        run_dir = tmp_path / f'{task}_{epochs}'
        result = budget.run_budget(task, epochs, run_dir, train, test, receipts,
            {'fixture': True}, tokens=tokens, title_skus=skus)
        manifest = json.loads((run_dir / 'run_manifest.json').read_text())
        history = json.loads((run_dir / 'history.json').read_text())
        predictions = json.loads((run_dir / 'fixed_weight_predictions.json').read_text())
        last = torch.load(run_dir / 'last_checkpoint.pt', weights_only=False)
        assert result['status'] == manifest['status'] == 'complete'
        assert last['epoch'] == result['selected_epoch'] == epochs
        assert len(history) == epochs
        assert manifest['model_state_sha256_before'] == initial
        assert result['selection_uses_test'] is False
        assert result['task_test_evaluations'] == manifest['task_test_evaluations'] == 1
        assert len(manifest['frozen_encoder_audits']) == epochs
        assert manifest['encoder_state_sha256_before'] == initial_encoder
        assert manifest['fixed_last_reload']['strict_model_reload_verified']
        assert result['checkpoint_sha256'] == budget.file_sha256(run_dir / 'last_checkpoint.pt')
        assert manifest['fixed_last_reload']['checkpoint_sha256'] == result['checkpoint_sha256']
        assert len(set(predictions['query_ids'])) == predictions['queries']
        assert len(set(predictions['candidate_ids'])) == predictions['candidates']
        assert not (run_dir / 'best_checkpoint.pt').exists()
        assert not (run_dir / 'best_predictions.json').exists()
        for state in manifest['frozen_encoder_audits'] + [manifest['frozen_encoder_final_audit']]:
            assert state['encoder_unchanged']
            assert not state['encoder_parameter_gradients_present']
            assert state['encoder_all_modules_eval']
        if epochs:
            assert manifest['model_state_sha256_after'] != initial
            assert manifest['text_state_sha256_after'] != manifest['text_state_sha256_before']
            assert manifest['optimizer_parameters'] == 64_096
            assert all(not name.startswith('encoder.') for name in manifest['optimizer_parameter_names'])
            assert manifest['optimizer_steps'] == [epochs]
            orders = json.loads((run_dir / 'sampler_orders.json').read_text())
            assert all(sorted(item['indices']) == list(range(6)) for item in orders)
            restored = budget.build_budget_model(task)
            restored.load_state_dict(last['model'], strict=True)
            assert not torch.equal(base.projection.weight, restored.projection.weight)
            assert not torch.equal(base.text_projection.weight, restored.text_projection.weight)
            assert not torch.equal(base.text_encoder.gru.weight_ih_l0, restored.text_encoder.gru.weight_ih_l0)
            assert encoder_state_sha256(restored.encoder) == initial_encoder
        else:
            assert manifest['model_state_sha256_after'] == initial
            assert manifest['optimizer_steps'] == []
            if task != budget.TASKS[0]:
                assert 'not_usable_zero_shot' in result['zero_epoch_interpretation']
    assert len(observations) == len(iterations)
    assert torch.equal(train['features'], before_test)


def test_cache_provenance_and_tensor_tampering_rejected(tmp_path, offline):
    model = budget.build_budget_model(budget.TASKS[0])
    cache, receipt, output = make_cache(tmp_path, model, 'train', 67)
    assert receipt['encoder_recomputed_samples'] == 35  # first32 + tail3
    with pytest.raises(RuntimeError, match='provenance'):
        budget.load_feature_cache(output, data_identity={'fixture': 'changed'},
            encoder_hash=encoder_state_sha256(model.encoder))
    cache['features'][0, 0] += 1
    torch.save(cache, output / 'features.pt')
    with pytest.raises(RuntimeError, match='provenance'):
        budget.load_feature_cache(output, data_identity={'fixture': True},
            encoder_hash=encoder_state_sha256(model.encoder))
    with pytest.raises(FileExistsError):
        budget.cache_features(model.encoder, TinyImages('train'), [], split='train',
            data_identity={'fixture': True}, output=output)


def test_zero_i2i_uses_raw_512_with_no_projection(tmp_path, offline):
    model = budget.build_budget_model(budget.TASKS[0])
    assert not hasattr(model, 'projection')
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 0
    pixels = torch.stack([TinyImages('x')[i]['image'] for i in range(2)])
    assert torch.equal(model(pixels), torch.nn.functional.normalize(model.encoder(pixels), dim=-1))


def test_real_two_persistent_worker_sampling_matches_original_image_loader():
    # The dataset's item contents do not affect the RandomSampler's generator.
    # Preserve spawn, worker count and persistent iterator lifecycle for epochs.
    actual = budget.training_index_loader(67)
    original = DataLoader(TensorDataset(torch.arange(67)), batch_size=32, shuffle=True,
        generator=torch.Generator().manual_seed(42), num_workers=2,
        multiprocessing_context='spawn', persistent_workers=True, pin_memory=True)
    try:
        for _ in range(3):
            actual_order = [i for batch in actual for i in batch.tolist()]
            original_order = [i for batch in original for i in batch[0].tolist()]
            assert actual_order == original_order
            assert sorted(actual_order) == list(range(67))
    finally:
        budget.shutdown_loader(actual)
        budget.shutdown_loader(original)


def test_published_protocol_keeps_all_seven_and_rejects_changes(monkeypatch):
    jobs = budget.validate_profile(budget.TASKS)
    assert len(jobs) == 7
    assert [x['epochs'] for x in jobs] == [0, 0, 1, 5, 0, 1, 5]
    original = budget.profile_identity()
    invalid = copy.deepcopy(original)
    invalid['configuration']['sampler_workers'] = 0
    monkeypatch.setattr(budget, 'profile_identity', lambda: invalid)
    with pytest.raises(ValueError, match='fixed protocol'):
        budget.validate_profile(budget.TASKS)


def test_suite_real_cache_path_retains_all_conditions(tmp_path, offline, monkeypatch):
    class FakeEasy:
        def __init__(self, root):
            self.identity = {'fixture': True}
            self.title_skus = ['a', 'b']
            self.title_tokens = tokenize_text(['alpha', 'beta'])

        def images(self, split):
            return TinyImages(split)

    profile = budget.profile_identity()
    monkeypatch.setattr(budget, 'TASK', tmp_path)
    monkeypatch.setattr(budget, 'profile_identity', lambda: profile)
    monkeypatch.setattr(budget, 'require_published_source', lambda: None)
    monkeypatch.setattr(budget, 'Easy100', FakeEasy)
    monkeypatch.setattr(budget, 'abo_data', lambda: (TinyImages('train'), TinyImages('test'), {'fixture': True}))
    original_load = budget.load_feature_cache
    loads = []

    def observed_load(*args, **kwargs):
        cache, receipt = original_load(*args, **kwargs)
        loads.append(receipt['split'])
        assert not torch.is_inference(cache['features'])
        return cache, receipt

    monkeypatch.setattr(budget, 'load_feature_cache', observed_load)
    args = SimpleNamespace(device='cpu', threads=1, tasks=list(budget.TASKS),
        suite_dir=tmp_path / 'runs' / 'smoke' / 'budget_fixture', data_root=tmp_path)
    results = budget.run(args)
    protocol = json.loads((args.suite_dir / 'protocol.json').read_text())
    summary = json.loads((args.suite_dir / 'results.json').read_text())
    assert len(results) == len(protocol['conditions']) == 7
    assert loads == ['train', 'test', 'train', 'test']
    assert summary['all_conditions_complete'] and not summary['selection_performed']
    assert all(item['task_test_evaluations'] == 1 for item in results)
    assert protocol['execution_profile']['sha256'] == profile['sha256']
    with pytest.raises(FileExistsError):
        budget.run(args)
