"""Offline checks of exact sampler/AdamW continuation and frozen FP32 caching."""
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from LightGenV2.tasks.t19_resnet20_baselines import continue_lgvq50 as continuation
from LightGenV2.tasks.t19_resnet20_baselines.models import QualityModel


class TinyFrozenEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 512)
        self.bn = nn.BatchNorm1d(512)
        self.requires_grad_(False)
        self.train(False)
        self.pretrained_provenance = {'offline_fixture': True}

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, frames):
        return self.bn(self.linear(frames.mean((-1, -2))))


class TinyQuality(QualityModel):
    def __init__(self, target):
        nn.Module.__init__(self)
        self.target = target
        self.feature_dim = 512
        self.encoder = TinyFrozenEncoder()
        self.head = nn.Sequential(nn.Linear(512 * (1 if target == 'spatial' else 7), 64),
                                  nn.ReLU(), nn.Linear(64, 5))
        self.register_buffer('quality_levels', torch.arange(1., 6.))


class ToyVideos(Dataset):
    def __init__(self, count, split):
        self.count, self.split = count, split

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        return {'image': torch.arange(4 * 3 * 4 * 4).reshape(4, 3, 4, 4).float() / 100 + index / 20,
                'score': torch.tensor(float(index % 5 + 1)), 'sample_id': f'{self.split}_{index}'}


class ToyProblem:
    def __init__(self, task):
        self.train = ToyVideos(17, 'train')
        self.test = ToyVideos(5, 'test')
        self.train_mos = torch.tensor([self.train[i]['score'] for i in range(len(self.train))])
        self.identity = {'offline_static_LGVQ_fixture': True}

    def configure_model(self, model, device):
        self.boundaries = model.set_train_score_range(self.train_mos).to(device)


def reference_epoch(model, optimizer, loader, problem):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    order = []
    for step, indices in enumerate(loader):
        rows = [problem.train[int(i)] for i in indices]
        images = torch.stack([row['image'] for row in rows])
        scores = torch.stack([row['score'] for row in rows])
        group_samples = min(32, len(problem.train) - (step // 4) * 32)
        logits = model(images)['quality_logits']
        loss = F.cross_entropy(logits, torch.bucketize(scores, problem.boundaries, right=False))
        (loss * len(indices) / group_samples).backward()
        if (step + 1) % 4 == 0 or step + 1 == len(loader):
            torch.nn.utils.clip_grad_norm_(model.head.parameters(), 1.)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        order.extend(indices.tolist())
    return order


def test_replay_matches_actual_two_worker_persistent_loader_through_epoch34():
    generator = torch.Generator().manual_seed(42)
    original = DataLoader(continuation.IndexDataset(19), batch_size=8, shuffle=True,
        generator=generator, num_workers=2, multiprocessing_context='spawn',
        persistent_workers=True, pin_memory=True)
    replayed = None
    try:
        previous = [torch.cat(list(original)) for _ in range(30)]
        replayed, audit = continuation.replay_index_loader(19, 8, 30, seed=42, num_workers=2)
        assert [entry['sample_order_sha256'] for entry in audit['replayed_epoch_orders']] == [
            continuation.structured_sha256(order) for order in previous]
        assert torch.equal(original.generator.get_state(), replayed.generator.get_state())
        for _ in range(4):
            assert torch.equal(torch.cat(list(original)), torch.cat(list(replayed)))
        assert audit['persistent_workers'] is True
        assert audit['loader_retained_for_continuation_epoch'] == 31
        # A newly seed42-created loader gives epoch1, not a legitimate epoch31.
        fresh, _ = continuation.replay_index_loader(19, 8, 0, seed=42, num_workers=0)
        assert not torch.equal(torch.cat(list(original)), torch.cat(list(fresh)))
    finally:
        continuation.shutdown_loader(original)
        continuation.shutdown_loader(replayed)


@pytest.mark.parametrize('target', ['temporal', 'spatial'])
def test_cached_fp32_features_match_every_direct_batch_and_singleton(target):
    torch.manual_seed(42)
    model = TinyQuality(target)
    frozen_before = continuation.encoder_state_sha256(model.encoder)
    cache, audit = continuation.cache_dataset(model, ToyVideos(17, 'train'), 'cpu', 'train', workers=0)
    assert cache['features'].shape == (17, 4, 512)
    assert cache['features'].dtype == torch.float32
    assert audit['samples'] == 17 and audit['all_samples_checked_against_direct_forward']
    assert audit['max_abs_logits_error'] == audit['max_abs_score_error'] == 0.
    assert audit['singleton_samples_checked'] == 1
    assert audit['cache_binding_sha256'] == continuation.structured_sha256(cache)
    assert continuation.encoder_state_sha256(model.encoder) == frozen_before
    damaged = {**cache, 'sample_ids': list(reversed(cache['sample_ids']))}
    assert continuation.structured_sha256(damaged) != audit['cache_binding_sha256']


class CpuTorch:
    cuda = SimpleNamespace(get_device_name=lambda: 'CPU offline fixture', empty_cache=lambda: None)

    def __getattr__(self, name):
        return getattr(torch, name)

    @staticmethod
    def device(value):
        assert value == 'cuda:0'
        return torch.device('cpu')


@pytest.mark.parametrize('target', ['temporal', 'spatial'])
def test_run_continues_original_optimizer_and_sampler_to_exact_epoch50(tmp_path, monkeypatch, target):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    reference = TinyQuality(target)
    problem = ToyProblem('lgvq_' + target)
    problem.configure_model(reference, 'cpu')
    optimizer = torch.optim.AdamW(reference.head.parameters(), lr=.001, weight_decay=.0001)
    reference_loader, _ = continuation.replay_index_loader(17, 8, 0, seed=42, num_workers=2)
    parent_run = tmp_path / 'runs/simulation' / ('parent_' + target)
    parent_run.mkdir(parents=True)
    try:
        for _ in range(30):
            reference_epoch(reference, optimizer, reference_loader, problem)
        metadata = {'task': 'lgvq_' + target, 'backbone': 'resnet18_imagenet_frozen',
            'seed': 42, 'epochs': 30, 'training_scope': 'decoder_only', 'learning_rate': .001,
            'schedule': 'constant', 'warmup_epochs': 0, 'effective_batch': 32, 'microbatch': 8,
            'torch': str(torch.__version__), 'status': 'complete', 'data_identity': problem.identity,
            'encoder_state_sha256_before': continuation.encoder_state_sha256(reference.encoder)}
        source = parent_run / 'last_checkpoint.pt'
        torch.save({'model': reference.state_dict(), 'optimizer': optimizer.state_dict(),
                    'epoch': 30, 'manifest': metadata}, source)
        source_sha = continuation.file_sha256(source)
        (parent_run / 'run_manifest.json').write_text(json.dumps(metadata), encoding='utf-8')
        (parent_run / 'result.json').write_text(json.dumps({'status': 'complete', 'selected_epoch': 30,
            'selection': 'fixed_last_epoch', 'checkpoint_sha256': source_sha}), encoding='utf-8')
        expected_orders = [reference_epoch(reference, optimizer, reference_loader, problem) for _ in range(20)]
        new_model = TinyQuality(target)
        monkeypatch.setattr(continuation, 'TASK', tmp_path)
        monkeypatch.setattr(continuation, 'EXPECTED_COUNTS', {'train': 17, 'test': 5})
        monkeypatch.setattr(continuation, 'require_published_source', lambda: None)
        monkeypatch.setattr(continuation, 'verify_official_cache', lambda: None)
        monkeypatch.setattr(continuation, 'wait_for_gpu', lambda args: [])
        monkeypatch.setattr(continuation, 'git', lambda *args: 'offline_main_commit')
        monkeypatch.setattr(continuation, 'Problem', ToyProblem)
        monkeypatch.setattr(continuation, 'build_model', lambda *args, **kwargs: new_model)
        monkeypatch.setattr(continuation, 'torch', CpuTorch())
        cache_original = continuation.cache_dataset
        monkeypatch.setattr(continuation, 'cache_dataset', lambda model, data, device, split:
            cache_original(model, data, device, split, workers=0))
        monkeypatch.setattr(continuation, 'write_json', lambda path, value: path.write_text(
            json.dumps(value, indent=2), encoding='utf-8'))
        monkeypatch.setattr(continuation, 'save_checkpoint', lambda path, value: torch.save(value, path))
        args = SimpleNamespace(source_run=parent_run, target=target,
            run_dir=tmp_path / 'runs/simulation' / ('continued_' + target), gpu_uuid='GPU-fixture',
            queue_status_path=None, shared_owner_pids=[], max_physical_gpus=4,
            max_jobs_per_gpu=3, smoke_batches=0)
        continuation.run(args)
        assert continuation.file_sha256(source) == source_sha
        result = json.loads((args.run_dir / 'result.json').read_text())
        manifest = json.loads((args.run_dir / 'run_manifest.json').read_text())
        history = json.loads((args.run_dir / 'history.json').read_text())
        checkpoint = torch.load(args.run_dir / 'last_checkpoint.pt', map_location='cpu', weights_only=False)
        assert result['status'] == manifest['status'] == 'complete'
        assert result['selected_epoch'] == checkpoint['epoch'] == 50
        assert result['formal_result'] is True and result['partial_epoch'] is False
        assert result['prediction_count'] == 5
        assert result['optimizer_state_restored'] is True
        assert manifest['parent_checkpoint_unchanged'] is True
        assert manifest['encoder_state_sha256_before'] == manifest['encoder_state_sha256_after']
        assert manifest['parent_optimizer_steps'] == 30
        assert len(manifest['frozen_encoder_audits']) == 20
        assert [row['epoch'] for row in history] == list(range(31, 51))
        assert [row['epoch'] for row in history if 'srcc' in row] == [35, 40, 45, 50]
        assert [row['sample_order_sha256'] for row in history] == [
            continuation.structured_sha256(torch.tensor(order)) for order in expected_orders]
        for name, value in reference.state_dict().items():
            # Reusing a per-sample FP32 feature from another frozen-encoder
            # batch size preserves the graph, but is not a bitwise BLAS promise.
            torch.testing.assert_close(checkpoint['model'][name], value, rtol=1e-5, atol=1e-5)
        expected_optimizer = optimizer.state_dict()
        assert checkpoint['optimizer']['param_groups'] == expected_optimizer['param_groups']
        for index, state in expected_optimizer['state'].items():
            assert checkpoint['optimizer']['state'][index]['step'] == state['step'] == 50
            for name in ('exp_avg', 'exp_avg_sq'):
                torch.testing.assert_close(checkpoint['optimizer']['state'][index][name], state[name],
                                           rtol=1e-4, atol=1e-6)
        assert set(path.name for path in args.run_dir.glob('*.pt')) == {'best_checkpoint.pt', 'last_checkpoint.pt'}
    finally:
        continuation.shutdown_loader(reference_loader)
        torch.set_num_threads(previous_threads)


def test_optimizer_restore_rejects_new_or_incomplete_state(monkeypatch):
    monkeypatch.setattr(continuation, 'EXPECTED_COUNTS', {'train': 17, 'test': 5})
    model = TinyQuality('spatial')
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=.001, weight_decay=.0001)
    with pytest.raises(ValueError, match='missing trained head'):
        continuation.restore_optimizer(model, optimizer.state_dict())
