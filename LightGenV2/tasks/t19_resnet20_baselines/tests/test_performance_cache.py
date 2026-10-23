"""Lossless caching, deduplication, interruption recovery and worker contracts."""
import json
import pickle

import pytest
import torch

from LightGenV2.tasks.t19_resnet20_baselines import performance_cache as cache


class DeterministicEditor:
    def __init__(self, count=6, fail_at=None):
        self.count, self.fail_at = count, fail_at

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if self.fail_at is not None and index == self.fail_at:
            raise RuntimeError('intentional renderer interruption')
        # Include every possible byte value, with references shared by pairs.
        reference = (torch.arange(3 * 16 * 16).reshape(3, 16, 16) + index // 2).remainder(256).byte()
        target = (reference.to(torch.int64) + index).remainder(256).byte()
        tokens = torch.tensor([1, 68 + index, 2, 0], dtype=torch.int64)
        return cache._restore_row(reference, target, tokens,
                                  {'sample_id': f'source{index//2}:mode:{index}', 'mode': 'joint'})


def build(dataset, path, **options):
    return cache.build_editor_cache(dataset, path, {'split': 'train', 'manifest_sha': 'fixture'},
                                    source_files=[__file__], workers=0, checkpoint_interval=2, **options)


def test_uint8_cache_exact_all_bytes_deduplicated_readonly_and_no_per_item_hash(tmp_path, monkeypatch):
    source = DeterministicEditor()
    manifest = build(source, tmp_path)
    assert manifest['status'] == 'complete' and manifest['all_rows_exactly_verified']
    assert manifest['stores']['reference']['count'] == 3
    assert manifest['dataset_length'] == manifest['completed_samples'] == 6
    cached = cache.CachedEditorPairs(tmp_path, verify_integrity=True)
    # All identity hashes are startup/build operations, never fetch operations.
    monkeypatch.setattr(cache, '_file_sha', lambda path: pytest.fail('getitem must not hash files'))
    monkeypatch.setattr(cache, '_tensor_sha', lambda value: pytest.fail('getitem must not hash tensors'))
    for index in range(len(source)):
        cache._assert_equal_rows(source[index], cached[index], index)
    assert all(not value.flags.writeable for value in cached._maps.values())
    assert pickle.loads(pickle.dumps(cached))._maps is None


def test_partial_journal_resume_discards_only_uncommitted_tail_and_keeps_order(tmp_path):
    with pytest.raises(RuntimeError, match='intentional renderer'):
        build(DeterministicEditor(fail_at=3), tmp_path)
    partial = json.loads((tmp_path / 'manifest.json').read_text())
    assert partial['status'] == 'interrupted' and partial['completed_samples'] == 2
    assert len((tmp_path / 'index.jsonl').read_text().splitlines()) == 3
    manifest = build(DeterministicEditor(), tmp_path, resume=True)
    assert manifest['status'] == 'complete' and manifest['completed_samples'] == 6
    cached = cache.CachedEditorPairs(tmp_path)
    assert [row['index'] for row in cached.records] == list(range(6))
    assert cache.verify_editor_cache(tmp_path, DeterministicEditor(), workers=0)['checked_samples'] == 6


def test_refuse_cache_protocol_source_changes_corruption_and_incomplete_reads(tmp_path):
    source = DeterministicEditor()
    build(source, tmp_path)
    with pytest.raises(FileExistsError):
        build(source, tmp_path)
    with pytest.raises(ValueError, match='source data identity'):
        cache.CachedEditorPairs(tmp_path, source_identity={'split': 'test'})
    with pytest.raises(ValueError, match='different dataset/source'):
        cache.build_editor_cache(source, tmp_path, {'split': 'changed'}, source_files=[__file__], workers=0, resume=True)
    path = tmp_path / 'reference.bin'
    with path.open('r+b') as stream:
        stream.write(b'corrupt')
    with pytest.raises(ValueError, match='blob identity'):
        cache.verify_cache_integrity(tmp_path)
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    manifest['status'] = 'interrupted'
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='complete'):
        cache.CachedEditorPairs(tmp_path)


def test_not_all_float_inputs_are_losslessly_representable_as_renderer_uint8():
    row = DeterministicEditor()[0]
    row['image'][0, 0, 0] += .001
    with pytest.raises(ValueError, match='exactly reproduce'):
        cache._encode_row(0, row)


def test_incomplete_resume_refuses_corrupted_committed_bytes(tmp_path):
    with pytest.raises(RuntimeError, match='intentional renderer'):
        build(DeterministicEditor(fail_at=3), tmp_path)
    path = tmp_path / 'reference.bin'
    with path.open('r+b') as stream:
        stream.write(b'bad committed cache bytes')
    with pytest.raises(ValueError, match='committed blob SHA'):
        build(DeterministicEditor(), tmp_path, resume=True)


def test_parallel_cpu_build_and_full_equal_validation(tmp_path):
    source = DeterministicEditor(count=3)
    manifest = cache.build_editor_cache(source, tmp_path, {'split': 'train'},
        source_files=[__file__], workers=2, checkpoint_interval=2)
    assert manifest['completed_samples'] == 3
    assert cache.verify_editor_cache(tmp_path, source, workers=2)['all_checked_rows_exactly_equal']
    result = cache.verify_editor_cache(tmp_path, source, mode='random', sample_count=2, workers=0)
    assert result['checked_samples'] == 2 and len(set(result['indices'])) == 2


def test_ram_vectorized_batches_exact_all_bytes_random_partial_and_repeated_indices(tmp_path, monkeypatch):
    from torch.utils.data import default_collate
    source = DeterministicEditor(17)
    build(source, tmp_path)
    manifest_before = (tmp_path / 'manifest.json').read_bytes()
    cached = cache.CachedEditorPairs(tmp_path)
    with pytest.raises(RuntimeError, match='preload'):
        cached.get_batch([0])
    assert cached.preload() is cached
    assert all(value.device.type == 'cpu' for value in cached._ram.values())
    assert cached._ram['reference'].dtype == cached._ram['target'].dtype == torch.uint8
    assert cached._ram['tokens'].dtype == torch.int64
    assert cached.reader_provenance['ram_preloaded']
    assert cached.reader_provenance['reader_code_sha256'] == cache._file_sha(cache.__file__)
    assert (tmp_path / 'manifest.json').read_bytes() == manifest_before
    # References contain all 256 byte values. Repeated IDs deliberately address
    # the same deduplicated store, while output metadata keeps requested order.
    orders = [[0], [16, 2, 2], [0, 1, 0, 1, 16, 5, 6, 6],
              torch.randperm(17, generator=torch.Generator().manual_seed(42))[:8],
              list(range(17))]
    expected = [default_collate([source[int(index)] for index in order]) for order in orders]
    monkeypatch.setattr(cache, '_file_sha', lambda path: pytest.fail('RAM batch must not hash files'))
    monkeypatch.setattr(cache, '_tensor_sha', lambda value: pytest.fail('RAM batch must not hash tensors'))
    monkeypatch.setattr('builtins.open', lambda *args, **kwargs: pytest.fail('RAM batch must not open files'))
    monkeypatch.setattr(cache.Path, 'open', lambda *args, **kwargs: pytest.fail('RAM batch must not open files'))
    for order, wanted in zip(orders, expected):
        actual = cached.get_batch(order)
        for name in ('image', 'target', 'tokens'):
            assert torch.equal(actual[name], wanted[name])
        assert actual['sample_id'] == wanted['sample_id'] and actual['mode'] == wanted['mode']
        for index in order:
            cache._assert_equal_rows(source[int(index)], cached[int(index)], int(index))
    assert cached.preload() is cached  # Idempotent and also file-free.
    with pytest.raises(IndexError):
        cached.get_batch([17])
    with pytest.raises(ValueError, match='integers'):
        cached.get_batch([0.5])


def test_named_old_builder_reader_compatibility_is_narrow_and_does_not_rewrite_cache(tmp_path):
    source = DeterministicEditor()
    cache.build_editor_cache(source, tmp_path, {'split': 'train'},
        source_files=[cache.__file__, __file__], workers=0)
    path = tmp_path / 'manifest.json'
    manifest = json.loads(path.read_text())
    reader_path = str(cache.Path(cache.__file__).resolve())
    old_sha = '5521db609a053889eb7bd27b35dc9a85333bf23c91435cb4c8b12190cac7cae5'
    assert cache.COMPATIBLE_BUILDERS[old_sha] == '487beceba'
    manifest['source_code_sha256'][reader_path] = old_sha
    path.write_text(json.dumps(manifest))
    before = path.read_bytes()
    cached = cache.CachedEditorPairs(tmp_path).preload()
    assert cached.reader_provenance['accepted_builder_sha256'] == old_sha
    assert cached.reader_provenance['compatible_builder_commit'] == '487beceba'
    assert path.read_bytes() == before
    # No compatibility exception applies to any other renderer/data file.
    manifest['source_code_sha256'][str(cache.Path(__file__).resolve())] = old_sha
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='renderer/source code'):
        cache.CachedEditorPairs(tmp_path)
    manifest['source_code_sha256'][str(cache.Path(__file__).resolve())] = cache._file_sha(__file__)
    manifest['source_code_sha256'][reader_path] = 'f' * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='renderer/source code'):
        cache.CachedEditorPairs(tmp_path)


def test_preload_verifies_loaded_bytes_even_without_constructor_file_hashes(tmp_path):
    build(DeterministicEditor(), tmp_path)
    cached = cache.CachedEditorPairs(tmp_path, verify_integrity=False)
    with (tmp_path / 'target.bin').open('r+b') as stream:
        stream.write(b'corruption after constructing reader')
    with pytest.raises(ValueError, match='blob identity'):
        cached.preload()


class TinyFrozenEditorEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1)
        self.bn = torch.nn.BatchNorm2d(3)
        self.pretrained_provenance = {'offline_fixture': True}
        self.requires_grad_(False)
        self.train(False)

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, images):
        return self.bn(self.conv(images))


class TinyCachedEditor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from LightGenV2.tasks.t19_resnet20_baselines.models import ByteTextEncoder
        self.encoder = TinyFrozenEditorEncoder()
        self.text_encoder = ByteTextEncoder()
        self.text_projection = torch.nn.Linear(64, 3)
        self.rgb = torch.nn.Conv2d(3, 3, 1)

    def forward(self, images, tokens):
        text = self.text_projection(self.text_encoder(tokens))[:, :, None, None]
        return self.rgb(self.encoder(images) + text).sigmoid()


class TinyEditorProblem:
    def __init__(self, task='abo_conditional_edit'):
        from types import SimpleNamespace
        self.task = task
        self.train = DeterministicEditor(17)
        self.test = DeterministicEditor(5)
        self.final_test = DeterministicEditor(4)
        for split, dataset in (('train', self.train), ('val', self.test), ('test', self.final_test)):
            dataset.base = SimpleNamespace(split=split)
        self.identity = {'offline_deterministic_editor_identity': True}
        self.batch_size, self.batch_sampler, self.metric = 8, None, 'psnr_db'
        self.seen_training_order = []

    def configure_model(self, model, device):
        pass

    def loss(self, model, batch, device):
        self.seen_training_order.extend(int(value.rsplit(':', 1)[1]) for value in batch['sample_id'])
        predicted = model(batch['image'].to(device), batch['tokens'].to(device))
        target = batch['target'].to(device)
        loss = torch.nn.functional.mse_loss(predicted, target)
        for axis in (-1, -2):
            loss = loss + .05 * torch.nn.functional.l1_loss(
                torch.diff(predicted, dim=axis), torch.diff(target, dim=axis))
        return loss

    @torch.no_grad()
    def evaluate(self, model, loader, device):
        model.eval()
        records = []
        for batch in loader:
            predicted = model(batch['image'].to(device), batch['tokens'].to(device))
            mse = (predicted - batch['target'].to(device)).square().mean((1, 2, 3))
            psnr = -10 * torch.log10(mse.clamp_min(1e-12))
            records.extend({'sample_id': sid, 'psnr_db': float(value)}
                           for sid, value in zip(batch['sample_id'], psnr))
        return {'psnr_db': sum(row['psnr_db'] for row in records) / len(records)}, records


def _reference_editor_epoch(model, optimizer, loader, problem):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    trainable = [value for value in model.parameters() if value.requires_grad]
    total, count = 0., 0
    for step, batch in enumerate(loader):
        n = len(batch['sample_id'])
        group_samples = min(32, len(problem.train) - (step // 4) * 32)
        loss = problem.loss(model, batch, 'cpu')
        (loss * n / group_samples).backward()
        if (step + 1) % 4 == 0 or step + 1 == len(loader):
            torch.nn.utils.clip_grad_norm_(trainable, 1.)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        total += float(loss.detach()) * n
        count += n
    return total / count


def test_cached_editor_actual_trainer_restores_epoch3_and_matches_uninterrupted_epoch30(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from torch.utils.data import DataLoader
    from LightGenV2.tasks.t19_resnet20_baselines import train_tasks as trainer
    from LightGenV2.tasks.t19_resnet20_baselines import continue_lgvq50 as continuation
    from LightGenV2.tasks.t19_resnet20_baselines.frozen_resnet18 import (
        encoder_state_sha256, assert_encoder_unchanged,
    )
    from LightGenV2.tasks.t19_resnet20_baselines.tests.test_lgvq_continuation import CpuTorch

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(4)
    task_root = tmp_path / 'task'
    parent_run = task_root / 'runs/simulation/editor_original'
    parent_run.mkdir(parents=True)
    for filename in ('train_tasks.py', 'task_problems.py', 'models.py', 'lgvq_data.py',
                     'other_data.py', 'editor_dataset.py', 'train_easy100.py', 'frozen_resnet18.py',
                     'frozen_visual_models.py', 'frozen_multimodal_models.py', 'performance_cache.py',
                     'continue_lgvq50.py'):
        (task_root / filename).write_text('# offline source fixture\n')
    problem = TinyEditorProblem()
    cache_root = task_root / 'runs/editor_pair_cache'
    for split, dataset in (('train', problem.train), ('val', problem.test), ('test', problem.final_test)):
        cache.build_editor_cache(dataset, cache_root / split, problem.identity,
                                 source_files=[__file__], workers=0)
        assert cache.verify_editor_cache(cache_root / split, dataset, workers=0)['all_checked_rows_exactly_equal']
    torch.manual_seed(42)
    reference = TinyCachedEditor()
    visual_sha = encoder_state_sha256(reference.encoder)
    optimizer = torch.optim.AdamW((p for p in reference.parameters() if p.requires_grad),
                                  lr=.001, weight_decay=.0001)
    original_loader = DataLoader(problem.train, batch_size=8, shuffle=True,
        generator=torch.Generator().manual_seed(42), num_workers=2,
        multiprocessing_context='spawn', persistent_workers=True, pin_memory=True)
    parent_history, losses = [], []
    try:
        audits = []
        for epoch in range(1, 4):
            loss = _reference_editor_epoch(reference, optimizer, original_loader, problem)
            losses.append(loss)
            parent_history.append({'epoch': epoch, 'train_loss': loss, 'learning_rate': .001,
                                   'elapsed_seconds': float(epoch)})
            audits.append({'epoch': epoch, **assert_encoder_unchanged(reference.encoder, visual_sha)})
        parent_manifest = dict(task='abo_conditional_edit', backbone='resnet18_imagenet_frozen',
            seed=42, epochs=30, learning_rate=.001, effective_batch=32, microbatch=8,
            schedule='constant', warmup_epochs=0, text_encoder='byte_gru',
            training_scope='frozen_visual_trainable_text_and_decoder', data_identity=problem.identity,
            encoder_state_sha256_before=visual_sha, frozen_encoder_audits=audits)
        parent_path = parent_run / 'last_checkpoint.pt'
        torch.save({'model': reference.state_dict(), 'optimizer': optimizer.state_dict(),
                    'epoch': 3, 'manifest': parent_manifest}, parent_path)
        parent_sha = cache._file_sha(parent_path)
        (parent_run / 'history.json').write_text(json.dumps(parent_history))
        for _ in range(27):
            losses.append(_reference_editor_epoch(reference, optimizer, original_loader, problem))
        expected_order = problem.seen_training_order[3 * len(problem.train):]
        resumed_problem = TinyEditorProblem()
        monkeypatch.setattr(trainer, 'TASK', task_root)
        monkeypatch.setattr(trainer, 'EDITOR_COUNTS', {'train': 17, 'val': 5, 'test': 4})
        monkeypatch.setattr(trainer, 'Problem', lambda task: resumed_problem)
        monkeypatch.setattr(trainer, 'build_model', lambda *args, **kwargs: TinyCachedEditor())
        monkeypatch.setattr(trainer, 'require_published_source', lambda: None)
        monkeypatch.setattr(trainer, 'wait_for_gpu', lambda args: [])
        monkeypatch.setattr(trainer, 'git', lambda *args: 'offline_published_main')
        monkeypatch.setattr(trainer, 'torch', CpuTorch())
        monkeypatch.setattr(trainer, 'write_json', lambda path, value: path.write_text(json.dumps(value)))
        monkeypatch.setattr(trainer, 'save_checkpoint', lambda path, value: torch.save(value, path))
        args = SimpleNamespace(task='abo_conditional_edit', backbone='resnet18_imagenet_frozen',
            text_encoder='byte_gru', seed=42, epochs=30, smoke_batches=0, preflight=False,
            run_dir=task_root / 'runs/simulation/editor_cached_continuation', gpu_uuid='GPU-fixture',
            queue_status_path=None, shared_owner_pids=[], max_physical_gpus=4, max_jobs_per_gpu=3,
            resume_from=parent_path, editor_cache_root=cache_root)
        trainer.run(args)
        checkpoint = torch.load(args.run_dir / 'last_checkpoint.pt', map_location='cpu', weights_only=False)
        manifest = json.loads((args.run_dir / 'run_manifest.json').read_text())
        history = json.loads((args.run_dir / 'history.json').read_text())
        result = json.loads((args.run_dir / 'result.json').read_text())
        assert checkpoint['epoch'] == result['selected_epoch'] == 30
        assert [entry['epoch'] for entry in history] == list(range(1, 31))
        assert [entry['train_loss'] for entry in history] == losses
        assert resumed_problem.seen_training_order == expected_order
        audit = manifest['engineering_continuation']
        assert audit['completed_epoch'] == 3 and audit['model_and_AdamW_restored']
        assert audit['checkpoint_sha256'] == parent_sha == cache._file_sha(parent_path)
        assert manifest['encoder_state_sha256_before'] == visual_sha
        assert manifest['frozen_encoder_final_audit']['actual_encoder_state_sha256'] == visual_sha
        assert len(manifest['frozen_encoder_audits']) == 30
        assert manifest['status'] == result['status'] == 'complete'
        for name, expected in reference.state_dict().items():
            torch.testing.assert_close(checkpoint['model'][name], expected, rtol=0, atol=0)
        expected_optimizer = optimizer.state_dict()
        assert checkpoint['optimizer']['param_groups'] == expected_optimizer['param_groups']
        for index, expected in expected_optimizer['state'].items():
            for name, value in expected.items():
                torch.testing.assert_close(checkpoint['optimizer']['state'][index][name], value, rtol=0, atol=0)
    finally:
        continuation.shutdown_loader(original_loader)
        torch.set_num_threads(previous_threads)
