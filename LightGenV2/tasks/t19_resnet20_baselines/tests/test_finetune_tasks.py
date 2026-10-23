"""Fixed-budget fine tuning, immutable parents, and the approved pose loss repair."""
import json
import math
import shutil
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader

from LightGenV2.tasks.t19_resnet20_baselines import finetune_tasks as stage
from LightGenV2.tasks.t19_resnet20_baselines.frozen_resnet18 import (
    assert_encoder_unchanged, encoder_state_sha256, file_sha256,
)
from LightGenV2.tasks.t19_resnet20_baselines.tests.test_performance_cache import (
    TinyCachedEditor, TinyEditorProblem, _reference_editor_epoch,
)


def test_pose_repair_has_exact_gaussian_optimum_and_masks_invisible_joints():
    problem = stage.FineTuneProblem.__new__(stage.FineTuneProblem)
    problem.task = 'lsp'
    problem.settings = SimpleNamespace(heatmap_loss_weight=1.)
    target = torch.zeros(1, 2, 56, 56)
    target[0, 0, 8, 9] = 1.
    prediction = target.clone().requires_grad_()
    batch = {'image': torch.zeros(1, 3, 224, 224), 'heatmaps': target,
             'visible': torch.tensor([[True, False]]), 'keypoints': torch.tensor([[[36., 32.], [0., 0.]]])}
    assert float(problem.loss(lambda image: prediction, batch, 'cpu').detach()) == 0.
    with torch.no_grad():
        prediction[:, 0] += .1
        prediction[:, 1] = 1000.
    loss = problem.loss(lambda image: prediction, batch, 'cpu')
    assert float(loss.detach()) == pytest.approx(.01)
    loss.backward()
    assert prediction.grad[:, 0].abs().sum() > 0
    assert prediction.grad[:, 1].abs().sum() == 0


def fixture(tmp_path, monkeypatch, task='openmoji'):
    root = tmp_path / 'task'
    root.mkdir()
    real_task = stage.TASK
    for name in stage.SOURCE_FILES:
        shutil.copyfile(real_task / name, root / name)
    (root / 'configs').mkdir()
    problem = TinyEditorProblem(task)
    if task != 'abo_conditional_edit':
        problem.final_test = None
    problem.batch_size = 8
    torch.manual_seed(42)
    model = TinyCachedEditor()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.001, weight_decay=.0001)
    loader = DataLoader(problem.train, batch_size=8, shuffle=False)
    audits, history = [], []
    visual_sha = encoder_state_sha256(model.encoder)
    for epoch in range(1, 31):
        value = _reference_editor_epoch(model, optimizer, loader, problem)
        audits.append({'epoch': epoch, **assert_encoder_unchanged(model.encoder, visual_sha)})
        history.append({'epoch': epoch, 'train_loss': value})
    parent_dir = root / 'runs/simulation/parent30'
    parent_dir.mkdir(parents=True)
    parent_path = parent_dir / 'last_checkpoint.pt'
    manifest = dict(task=task, backbone='resnet18_imagenet_frozen', seed=42, epochs=30,
                    learning_rate=.001, schedule='constant', warmup_epochs=0, effective_batch=32,
                    microbatch=8, data_identity=problem.identity, text_encoder='byte_gru',
                    frozen_encoder_audits=audits, encoder_state_sha256_before=visual_sha)
    torch.save(dict(epoch=30, model=model.state_dict(), optimizer=optimizer.state_dict(), manifest=manifest), parent_path)
    (parent_dir / 'history.json').write_text(json.dumps(history))
    (parent_dir / 'result.json').write_text(json.dumps(dict(status='complete', selected_epoch=30,
        checkpoint_sha256=file_sha256(parent_path), selection_uses_test=False)))
    plan = dict(seed=42, learning_rate=.001, effective_batch=32, tasks={task:dict(
        parent_checkpoint='runs/simulation/parent30/last_checkpoint.pt', final_epoch=32,
        loss_profile=stage.LOSS_PROFILES[task])})
    if task == 'abo_conditional_edit':
        from LightGenV2.tasks.t19_resnet20_baselines import performance_cache
        cache_root = root / 'runs/cache/editor'
        for split, dataset in (('train', problem.train), ('val', problem.test), ('test', problem.final_test)):
            performance_cache.build_editor_cache(dataset, cache_root / split, problem.identity,
                                                source_files=[__file__], workers=0)
        plan['tasks'][task]['editor_cache_root'] = 'runs/cache/editor'
        monkeypatch.setattr(stage, 'EDITOR_COUNTS', {'train': 17, 'val': 5, 'test': 4})
    plan_path = root / 'configs/plan.json'
    plan_path.write_text(json.dumps(plan))
    monkeypatch.setattr(stage, 'TASK', root)
    monkeypatch.setattr(stage, 'FINAL_EPOCHS', {task: 32})
    monkeypatch.setattr(stage, 'COUNTS', {task: (17, 5)})
    monkeypatch.setattr(stage, 'FineTuneProblem', lambda name: problem)
    monkeypatch.setattr(stage, 'build_model', lambda *a, **k: TinyCachedEditor())
    monkeypatch.setattr(stage, 'require_published_source', lambda: None)
    monkeypatch.setattr(stage, 'git', lambda *a: 'offline_published_fixture')
    return root, problem, model, optimizer, plan_path, parent_path


def args(root, plan, smoke, task='openmoji'):
    return SimpleNamespace(task=task, plan=plan, device='cpu', gpu_uuid='GPU-offline',
        run_dir=root / ('runs/smoke/smoke' if smoke else 'runs/simulation/formal'),
        smoke_batches=2 if smoke else 0, smoke_result=None if smoke else root / 'runs/smoke/smoke/result.json')


@pytest.mark.parametrize('task', ['openmoji', 'lsp', 'abo_conditional_edit'])
def test_own_smoke_then_formal_preserves_parent_and_matches_reference_sampler_optimizer(tmp_path, monkeypatch, task):
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        root, problem, expected, optimizer, plan, parent = fixture(tmp_path, monkeypatch, task)
        parent_sha = file_sha256(parent)
        if task == 'lsp':
            expected_loader = DataLoader(problem.train, batch_size=8, shuffle=True,
                generator=torch.Generator().manual_seed(42), num_workers=2,
                multiprocessing_context='spawn', persistent_workers=True)
            expected_batches = expected_loader
        else:
            expected_loader, _ = stage.replay_index_loader(17, 8, 30, pin_memory=False)
            expected_batches = stage.CachedIndexBatches(problem.train, expected_loader)
        try:
            for _ in range(2):
                _reference_editor_epoch(expected, optimizer, expected_batches, problem)
        finally:
            stage.shutdown_loader(expected_loader)
        smoke_args = args(root, plan, True, task)
        stage.run(smoke_args)
        smoke = json.loads((smoke_args.run_dir / 'result.json').read_text())
        assert smoke['mode'] == 'smoke' and smoke['test_evaluations'] == 0
        assert smoke['selected_epoch'] == 31 and smoke['trainable_state_changed']
        smoke_manifest = json.loads((smoke_args.run_dir / 'run_manifest.json').read_text())
        if task == 'lsp':
            assert smoke_manifest['data_loader_workers'] == 0
        formal_args = args(root, plan, False, task)
        stage.run(formal_args)
        result = json.loads((formal_args.run_dir / 'result.json').read_text())
        checkpoint = torch.load(formal_args.run_dir / 'last_checkpoint.pt', weights_only=False)
        manifest = json.loads((formal_args.run_dir / 'run_manifest.json').read_text())
        assert result['status'] == 'complete' and result['selected_epoch'] == 32
        assert result['test_evaluations'] == 1 and result['selection_uses_test'] is False
        assert file_sha256(parent) == parent_sha == result['parent_checkpoint_sha256']
        assert manifest['optimizer_steps_after'] == math.ceil(17 / 32) * 32
        assert len(manifest['frozen_encoder_audits']) == 32
        for name, value in expected.state_dict().items():
            torch.testing.assert_close(checkpoint['model'][name], value, rtol=0, atol=0)
        assert stage.structured_sha256(checkpoint['optimizer']) == stage.structured_sha256(optimizer.state_dict())
        assert len(json.loads((formal_args.run_dir / 'fixed_weight_predictions.json').read_text())) == (
            4 if task == 'abo_conditional_edit' else 5)
        if task == 'lsp':
            assert 'fresh_seed42' in manifest['sampler_policy']
            assert manifest['data_loader_workers'] == 2
        else:
            assert manifest['sampler_replay']['completed_epochs_replayed'] == 30
    finally:
        torch.set_num_threads(previous)


def test_formal_rejects_missing_or_mismatched_smoke_and_never_overwrites_parent(tmp_path, monkeypatch):
    root, problem, model, optimizer, plan, parent = fixture(tmp_path, monkeypatch)
    digest = file_sha256(parent)
    formal = args(root, plan, False)
    formal.smoke_result = None
    with pytest.raises(ValueError, match='own smoke'):
        stage.run(formal)
    assert file_sha256(parent) == digest
    state = json.loads((formal.run_dir / 'run_manifest.json').read_text())
    assert state['status'] == 'failed'
    with pytest.raises(FileExistsError):
        stage.run(formal)
    fake = tmp_path / 'fake.json'
    fake.write_text(json.dumps(dict(status='complete', mode='smoke', test_evaluations=0,
                                   trainable_state_changed=True, parent_checkpoint_sha256='wrong')))
    with pytest.raises(ValueError, match='matching smoke'):
        stage.validate_smoke(fake, digest, 'plan', {}, 'unchanged_editing_objective')


def test_parent_rejects_nonlast_wrong_epoch_and_changed_adamw(tmp_path, monkeypatch):
    root, problem, model, optimizer, plan, parent = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='fixed last'):
        stage.load_parent(parent.with_name('best_checkpoint.pt'), problem, model, optimizer)
    payload = torch.load(parent, weights_only=False)
    payload['epoch'] = 29
    torch.save(payload, parent)
    with pytest.raises(ValueError, match='epoch/data'):
        stage.load_parent(parent, problem, model, optimizer)
    payload['epoch'] = 30
    payload['optimizer']['param_groups'][0]['lr'] = .01
    torch.save(payload, parent)
    result = json.loads((parent.parent / 'result.json').read_text())
    result['checkpoint_sha256'] = file_sha256(parent)
    (parent.parent / 'result.json').write_text(json.dumps(result))
    with pytest.raises(ValueError, match='optimizer'):
        stage.load_parent(parent, problem, model, optimizer)


def test_queue_commands_require_own_smoke_and_preserve_resource_limits(tmp_path):
    from LightGenV2.tasks.t19_resnet20_baselines.queue_finetune import stage_command
    job = dict(task='lsp', phase='smoke', name='own_smoke', smoke_name='own_smoke')
    smoke = stage_command(job, tmp_path, tmp_path / 'plan.json', 'GPU-fixed')
    assert smoke[smoke.index('--smoke-batches') + 1] == '4'
    assert '--smoke-result' not in smoke
    job.update(phase='formal', name='own_formal')
    formal = stage_command(job, tmp_path, tmp_path / 'plan.json', 'GPU-fixed')
    assert 'runs' in formal[formal.index('--smoke-result') + 1]
    assert formal[formal.index('--smoke-result') + 1].endswith('own_smoke/result.json') or formal[
        formal.index('--smoke-result') + 1].endswith('own_smoke\\result.json')
    assert formal[formal.index('--max-physical-gpus') + 1] == '4'
    assert formal[formal.index('--max-jobs-per-gpu') + 1] == '2'
