"""User-authorized twelve-layer graph; reuse the frozen FishNet training loop."""
import argparse
import copy
import gc
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import fishnet_ablation as f
from model import coherent_modulation

HERE = Path(__file__).resolve().parent
GPU2 = 'GPU-6dcca91a-8e08-1a50-9aa6-81defeaed50b'
GPU4 = 'GPU-1b963983-7909-af6e-0528-f0f0661ab549'
PARAMETERS = 2649088


def graph_config(base):
    """Change only the stage count and descriptive depth fields in the L6 graph."""
    graph = copy.deepcopy(base)
    graph['model']['num_cycles'] = 6
    graph['expert_bank']['d2nn'].update(num_layers=6, nonlinear_schedule=[True] * 6)
    graph['global_fc']['num_masks'] = 6
    graph['experiment']['baseline_layers'] = 12
    graph['experiment']['expected_parameters']['moe'] = PARAMETERS
    graph['variant'].update(id='moe_L12_oeo_relu_softsign', depth=12)
    graph['output']['dir'] = './runs/moe_L12_oeo_relu_softsign'
    return graph


def install_depth12_residual(model, phase_type, rho):
    assert model.net.num_cycles == 6 and rho in (0., .3)

    class Depth12Phase(phase_type):
        def forward(self, field):
            assert not self._dropout_enabled()
            return coherent_modulation(field, self.get_phase(), self.residual_rho)

    targets = []
    for name, module in model.named_modules():
        if isinstance(module, phase_type) and ('.expert_bank.' in name or '.global_fcs.' in name):
            module.__class__ = Depth12Phase
            module.residual_rho = float(rho)
            targets.append(name)
    assert len(targets) == 60 and all('router' not in name for name in targets)
    assert sum(p.numel() for p in model.parameters()) == PARAMETERS
    assert len(list(model.parameters())) == 61
    model.net.expert_bank.vectorize_homogeneous_d2nn = False
    return targets


def install_graph():
    # This registry and function binding are local to this extension process.
    f.r.CONFIGS['moe_L12_oeo_relu_softsign'] = graph_config(f.r.CONFIGS['moe_L6_oeo_relu_softsign'])
    def build(arch, depth, cfg):
        assert arch == 'moe' and depth == 12
        model = f.t.original_build(arch, depth, cfg)
        install_depth12_residual(model, f.t.PhaseLayer, cfg['residual_rho'])
        return model
    f.t.build = build
    f.t.install_residual = install_depth12_residual
    f.gpu_budget.ALLOWED.update([GPU2, GPU4])


def sources(profile, extension):
    return dict(**f.source_identity(profile), depth_extension=f.r.sha(__file__),
                depth_extension_config=f.r.sha(extension))


def worker(a, profile, extension):
    assert a.gpu in extension['gpu_uuids'] and os.environ.get('CUDA_VISIBLE_DEVICES') == a.gpu
    assert a.data and a.depth == 12 and extension['epochs'] == profile['epochs'] == 100
    assert extension['seed'] == profile['seed'] == 17
    assert f.selection.selection_mode(profile) == 'validation'
    policy = f.gpu_budget.configure(f.torch, a.gpu, shared=True, memory_gib=3.)
    a.out.mkdir(parents=True, exist_ok=False)
    f.torch.set_num_threads(4)
    manifest = f.r.read(a.data.parent / 'data_manifest.json')
    original = HERE / 'fishnet_profile.json'
    f.selection.validate_prepared_profile(profile, manifest, f.r.read(original), f.r.sha(original))
    assert manifest['preparation_source_sha256'] == f.r.sha(HERE / 'prepare_fishnet.py')
    assert manifest['cache_sha256'] == f.r.sha(a.data)
    cfg = f.config(profile, manifest, a.rho)
    cfg.update(microbatch=2, effective_batch_size=16, evaluation_batch_size=2,
               spare_memory_authorized=True, main_optical_layers=12, cycles=6,
               depth_extension_config_sha256=f.r.sha(a.extension_config))
    src = sources(a.profile, a.extension_config)
    f.r.save(a.out / 'metadata.json', dict(config=cfg, sources=src, command=sys.argv,
        pid=os.getpid(), git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        data_sha256=f.r.sha(a.data), manifest_sha256=f.r.sha(a.data.parent / 'data_manifest.json'),
        gpu_policy=policy, environment=f.t.m.environment(), time=f.r.now(),
        selection_split='val', test_development=False, training_reads_test_for_gradients=False,
        test_previously_observed=True, graph=extension))
    try:
        if a.phase == 'smoke':
            f.smoke(a, profile, cfg)
        else:
            f.train(a, profile, manifest, cfg, src)
    except BaseException:
        f.r.save(a.out / 'status.json', dict(state='failed', traceback=traceback.format_exc(), time=f.r.now()))
        raise
    finally:
        gc.collect()
        f.torch.cuda.empty_cache()


def pair(a, profile, extension):
    assert a.gpus == extension['gpu_uuids'] and len(set(a.gpus)) == 2
    a.out.mkdir(parents=True, exist_ok=False)
    data = a.data_root / 'fishnet_fixed_split.npz'
    f.r.save(a.out / 'metadata.json', dict(command=sys.argv, pid=os.getpid(), profile=profile,
        extension=extension, sources=sources(a.profile, a.extension_config), time=f.r.now(),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()))
    children, logs = [], []
    def launch(phase, rho, gpu, folder):
        f.gpu_budget.preflight(gpu, shared=True, memory_gib=3.)
        cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--phase', phase,
               '--depth', '12', '--rho', str(rho), '--data', str(data), '--profile', str(a.profile),
               '--extension-config', str(a.extension_config), '--out', str(a.out / folder), '--gpu', gpu]
        log = (a.out / (folder + '.log')).open('w'); logs.append(log)
        child = subprocess.Popen(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu),
                                 stdout=log, stderr=subprocess.STDOUT)
        children.append(child)
        return dict(pid=child.pid, command=cmd, phase=phase, depth=12, rho=rho, gpu_uuid=gpu)
    try:
        receipt = launch('smoke', .3, a.gpus[0], 'smoke')
        f.r.save(a.out / 'current_processes.json', [receipt])
        f.r.save(a.out / 'status.json', dict(state='smoke', completed=0, total=2))
        assert children[-1].wait() == 0, 'L12 smoke failed; preserve log, do not train'
        receipts = [launch('train', rho, gpu, f'L12_rho{rho:.1f}')
                    for rho, gpu in zip([0., .3], a.gpus)]
        f.r.save(a.out / 'current_processes.json', receipts)
        train_children = children[-2:]
        while any(child.poll() is None for child in train_children):
            assert all(child.poll() in (None, 0) for child in train_children), 'L12 arm failed; preserve both runs'
            f.r.save(a.out / 'status.json', dict(state='running', completed=sum(child.poll() == 0 for child in train_children),
                                               total=2, processes=receipts))
            time.sleep(5)
        assert all(child.returncode == 0 for child in train_children)
        results = [f.r.read(a.out / f'L12_rho{rho:.1f}' / 'result.json') for rho in [0., .3]]
        for key in ['depth', 'parameters', 'initial_parameter_sha256', 'orders', 'transforms',
                    'data_sha256', 'epochs_completed']:
            assert results[0][key] == results[1][key], ('Pair mismatch', key)
        metas = [f.r.read(a.out / f'L12_rho{rho:.1f}' / 'metadata.json') for rho in [0., .3]]
        assert metas[0]['sources'] == metas[1]['sources']
        changed = {k for k in metas[0]['config'] if metas[0]['config'][k] != metas[1]['config'][k]}
        assert changed == {'residual_rho'}
        f.r.save(a.out / 'results.json', results)
        f.r.save(a.out / 'pair_lock_L12.json', dict(depth=12, paired_initialization=True, paired_orders=True,
            paired_transforms=True, same100_epochs=True, sources_equal=True, config_difference=['residual_rho'],
            checkpoints=[x['checkpoint_sha256'] for x in results], selection=profile['selection'],
            selection_split='val', test_development=False))
        f.r.save(a.out / 'status.json', dict(state='complete', completed=2, total=2,
                                           gpu_released_on_exit=True, time=f.r.now()))
    except BaseException:
        for child in children:
            if child.poll() is None:
                child.terminate(); child.wait()
        f.r.save(a.out / 'status.json', dict(state='failed', traceback=traceback.format_exc(), time=f.r.now()))
        raise
    finally:
        for log in logs:
            log.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['pair', 'smoke', 'train'], required=True)
    p.add_argument('--profile', type=Path, default=HERE / 'fishnet_validation_profile.json')
    p.add_argument('--extension-config', type=Path, default=HERE / 'fishnet_depth12_config.json')
    p.add_argument('--data-root', type=Path)
    p.add_argument('--data', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--depth', type=int, choices=[12], default=12)
    p.add_argument('--rho', type=float, choices=[0., .3], default=.3)
    p.add_argument('--gpu')
    p.add_argument('--gpus', nargs=2)
    a = p.parse_args()
    profile, extension = f.r.read(a.profile), f.r.read(a.extension_config)
    assert extension['main_optical_layers'] == 12 and extension['cycles'] == 6
    assert extension['parameters'] == PARAMETERS and extension['rhos'] == [0., .3]
    install_graph()
    if a.phase == 'pair':
        assert a.data_root
        pair(a, profile, extension)
    else:
        worker(a, profile, extension)


if __name__ == '__main__':
    main()
