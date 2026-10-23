"""One GPU per seed, two dataset streams per GPU, explicit stage dependencies.

Every final model has original60 + continuation30 + margin30 epochs.
Only final 120-epoch results are aggregated.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

from .train import save, sha


def aggregate_metrics(rows):
    result = {}
    for split in ('train', 'val', 'test'):
        values = [row[f'{split}_accuracy'] for row in rows]
        result[f'{split}_mean'] = statistics.mean(values) if values else None
        result[f'{split}_sample_sd'] = statistics.stdev(values) if len(values)>1 else None
    return result


def ready_job(pending, running, seed=None):
    active_paths = {j['out'] for _, _, j in running.values()}
    active_streams = {(j['dataset'], j['seed']) for _, _, j in running.values()}
    candidates = [j for j in pending if j['source'] not in active_paths
                  and (seed is None or j['seed'] == seed)
                  and (j['dataset'], j['seed']) not in active_streams
                  and (Path(j['source'])/'result.json').exists()]
    # Start the long Organ pipelines first to minimize the final completion time.
    return min(candidates, key=lambda j: (j['dataset'] != 'organcmnist', j['seed'], j['stage'])) if candidates else None


def summary(root, final_runs):
    finals = []
    for ds in ('kather2016', 'organcmnist'):
        per_seed = []
        for seed in (17, 27, 37):
            run = final_runs[(ds, seed)]
            if not (run/'test_result.json').exists() or not (run/'result.json').exists():
                continue
            r = json.loads((run/'result.json').read_text())
            t = json.loads((run/'test_result.json').read_text())
            m = json.loads((run/'metadata.json').read_text())
            actual = sha(run/'best_checkpoint.pt')
            if actual != r['checkpoint_sha256'] or actual != t['checkpoint_sha256']:
                raise ValueError(f'Checkpoint identity mismatch: {run}')
            if int(m['arguments']['seed']) != seed:
                raise ValueError(f'Seed mismatch: {run}')
            diag = m['diagnosis']
            if (diag.get('margin_weight') != .2 or diag.get('capture_weight') != .005
                    or diag.get('selection_metric') != 'accuracy'
                    or int(m['arguments']['epochs']) != 120):
                raise ValueError(f'Final-stage protocol mismatch: {run}')
            per_seed.append(dict(seed=seed, run=str(run), checkpoint_sha256=actual,
                                 train_accuracy=r['train']['accuracy'], val_accuracy=r['val']['accuracy'],
                                 test_accuracy=t['test']['accuracy'], best_epoch=r['best_epoch']))
        finals.append(dict(dataset=ds, top_k=4, expected_seeds=[17, 27, 37], n_seeds=len(per_seed),
                           per_seed=per_seed, total_training_epochs=120,
                           **aggregate_metrics(per_seed)))
    save(root/'table_summary.json', dict(
        role='development_followup_after_observed_test', uncertainty='sample_sd_ddof1_null_if_n_lt2',
        scope='Only Kather2016 and OrganCMNIST k4, 120 total epochs; no 60-epoch table.',
        updated_k4=finals))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--gpus', type=int, nargs=3, default=[0, 1, 2],
                    help='Physical GPUs assigned in order to seeds 17,27,37')
    ap.add_argument('--sources', type=Path, default=Path('/DATA/DATA1/guest3/t10_fixed478_3datasets_20260921_runs'))
    a = ap.parse_args()
    if len(set(a.gpus)) != 3 or 6 in a.gpus:
        raise ValueError('Three distinct GPUs; GPU6 excluded')
    a.out.mkdir(parents=True, exist_ok=False)
    simulation = a.out.parent
    jobs = []
    finals = {}
    for ds in ('kather2016', 'organcmnist'):
        for seed in (17, 27, 37):
            source = a.sources/f'{ds}__moe_oeo_N4_k4_L4_lr0.002_s{seed}'
            stage1 = (simulation/'topk_diagnosis_20261023'/f'{ds}_k4_s17_extra30'
                      if seed == 17 else a.out/f'{ds}_s{seed}'/'continuation30')
            stage2 = (simulation/'kather_k4_margin_20261023'
                      if ds == 'kather2016' and seed == 17 else a.out/f'{ds}_s{seed}'/'margin30')
            finals[(ds, seed)] = stage2
            for stage, src, dst in [(1, source, stage1), (2, stage1, stage2)]:
                jobs.append(dict(dataset=ds, seed=seed, stage=stage, source=str(src), out=str(dst)))
    reused = [j for j in jobs if (Path(j['out'])/'result.json').exists()]
    pending = [j for j in jobs if j not in reused]
    finished = list(reused)
    running = {}
    child_pids = []

    def stop(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    save(a.out/'plan.json', dict(jobs=jobs, reused=reused, new_trainings=len(pending),
         seed_gpu_mapping=dict(zip((17, 27, 37), a.gpus)), max_gpus=3,
         max_dataset_streams_per_gpu=2, pid=os.getpid(), commit=commit, command=sys.argv))
    try:
        summary(a.out, finals)
        while pending or running:
            for seed, gpu in zip((17, 27, 37), a.gpus):
              for _ in range(2):
                owned = [j for (g, _), (_, _, j) in running.items() if g == gpu]
                if len(owned) >= 2:
                    break
                job = ready_job(pending, running, seed)
                if job is None:
                    break
                if not owned:
                    used = int(subprocess.check_output(['nvidia-smi', '-i', str(gpu),
                        '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
                    if used > 200:
                        break
                pending.remove(job)
                uuid = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=uuid',
                                            '--format=csv,noheader'], text=True).strip()
                env = os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID')
                cmd = [sys.executable, '-u', '-m', 'LightGenV2.tasks.t10_expert_scaling.diagnose_topk',
                   '--source', job['source'], '--out', job['out'], '--epochs', '30']
                if job['stage'] == 2:
                    cmd += ['--lr', '.0005', '--margin-weight', '.2', '--capture-weight', '.005',
                        '--selection-metric', 'accuracy']
                if job['dataset'] == 'organcmnist':
                    cmd += ['--test-source', '/DATA/DATA1/guest3/demo_reproduction_data/organcmnist/organcmnist.npz']
                log = (a.out/f"{job['dataset']}_s{job['seed']}_stage{job['stage']}.log").open('w')
                active = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                child_pids.append(active.pid)
                running[(gpu, job['dataset'])] = (active, log, dict(**job, pid=active.pid, gpu_uuid=uuid))
            for gpu, (proc, log, job) in list(running.items()):
                if proc.poll() is None:
                    continue
                log.close()
                del running[gpu]
                if proc.returncode:
                    raise RuntimeError(f'Child failed: {job}')
                finished.append(job)
                summary(a.out, finals)
            save(a.out/'status.json', dict(state='training', complete=finished, pending=pending,
                 running={str(g):j for g, (_, _, j) in running.items()}, max_gpus=3))
            if pending or running:
                time.sleep(10)
        save(a.out/'status.json', dict(state='complete', complete=finished, pending=[]))
    except BaseException as error:
        save(a.out/'status.json', dict(state='failed_or_interrupted', error=repr(error),
                                       complete=finished, pending=pending))
        raise
    finally:
        for active, log, _ in running.values():
            if active.poll() is None:
                os.killpg(active.pid, signal.SIGTERM)
                try:
                    active.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(active.pid, signal.SIGKILL)
                    active.wait()
            log.close()
        save(a.out/'release_check.json', dict(own_children_exited=True, child_pids=child_pids,
                                             timestamp=time.time()))


if __name__ == '__main__':
    main()
