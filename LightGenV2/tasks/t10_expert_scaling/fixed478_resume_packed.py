"""Resume the fixed-478 study on two GPUs with multiple jobs per GPU."""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--gpus', type=int, nargs='+', default=[0, 1])
    ap.add_argument('--slots-per-gpu', type=int, default=2)
    args = ap.parse_args()
    if not 1 <= len(args.gpus) <= 3 or len(set(args.gpus)) != len(args.gpus) or args.slots_per_gpu < 1:
        raise ValueError('Use one to three distinct GPUs and at least one slot per GPU')

    jobs_file = args.out / 'jobs.json'
    if not jobs_file.exists():
        raise FileNotFoundError(jobs_file)
    jobs_doc = json.loads(jobs_file.read_text())
    jobs = jobs_doc['moe'] + jobs_doc['baseline']
    finished = [j for j in jobs if (args.out / j['name'] / 'result.json').exists()]
    pending = [j for j in jobs if j not in finished]
    failed = []
    running = {}
    slots = [(gpu, slot) for gpu in args.gpus for slot in range(args.slots_per_gpu)]
    uuids = {gpu: subprocess.check_output([
        'nvidia-smi', '-i', str(gpu), '--query-gpu=uuid', '--format=csv,noheader'
    ], text=True).strip() for gpu in args.gpus}

    design_path = args.out / 'design.json'
    design = json.loads(design_path.read_text())
    design.update(gpus=args.gpus, slots_per_gpu=args.slots_per_gpu,
                  scheduling='packed_multi_process', resumed=True)
    save(design_path, design)

    def aggregate():
        subprocess.run([sys.executable, '-m',
                        'LightGenV2.tasks.t10_expert_scaling.aggregate_fixed478',
                        '--out', str(args.out)], check=True)

    aggregate()

    def write_status(state='training'):
        save(args.out / 'status.json', dict(
            state=state, stage='full', total=len(jobs), complete=len(finished),
            remaining=len(pending), running={f'gpu{g}_slot{s}': j for (g, s), (_, _, j) in running.items()},
            failed=failed, gpus=args.gpus, slots_per_gpu=args.slots_per_gpu))

    try:
        while pending or running:
            for key in slots:
                if key in running or not pending:
                    continue
                gpu, slot = key
                if not any(k[0] == gpu for k in running):
                    memory = int(subprocess.check_output([
                        'nvidia-smi', '-i', str(gpu), '--query-gpu=memory.used',
                        '--format=csv,noheader,nounits'], text=True).strip())
                    if memory > 200:
                        continue
                job = pending.pop(0)
                folder = args.out / job['name']
                resume = (folder / 'last_checkpoint.pt').exists()
                if folder.exists() and not resume:
                    if folder.is_symlink():
                        raise RuntimeError(f'Incomplete symlink cannot be replaced: {folder}')
                    shutil.rmtree(folder)
                env = os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=uuids[gpu], CUDA_DEVICE_ORDER='PCI_BUS_ID')
                cmd = [sys.executable, '-u', '-m', 'LightGenV2.tasks.t10_expert_scaling.train',
                       '--data', job['data'], '--out', str(folder), '--arch', job['arch'],
                       '--experts', '4', '--top-k', str(job['top_k']), '--layers', '4',
                       '--lr', '.002', '--microbatch', '2', '--epochs', '60',
                       '--seed', str(job['seed'])]
                if resume:
                    cmd.append('--resume')
                log_path = args.out / 'logs' / f"{job['name']}.resume.log"
                log = log_path.open('a')
                proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True)
                running[key] = (proc, log, job)
                save(args.out / f'gpu{gpu}_slot{slot}.json',
                     dict(pid=proc.pid, gpu_uuid=uuids[gpu], resumed=resume, job=job, command=cmd))
            changed = False
            for key, (proc, log, job) in list(running.items()):
                if proc.poll() is None:
                    continue
                log.close()
                del running[key]
                if proc.returncode:
                    failed.append(dict(job=job, exit_code=proc.returncode))
                    raise RuntimeError(f'Job failed: {job}')
                finished.append(job)
                changed = True
            if changed:
                aggregate()
            write_status()
            if pending or running:
                time.sleep(10)
        write_status('complete')
        aggregate()
    except BaseException as error:
        write_status('failed_or_interrupted')
        save(args.out / 'error.json', dict(error=repr(error)))
        raise
    finally:
        for proc, log, _ in running.values():
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            log.close()
        snapshot = subprocess.check_output([
            'nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv'
        ], text=True)
        save(args.out / 'release_check.json', dict(
            own_children_exited=True, remaining_gpu_processes=snapshot, failed=failed))


if __name__ == '__main__':
    main()
