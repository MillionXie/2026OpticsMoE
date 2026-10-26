"""Explicit RTX4090 development A/B; never an A100/formal-table result.

The original worker file is NOT edited. A fail-closed AST adapter changes only
its GPU UUID, two actual-SM literals and power-cap literal for this fresh process.
All model/input/math/timing/numerical gates are original, including bitwise
equality to the sealed A100 reference. A cross-architecture mismatch stops A/B.
The first round is LG05's no-pool implementation; only after its success does
the second fresh process enable the existing, GPU-unverified event-pool method.
No foreign signals, power/clock changes, retries, or formal artifact updates.
"""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import types

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SIM = REPO/'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation'
ENTRY = Path(__file__).resolve()
WORKER = HERE/'lgvq_async_worker_20261026.py'
GPU = 'GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d'
NAME = 'NVIDIA GeForce RTX 4090'
SMS = 128
CAP = 450.
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def development_adapter(source):
    """Exactly four hardware-only substitutions, identified by complete ASTs."""
    tree = ast.parse(source)
    replacements = {
        "GPU = 'GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55'": f'GPU = {GPU!r}',
        'torch.cuda.get_device_properties(0).multi_processor_count != 108':
            f'torch.cuda.get_device_properties(0).multi_processor_count != {SMS}',
        'power_limit != 250': f'power_limit != {CAP!r}',
        'actual_visible_multiprocessors=108': f'actual_visible_multiprocessors={SMS}',
    }
    specs = []
    for old, new in replacements.items():
        if old.startswith('actual_visible_'):
            a = ast.parse('f('+old+')').body[0].value.keywords[0]
            b = ast.parse('f('+new+')').body[0].value.keywords[0]
        else:
            a = ast.parse(old).body[0]; b = ast.parse(new).body[0]
            if isinstance(a, ast.Expr):a, b = a.value, b.value
        specs.append((ast.dump(a, include_attributes=False), b))
    counts = [0]*len(specs)
    class HardwareOnly(ast.NodeTransformer):
        def generic_visit(self, node):
            signature = ast.dump(node, include_attributes=False)
            for i, (expected, replacement) in enumerate(specs):
                if signature == expected:
                    counts[i] += 1
                    return ast.copy_location(replacement, node)
            return super().generic_visit(node)
    tree = HardwareOnly().visit(tree)
    if counts != [1, 1, 1, 1]:
        raise ValueError('Original hardware assertion AST changed; refuse adaptation: '+repr(counts))
    return ast.fix_missing_locations(tree), counts


def load_candidate():
    tree, counts = development_adapter(WORKER.read_text(encoding='utf8'))
    candidate = types.ModuleType('lgvq_async_development_worker')
    candidate.__file__ = str(WORKER)
    exec(compile(tree, str(WORKER), 'exec'), candidate.__dict__)
    return candidate, counts


def source_map():
    candidate, _ = load_candidate()
    result = candidate.source_map()
    for name in ('run_lgvq_development_20261027.py', 'run_temporal_remainder.py',
                 'first_tf_supplement_20261025.py'):
        p = HERE/name
        result[p.relative_to(REPO).as_posix()] = sha(p)
    return result


def verify_sources(commit, sources):
    if not re.fullmatch('[0-9a-f]{40}', commit):raise ValueError('Exact published commit required')
    for name, digest in sources.items():
        if sha(REPO/name) != digest:raise RuntimeError('Live source changed: '+name)
        data = subprocess.check_output(['git', 'show', commit+':'+name], cwd=REPO, timeout=10)
        if hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError('Source does not match published Git blob: '+name)


def control(pipe, command):
    return subprocess.run(['nvidia-cuda-mps-control'], input=command+'\n', text=True,
        capture_output=True, env=dict(os.environ, CUDA_MPS_PIPE_DIRECTORY=pipe), timeout=5)


def snapshot(pipe, pid):
    from run_temporal_remainder import process_snapshot
    row = process_snapshot(GPU, pid, pipe)
    value = subprocess.check_output(['nvidia-smi', '-i', GPU,
        '--query-gpu=uuid,name,power.limit,memory.used', '--format=csv,noheader,nounits'], text=True, timeout=5)
    uuid, name, watts, memory = [v.strip() for v in value.strip().split(',')]
    if uuid != GPU or name != NAME or float(watts) != CAP:
        raise RuntimeError('Explicit observed development hardware identity changed')
    row['platform'] = dict(uuid=uuid, name=name, power_limit_w=float(watts), memory_mib=float(memory))
    if row['foreign_gpu_processes'] or row['foreign_mps_clients']:
        raise RuntimeError('Foreign compute; never signal it or accept contaminated timing')
    return row


class Guard:
    def __init__(self, args, sources):
        from first_tf_supplement_20261025 import serialize_mps_commands
        serialize_mps_commands()
        self.args = args; self.sources = sources; self.audit = []; self.error = None
        self.stop = threading.Event(); self.thread = None; self.bound = False

    def check(self):
        self.fast()
        row = snapshot(self.args.pipe, os.getpid()); self.audit.append(row)
        if self.bound and os.getpid() not in row['clients']:
            raise RuntimeError('Own private MPS client disappeared')
        if source_map() != self.sources:raise RuntimeError('Frozen development source changed')

    def bind(self):
        self.bound = True; self.check()
        def watch():
            while not self.stop.wait(.5):
                try:self.check()
                except BaseException as error:self.error = error; return
        self.thread = threading.Thread(target=watch, daemon=True); self.thread.start()

    def fast(self):
        if self.error:raise RuntimeError('Development exclusive guard failed') from self.error
        if time.time() >= self.args.deadline:raise TimeoutError('Finite development deadline')

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(20)
            if self.thread.is_alive():raise RuntimeError('Guard watcher did not exit')
        write(self.args.output/'process_audit.json', self.audit)


def worker(args):
    receipt = args.root/'source_receipt.json'
    if sha(receipt) != args.receipt_sha:raise ValueError('Source receipt changed')
    bound = json.loads(receipt.read_text()); verify_sources(bound['commit'], bound['sources'])
    if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU:raise ValueError('Wrong development GPU')
    args.output.mkdir()
    candidate, counts = load_candidate()
    write(args.output/'identity.json', dict(profile='RTX4090_DEVELOPMENT_ONLY', gpu=GPU,
        name=NAME, expected_sms=SMS, observed_cap_w=CAP, source_HEAD=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        published_commit=bound['commit'], sources=bound['sources'], command=sys.argv,
        hardware_only_AST_substitution_counts=counts, original_worker_sha256=sha(WORKER),
        performance_change='event_pool_only' if args.reuse_events else 'none_LG05_baseline',
        reference_policy='original_sealed_A100_bitwise_unchanged',
        qualified_for_final_hybrid_table=False, not_A100_speedup_evidence=True,
        private_MPS_pipe=args.pipe, pid=os.getpid()))
    # The reused raw adapter computes the identical DAG. Downgrade its result
    # before the original worker writes cost_report.json; never publish this
    # different-hardware run under the formal qualification flag.
    import lgvq_async_raw_adapter_20261026 as adapter
    original_adapt = adapter.adapt
    def development_adapt(raw):
        trace, report = original_adapt(raw)
        report.update(qualified_for_final_hybrid_table=False, development_only=True,
            hardware_profile='RTX4090_450W_128SM', not_A100_speedup_evidence=True,
            physical_hardware_time_measured=False)
        return trace, report
    adapter.adapt = development_adapt
    guard = Guard(args, bound['sources'])
    try:
        guard.check()
        import torch
        torch.set_num_threads(8)
        report = candidate.worker(args, guard, 2048)
        guard.check()
        if report.get('all_predictions_bitwise_equal') is not True:raise ValueError('Original numerical gate missing')
        write(args.output/'status.json', dict(status='complete_development_only', pid=os.getpid(),
            unix=time.time(), qualified_for_final_hybrid_table=False))
    except BaseException as error:
        write(args.output/'status.json', dict(status='failed', pid=os.getpid(), unix=time.time(), error=repr(error)))
        raise
    finally:
        guard.close(); adapter.adapt = original_adapt


def stop_child(child, leaf):
    if child is None or child.poll() is not None:return
    proc = Path('/proc')/str(child.pid)
    argv = [x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
    if (proc.stat().st_uid != os.geteuid() or os.getpgid(child.pid) != child.pid
            or str(ENTRY) not in argv or str(leaf) not in argv or '--phase' not in argv
            or argv[argv.index('--phase')+1] != 'worker'):
        raise RuntimeError('Refuse signal to unverified child')
    os.killpg(child.pid, signal.SIGINT); child.wait(timeout=60)


def suite(args):
    import fcntl
    from first_tf_supplement_20261025 import serialize_mps_commands
    serialize_mps_commands()
    sources = source_map(); commit = os.environ['MEASUREMENT_PUBLISHED_COMMIT']
    verify_sources(commit, sources)
    if args.root.exists():raise FileExistsError('Preserve prior development root; no retry')
    args.root.mkdir()
    write(args.root/'source_receipt.json', dict(commit=commit, sources=sources))
    receipt_sha = sha(args.root/'source_receipt.json')
    state = dict(status='admission', pid=os.getpid(), deadline=args.deadline, command=sys.argv,
        gpu=GPU, profile='RTX4090_DEVELOPMENT_ONLY', completed=[], mps_released=False)
    child = None; active = False; leaf = None
    def save():write(args.root/'suite_status.json', state)
    save()
    try:
        with (SIM/'.lgvq_development_gpu1.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            if control(args.pipe, 'get_server_list').returncode == 0:raise RuntimeError('Private daemon already active')
            for i in range(12):
                if time.time() >= args.deadline-60:raise TimeoutError('Development admission deadline')
                row = snapshot(args.pipe, -1)
                if row['platform']['memory_mib'] >= 384:raise RuntimeError('Development GPU not idle')
                time.sleep(.25)
            Path(args.pipe).mkdir(mode=0o700)
            (args.root/'mps_logs').mkdir()
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, CUDA_DEVICE_ORDER='PCI_BUS_ID',
                CUDA_MPS_PIPE_DIRECTORY=args.pipe, CUDA_MPS_LOG_DIRECTORY=str(args.root/'mps_logs'),
                CUDA_MPS_ACTIVE_THREAD_PERCENTAGE='100', CUDA_MODULE_LOADING='LAZY',
                HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='8',
                MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
            subprocess.run(['nvidia-cuda-mps-control', '-d'], env=env, capture_output=True,
                text=True, timeout=10, check=True); active = True
            for label, reuse in [('baseline', False), ('event_pool', True)]:
                if time.time() >= args.deadline-60:raise TimeoutError('No new round beyond deadline')
                if source_map() != sources:raise RuntimeError('Development source changed')
                snapshot(args.pipe, -1)
                leaf = args.root/label
                command = [sys.executable, '-u', str(ENTRY), '--phase', 'worker', '--root', str(args.root),
                    '--output', str(leaf), '--deadline', str(args.deadline), '--receipt-sha', receipt_sha]
                if reuse:command.append('--reuse-events')
                write(args.root/(label+'.command.json'), command)
                with (args.root/(label+'.log')).open('x') as log:
                    child = subprocess.Popen(command, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    state.update(status='running', phase=label, child=child.pid); save()
                    child.wait(timeout=max(1, args.deadline-time.time()))
                if child.returncode != 0:raise RuntimeError('Preserve failed development leaf; no second round/retry')
                state['completed'].append(label); save()
            state.update(status='complete_development_only', finished=time.time())
    except BaseException as error:
        state.update(status='stopped', error=repr(error)); stop_child(child, leaf); raise
    finally:
        try:
            if active:
                for server in re.findall(r'^\s*(\d+)\s*$', control(args.pipe, 'get_server_list').stdout, re.M):
                    clients = control(args.pipe, 'get_client_list '+server)
                    if clients.returncode or re.search(r'^\s*\d+\s*$', clients.stdout, re.M):
                        raise RuntimeError('Refuse private daemon quit with active/unverifiable clients')
                ended = control(args.pipe, 'quit')
                if ended.returncode:raise RuntimeError('Private daemon shutdown failed')
                for _ in range(15):
                    if control(args.pipe, 'get_server_list').returncode != 0:break
                    time.sleep(.2)
            state['mps_released'] = control(args.pipe, 'get_server_list').returncode != 0
            if not state['mps_released']:raise RuntimeError('Private daemon not released')
        except BaseException as error:
            state.update(status='stopped', cleanup_error=repr(error)); raise
        finally:save()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['suite', 'worker'], required=True)
    p.add_argument('--root', type=Path, required=True); p.add_argument('--output', type=Path)
    p.add_argument('--deadline', type=float, required=True); p.add_argument('--receipt-sha')
    p.add_argument('--reuse-events', action='store_true')
    args = p.parse_args(); args.root = args.root.resolve()
    if args.root.parent != SIM or not re.fullmatch(r'lgvq_rtx4090_development_20261027_\d{2}', args.root.name):
        raise ValueError('New canonical development root required')
    if not math.isfinite(args.deadline) or not 0 < args.deadline-time.time() <= 3600:
        raise ValueError('Finite development deadline at most one hour required')
    args.pipe = '/DATA/DATA1/guest3/.cache/'+args.root.name
    args.optimize_runtime = True; args.parallel_text = False
    if args.phase == 'worker':
        if args.output is None or args.output.resolve() not in [args.root/'baseline', args.root/'event_pool']:
            raise ValueError('Exact development leaf required')
        args.output = args.output.resolve()
        if args.reuse_events != (args.output.name == 'event_pool'):raise ValueError('Frozen A/B plan differs')
    (suite if args.phase == 'suite' else worker)(args)


if __name__ == '__main__':main()
