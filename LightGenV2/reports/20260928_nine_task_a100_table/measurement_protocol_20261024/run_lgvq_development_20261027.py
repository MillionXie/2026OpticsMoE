"""Explicit RTX4090 development A/B; never an A100/formal-table result.

The original worker file is NOT edited. A fail-closed AST adapter changes only
its GPU UUID, two actual-SM literals and power-cap literal for this fresh process.
All model/input/math/timing/numerical gates are original. The default reference
is sealed A100; explicit --same-card-reference is the later human authorization
to compare ONLY this card's optimization with its preserved unmodified output.
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
SAME_CARD_REFERENCE = SIM/'lgvq_rtx4090_development_20261027_01'/'baseline'
SAME_CARD_SHA256 = '31fedd4d30d24b705774007be0baa34f78c34a7bfa21b8890718a7334fd3bc88'
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


def load_candidate(once_prompt_render=False):
    tree, counts = development_adapter(WORKER.read_text(encoding='utf8'))
    if once_prompt_render:
        from lgvq_development_prompt_render_20261027 import prompt_render_AST
        tree=prompt_render_AST(tree)
    candidate = types.ModuleType('lgvq_async_development_worker')
    candidate.__file__ = str(WORKER)
    exec(compile(tree, str(WORKER), 'exec'), candidate.__dict__)
    if once_prompt_render:
        from lgvq_development_prompt_render_20261027 import RenderOnce
        candidate.development_render_once=RenderOnce()
    return candidate, counts


def source_map():
    candidate, _ = load_candidate()
    result = candidate.source_map()
    for name in ('run_lgvq_development_20261027.py', 'run_temporal_remainder.py',
                 'first_tf_supplement_20261025.py', 'lgvq_development_graph_pools_20261027.py',
                 'lgvq_development_prompt_render_20261027.py', 'lgvq_development_conv_layout_20261027.py',
                 'lgvq_development_patch_linear_20261027.py','lgvq_development_patch_conv2d_20261027.py'):
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


def same_card_reference():
    """Admit the precise preserved baseline only; never relabel its rejection."""
    import numpy as np
    folder = SAME_CARD_REFERENCE
    read = lambda p:json.loads(p.read_text())
    suite = read(folder.parent/'suite_status.json'); identity = read(folder/'identity.json')
    actual = read(folder/'async_identity.json'); status = read(folder/'status.json')
    candidate, _ = load_candidate()
    if (suite['status'] != 'stopped' or suite['mps_released'] is not True
            or status['status'] != 'failed'
            or status['error'] != "ValueError('Every score must be bitwise equal to the sealed reference')"
            or identity['gpu'] != GPU or identity['expected_sms'] != SMS or identity['observed_cap_w'] != CAP
            or identity['performance_change'] != 'none_LG05_baseline'
            or actual['gpu_uuid'] != GPU or actual['checkpoint_sha256'] != '5303b574b200e14bf943af21c60a246720be453b9cf8847cd93c0eaaa243a77c'
            or actual['optimized_runtime'] is not True or actual['same_field_parallel_text'] is not False):
        raise ValueError('Exact unmodified same-card baseline identity required')
    for pid in (suite['pid'], identity['pid']):
        if Path('/proc', str(pid)).exists():raise RuntimeError('Baseline process not released')
    if control(identity['private_MPS_pipe'], 'get_server_list').returncode == 0:
        raise RuntimeError('Baseline private MPS not released')
    for name, digest in candidate.source_map().items():
        if identity['sources'].get(name) != digest:raise RuntimeError('Baseline mathematical/timing source changed: '+name)
    if sha(folder/'outputs.npz') != SAME_CARD_SHA256:raise ValueError('Preserved same-card outputs changed')
    if sha(REPO/candidate.REFERENCE_RELATIVE) != candidate.REFERENCE_SHA256:
        raise ValueError('Original A100 reference changed')
    with np.load(folder/'outputs.npz', allow_pickle=False) as z:array = z['prediction']
    candidate.validate_predictions(array, array.copy())  # original shape/dtype/finite/four-cycle gates
    audit = read(folder/'process_audit.json')
    if not audit or any(r['foreign_gpu_processes'] or r['foreign_mps_clients']
            or r['platform']['uuid'] != GPU for r in audit):raise ValueError('Baseline not exclusive')
    events = read(folder/'async_event_records.json')
    if len(events['records']) != 128 or [r['field'] for r in events['records']] != list(range(128)):
        raise ValueError('Full unmodified baseline event records required')
    return dict(profile='HUMAN_AUTHORIZED_SAME_CARD_DEVELOPMENT_ONLY',
        path=str(folder/'outputs.npz'), sha256=SAME_CARD_SHA256,
        original_A100_reference_sha256=candidate.REFERENCE_SHA256,
        baseline_software_ms16=events['measured_software_wall_ms']/128,
        baseline_identity_sha256=sha(folder/'identity.json'),
        baseline_events_sha256=sha(folder/'async_event_records.json'),
        baseline_rejection_preserved=True, not_A100_bitwise_equivalence=True)


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
    candidate, counts = load_candidate(args.once_prompt_render)
    reference = same_card_reference() if args.same_card_reference else None
    if reference:
        candidate.REFERENCE_RELATIVE = str(Path(reference['path']).relative_to(REPO))
        candidate.REFERENCE_SHA256 = reference['sha256']
    write(args.output/'identity.json', dict(profile='RTX4090_DEVELOPMENT_ONLY', gpu=GPU,
        name=NAME, expected_sms=SMS, observed_cap_w=CAP, source_HEAD=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        published_commit=bound['commit'], sources=bound['sources'], command=sys.argv,
        hardware_only_AST_substitution_counts=counts, original_worker_sha256=sha(WORKER),
        performance_change=('complete_patch_channel_temporal_fold_conv2d' if args.patch_conv2d_channels else
            'complete_patch_temporal_height_fold_conv2d' if args.patch_conv2d else
            'one_patch_linear_separate_bias' if args.patch_linear_separate_bias else
            'one_patch_biased_linear' if args.patch_linear else
            'same_field_text_visual_stream_overlap' if args.parallel_text else
            'Conv3d_channels_last_3d_storage_only' if args.channels_last_3d else
            'constant_prompt_TEXT_render_once_inside_window' if args.once_prompt_render else
            'CPU_intraop_threads_1' if args.cpu_threads==1 else
            'same_slot_same_stream_graph_allocator_pool' if args.share_graph_pools else
            'event_pool_only' if args.reuse_events else 'none_LG05_baseline'),
        CPU_intraop_threads=args.cpu_threads,
        reference_policy='same_card_development_bitwise' if reference else 'original_sealed_A100_bitwise_unchanged',
        same_card_reference=reference,
        qualified_for_final_hybrid_table=False, not_A100_speedup_evidence=True,
        private_MPS_pipe=args.pipe, pid=os.getpid()))
    # The reused raw adapter computes the identical DAG. Downgrade its result
    # before the original worker writes cost_report.json; never publish this
    # different-hardware run under the formal qualification flag.
    import lgvq_async_raw_adapter_20261026 as adapter
    original_adapt = adapter.adapt
    def development_adapt(raw):
        try:
            trace, report = original_adapt(raw)
        except ValueError as error:
            if (not args.software_only_development or str(error) !=
                    'No causal host/CUDA clock alignment; preserve raw, do not shift clocks arbitrarily'):
                raise
            # The CPU continuous wall and exact scores are independently saved.
            # Do NOT repair clocks, produce a projection, or qualify this DAG.
            trace = dict(status='REJECTED_HOST_CUDA_ALIGNMENT', error=str(error),
                clock_values_modified=False, projection_computed=False)
            report = dict(status='SOFTWARE_ONLY_DEVELOPMENT_DAG_REJECTED',
                all_predictions_bitwise_equal=True,
                actual_async_software_wall_ms=raw['actual_async_software_wall_ms'],
                software_time16_ms=raw['actual_async_software_time16_ms'],
                reference_sha256=raw['reference_sha256'],
                raw_board_energy=raw['raw_board_energy'], projection_computed=False,
                causal_DAG_accepted=False, clock_values_modified=False)
        report.update(qualified_for_final_hybrid_table=False, development_only=True,
            hardware_profile='RTX4090_450W_128SM', not_A100_speedup_evidence=True,
            physical_hardware_time_measured=False, same_card_reference=reference,
            A100_bitwise_equivalence_claimed=False)
        return trace, report
    adapter.adapt = development_adapt
    import lgvq_lowlevel_capture_20261026 as lowlevel
    original_tape = lowlevel.LowLevelTape
    import lgvq_live_frontend as live
    original_front_init=live.LiveFront.__init__
    if args.channels_last_3d:
        from lgvq_development_conv_layout_20261027 import install
        def layout_front_init(front,*a,**kw):
            original_front_init(front,*a,**kw)
            write(args.output/'conv_layout_identity.json',install(front))
        live.LiveFront.__init__=layout_front_init
    if args.patch_linear or args.patch_linear_separate_bias:
        from lgvq_development_patch_linear_20261027 import install
        def linear_front_init(front,*a,**kw):
            original_front_init(front,*a,**kw)
            write(args.output/'patch_linear_identity.json',install(front,separate_bias=args.patch_linear_separate_bias))
        live.LiveFront.__init__=linear_front_init
    if args.patch_conv2d or args.patch_conv2d_channels:
        from lgvq_development_patch_conv2d_20261027 import install
        def conv2d_front_init(front,*a,**kw):
            original_front_init(front,*a,**kw)
            write(args.output/'patch_conv2d_identity.json',install(front,channels_fold=args.patch_conv2d_channels))
        live.LiveFront.__init__=conv2d_front_init
    if args.share_graph_pools:
        from lgvq_development_graph_pools_20261027 import tape_class
        lowlevel.LowLevelTape = tape_class(original_tape)
    guard = Guard(args, bound['sources'])
    try:
        guard.check()
        import torch
        torch.set_num_threads(args.cpu_threads)
        report = candidate.worker(args, guard, 2048)
        guard.check()
        if args.once_prompt_render:
            cache=candidate.development_render_once
            if cache.requests!=128 or cache.renders!=1:raise ValueError('Exactly128 real tokenizer requests and one timed text render required')
            raw=json.loads((args.output/'async_event_records.json').read_text())
            first=raw['records'][0]['tokenizer']
            if not first['start']<=cache.first_interval[0]<=cache.first_interval[1]<=first['end']:
                raise ValueError('First string render must stay inside original measured first tokenizer')
            write(args.output/'prompt_render_identity.json',dict(requests=cache.requests,renders=cache.renders,
                first_render_interval=cache.first_interval,first_render_in_original_window=True,
                token_tensors_cached=False,features_cached=False,scores_cached=False))
        if report.get('all_predictions_bitwise_equal') is not True:raise ValueError('Original numerical gate missing')
        write(args.output/'status.json', dict(status='complete_development_only', pid=os.getpid(),
            unix=time.time(), qualified_for_final_hybrid_table=False))
    except BaseException as error:
        write(args.output/'status.json', dict(status='failed', pid=os.getpid(), unix=time.time(), error=repr(error)))
        raise
    finally:
        guard.close(); adapter.adapt = original_adapt; lowlevel.LowLevelTape = original_tape
        live.LiveFront.__init__=original_front_init


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
    reference = same_card_reference() if args.same_card_reference else None
    if args.root.exists():raise FileExistsError('Preserve prior development root; no retry')
    args.root.mkdir()
    write(args.root/'source_receipt.json', dict(commit=commit, sources=sources))
    if reference:write(args.root/'development_reference_receipt.json', reference)
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
                HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS=str(args.cpu_threads),
                MKL_NUM_THREADS=str(args.cpu_threads), OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
            subprocess.run(['nvidia-cuda-mps-control', '-d'], env=env, capture_output=True,
                text=True, timeout=10, check=True); active = True
            plan = ([('patch_conv2d_channels', False)] if args.patch_conv2d_channels else
                [('patch_conv2d', False)] if args.patch_conv2d else
                [('patch_linear_separate_bias', False)] if args.patch_linear_separate_bias else
                [('patch_linear', False)] if args.patch_linear else
                [('parallel_text', False)] if args.parallel_text else
                [('conv_layout', False)] if args.channels_last_3d else
                [('prompt_render', False)] if args.once_prompt_render else
                [('cpu_threads1', False)] if args.cpu_threads==1 else
                [('graph_pool', False)] if args.share_graph_pools else
                [('event_pool', True)] if reference else [('baseline', False), ('event_pool', True)])
            for label, reuse in plan:
                if time.time() >= args.deadline-60:raise TimeoutError('No new round beyond deadline')
                if source_map() != sources:raise RuntimeError('Development source changed')
                snapshot(args.pipe, -1)
                leaf = args.root/label
                command = [sys.executable, '-u', str(ENTRY), '--phase', 'worker', '--root', str(args.root),
                    '--output', str(leaf), '--deadline', str(args.deadline), '--receipt-sha', receipt_sha]
                if reuse:command.append('--reuse-events')
                if reference:command.append('--same-card-reference')
                if args.share_graph_pools:command.append('--share-graph-pools')
                if args.software_only_development:command.append('--software-only-development')
                command.extend(['--cpu-threads',str(args.cpu_threads)])
                if args.once_prompt_render:command.append('--once-prompt-render')
                if args.channels_last_3d:command.append('--channels-last-3d')
                if args.parallel_text:command.append('--parallel-text')
                if args.patch_linear:command.append('--patch-linear')
                if args.patch_linear_separate_bias:command.append('--patch-linear-separate-bias')
                if args.patch_conv2d:command.append('--patch-conv2d')
                if args.patch_conv2d_channels:command.append('--patch-conv2d-channels')
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
    p.add_argument('--same-card-reference', action='store_true',
        help='Explicit human authorization: event-pool round only, bytewise to preserved RTX4090 baseline')
    p.add_argument('--share-graph-pools', action='store_true')
    p.add_argument('--cpu-threads', type=int, choices=[1,8], default=8)
    p.add_argument('--once-prompt-render', action='store_true')
    p.add_argument('--channels-last-3d', action='store_true')
    p.add_argument('--parallel-text', action='store_true',
        help='Original same-field text arithmetic on its own stream; fork after H2D and join before use')
    p.add_argument('--patch-linear', action='store_true',
        help='Exact one-complete-patch biased convolution as matrix multiplication; original byte gate retained')
    p.add_argument('--patch-linear-separate-bias', action='store_true',
        help='Keep original cuDNN BF16 output rounding before separate bias addition')
    p.add_argument('--patch-conv2d', action='store_true',
        help='Fold complete-patch temporal/height axes for native biased convolution, no GEMM substitution')
    p.add_argument('--patch-conv2d-channels', action='store_true',
        help='Fold adjacent channel/temporal axes to six channels and 16x16 native convolution')
    p.add_argument('--software-only-development', action='store_true',
        help='Keep rejected CUDA clock evidence; report CPU-wall development only, never a repaired DAG projection')
    args = p.parse_args(); args.root = args.root.resolve()
    if args.root.parent != SIM or not re.fullmatch(r'lgvq_rtx4090_development_20261027_\d{2}', args.root.name):
        raise ValueError('New canonical development root required')
    if not math.isfinite(args.deadline) or not 0 < args.deadline-time.time() <= 3600:
        raise ValueError('Finite development deadline at most one hour required')
    args.pipe = '/DATA/DATA1/guest3/.cache/'+args.root.name
    args.optimize_runtime = True
    if args.patch_conv2d_channels and (not args.same_card_reference or args.cpu_threads!=8 or args.share_graph_pools
            or args.reuse_events or args.once_prompt_render or args.channels_last_3d or args.parallel_text
            or args.patch_linear or args.patch_linear_separate_bias or args.patch_conv2d):
        raise ValueError('Channel-fold Conv2d candidate changes only complete-patch convolution geometry')
    if args.patch_conv2d and (not args.same_card_reference or args.cpu_threads!=8 or args.share_graph_pools
            or args.reuse_events or args.once_prompt_render or args.channels_last_3d or args.parallel_text
            or args.patch_linear or args.patch_linear_separate_bias):
        raise ValueError('Conv2d-fold candidate changes only complete-patch native convolution geometry')
    if args.patch_linear_separate_bias and (not args.same_card_reference or args.cpu_threads!=8 or args.share_graph_pools
            or args.reuse_events or args.once_prompt_render or args.channels_last_3d or args.parallel_text or args.patch_linear):
        raise ValueError('Separate-bias patch candidate retains original rounding boundary, no other method')
    if args.patch_linear and (not args.same_card_reference or args.cpu_threads!=8 or args.share_graph_pools
            or args.reuse_events or args.once_prompt_render or args.channels_last_3d or args.parallel_text):
        raise ValueError('Patch-linear candidate changes only the complete-patch convolution implementation')
    if args.parallel_text and (not args.same_card_reference or args.cpu_threads!=8 or args.share_graph_pools
            or args.reuse_events or args.once_prompt_render or args.channels_last_3d):
        raise ValueError('Text-stream candidate changes only same-field dependency scheduling')
    if args.channels_last_3d and (not args.same_card_reference or args.cpu_threads!=8 or args.share_graph_pools or args.reuse_events or args.once_prompt_render):
        raise ValueError('Conv-layout candidate changes only original Conv3d storage format')
    if args.once_prompt_render and (not args.same_card_reference or args.cpu_threads!=8 or args.share_graph_pools or args.reuse_events):
        raise ValueError('Prompt-render candidate changes only constant TEXT render frequency')
    if args.share_graph_pools and (not args.same_card_reference or args.reuse_events):
        raise ValueError('Graph-pool candidate requires same-card reference and original non-reused events')
    if args.cpu_threads==1 and (not args.same_card_reference or args.share_graph_pools or args.reuse_events):
        raise ValueError('CPU-thread candidate requires same-card reference with original graph/event allocation')
    if args.software_only_development and not (args.patch_conv2d_channels or args.patch_conv2d or args.patch_linear_separate_bias or args.patch_linear or args.parallel_text or args.share_graph_pools or args.cpu_threads==1 or args.once_prompt_render or args.channels_last_3d):
        raise ValueError('Explicit software-only scope is limited to new authorized development candidates')
    if args.phase == 'worker':
        if args.output is None or args.output.resolve() not in [args.root/x for x in ('baseline','event_pool','graph_pool','cpu_threads1','prompt_render','conv_layout','parallel_text','patch_linear','patch_linear_separate_bias','patch_conv2d','patch_conv2d_channels')]:
            raise ValueError('Exact development leaf required')
        args.output = args.output.resolve()
        if args.reuse_events != (args.output.name == 'event_pool'):raise ValueError('Frozen A/B plan differs')
        if args.share_graph_pools != (args.output.name == 'graph_pool'):raise ValueError('Frozen graph-pool plan differs')
        if (args.cpu_threads==1) != (args.output.name=='cpu_threads1'):raise ValueError('Frozen CPU-thread plan differs')
        if args.once_prompt_render != (args.output.name=='prompt_render'):raise ValueError('Frozen prompt-render plan differs')
        if args.channels_last_3d != (args.output.name=='conv_layout'):raise ValueError('Frozen Conv-layout plan differs')
        if args.parallel_text != (args.output.name=='parallel_text'):raise ValueError('Frozen text-stream plan differs')
        if args.patch_linear != (args.output.name=='patch_linear'):raise ValueError('Frozen patch-linear plan differs')
        if args.patch_linear_separate_bias != (args.output.name=='patch_linear_separate_bias'):raise ValueError('Frozen separate-bias patch plan differs')
        if args.patch_conv2d != (args.output.name=='patch_conv2d'):raise ValueError('Frozen Conv2d-fold patch plan differs')
        if args.patch_conv2d_channels != (args.output.name=='patch_conv2d_channels'):raise ValueError('Frozen channel-fold patch plan differs')
    (suite if args.phase == 'suite' else worker)(args)


if __name__ == '__main__':main()
