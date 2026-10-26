"""Missing MPS100 triples after the running MPS50 suite releases its lock.

Only resource labels/assertions change in the qualified original worker AST.
No native model math, precision, input, ordering or cost boundary changes.
"""
import ast
import inspect
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import rental_energy_triples_20261026 as energy


def resource_worker(source):
    tree = ast.parse(source)
    counts = dict(cap=0, environment=0, SM=0)
    class ResourceOnly(ast.NodeTransformer):
        def visit_Constant(self, node):
            if type(node.value) is int and node.value == 50:
                counts['cap'] += 1
                return ast.copy_location(ast.Constant(100), node)
            if type(node.value) is int and node.value == 54:
                counts['SM'] += 1
                return ast.copy_location(ast.Constant(108), node)
            if node.value == '50':
                counts['environment'] += 1
                return ast.copy_location(ast.Constant('100'), node)
            return node
    tree = ResourceOnly().visit(tree)
    if counts != dict(cap=2, environment=1, SM=1):
        raise ValueError('Original resource contract changed: '+repr(counts))
    return ast.fix_missing_locations(tree), counts


def worker(args):
    original = energy.original
    before = original.worker
    tree, counts = resource_worker(inspect.getsource(before))
    namespace = dict(before.__globals__)
    # Resource-only compilation has its own globals dictionary. Resolve the
    # scoped observer/source wrapper dynamically rather than retaining the old
    # function object copied before energy.worker installs its source binding.
    namespace['sources'] = lambda: original.sources()
    exec(compile(tree, __file__+'::resource_only', 'exec'), namespace)
    original.worker = namespace['worker']
    before_sources = original.sources
    own = {Path(__file__).relative_to(energy.REPO).as_posix():original.sha(__file__)}
    original.sources = lambda: dict(before_sources(), **own)
    before_summary = energy.summarize
    def summary(*a, **kw):
        result = before_summary(*a, **kw)
        result.update(cap=100, resource_only_AST_changes=counts)
        return result
    energy.summarize = summary
    try:
        energy.worker(args)
    finally:
        original.worker, original.sources = before, before_sources
        energy.summarize = before_summary


def suite(args):
    import fcntl
    original = energy.original
    root = args.root
    root.mkdir()
    state = dict(status='waiting_previous_MPS50_release', pid=os.getpid(), deadline=args.deadline,
        source_commit=os.environ['MEASUREMENT_PUBLISHED_COMMIT'], source_sha256=original.sha(__file__),
        previous=str(args.previous), cap=100, completed=[], plan=[(m,r) for r in (1,2,3) for m in ('kimi','deepseek')])
    def save():energy.write(root/'suite_status.json', state)
    save()
    child = None
    active = False
    try:
        while True:
            previous = json.loads((args.previous/'suite_status.json').read_text())
            if previous['status'] == 'complete':
                proc = Path('/proc')/str(previous['pid'])/'stat'
                if not proc.exists() or proc.read_text().split(') ',1)[1].startswith('Z'):
                    break
            elif previous['status'] == 'stopped':
                raise RuntimeError('Preserve preceding failure; do not skip incomplete MPS50 suite')
            if time.time() >= args.deadline:
                raise TimeoutError('Finite previous-suite wait')
            time.sleep(3)
        with (original.TASK/'runs/simulation/.temporal_telemetry_a100.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            for _ in range(12):
                if original.gpu_processes():raise RuntimeError('Foreign GPU process')
                time.sleep(.25)
            if original.control('get_server_list').returncode == 0:raise RuntimeError('Existing private MPS daemon')
            Path(original.PIPE).mkdir(exist_ok=False)
            (root/'mps_logs').mkdir()
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=original.GPU, CUDA_DEVICE_ORDER='PCI_BUS_ID',
                CUDA_MPS_PIPE_DIRECTORY=original.PIPE, CUDA_MPS_LOG_DIRECTORY=str(root/'mps_logs'),
                CUDA_MPS_ACTIVE_THREAD_PERCENTAGE='100', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', TOKENIZERS_PARALLELISM='false')
            subprocess.run(['nvidia-cuda-mps-control','-d'],env=env,check=True);active=True
            for model, repeat in state['plan']:
                if time.time()>=args.deadline-600:raise TimeoutError('Reserve backup margin')
                leaf=root/f'{model}100_repeat{repeat:02d}'
                command=['/data/venvs/'+model+'/bin/python','-u',str(Path(__file__)),'--phase','worker',
                    '--root',str(root),'--output',str(leaf),'--model',model,'--repeat',str(repeat),
                    '--deadline',str(args.deadline),'--common-tokens',str(args.common_tokens)]
                energy.write(root/(leaf.name+'.command.json'),command)
                with (root/(leaf.name+'.log')).open('x') as log:
                    child=subprocess.Popen(command,cwd=energy.REPO,env=env,stdin=subprocess.DEVNULL,
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    state.update(status='running',phase=leaf.name,child=child.pid);save()
                    while child.poll() is None:
                        if time.time()>=args.deadline:raise TimeoutError('Finite continuation deadline')
                        time.sleep(1)
                if child.returncode!=0:raise RuntimeError('Preserve failed point; no blind retry')
                state['completed'].append(leaf.name);save()
            state.update(status='complete',finished=time.time())
    except BaseException as error:
        state.update(status='stopped',error=repr(error))
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGINT);child.wait(timeout=60)
        raise
    finally:
        if active:
            result=original.control('quit');state.update(mps_release_returncode=result.returncode,
                mps_release_stdout=result.stdout,mps_release_stderr=result.stderr)
        save()


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['suite','worker'],required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--previous',type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--model',choices=['kimi','deepseek'])
    p.add_argument('--repeat',type=int,choices=[1,2,3]);p.add_argument('--common-tokens',type=int,required=True)
    p.add_argument('--deadline',type=float,required=True)
    args=p.parse_args();args.root=args.root.resolve()
    if energy.REPO!=Path('/data/2026OpticsMoE') or args.root.parent!=energy.original.TASK/'runs/simulation':
        raise ValueError('Authorized rental directory only')
    if not args.root.name.startswith('a10080_energy_cap100_20261026_') or not 0<args.deadline-time.time()<=36000:
        raise ValueError('New finite MPS100 continuation required')
    energy.bind(args.root)
    if args.phase=='suite':suite(args)
    else:
        args.output=args.output.resolve()
        if args.output.parent!=args.root:raise ValueError('Wrong leaf')
        worker(args)


if __name__=='__main__':main()
