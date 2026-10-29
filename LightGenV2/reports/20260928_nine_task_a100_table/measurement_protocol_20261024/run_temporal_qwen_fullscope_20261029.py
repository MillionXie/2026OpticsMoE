"""New Qwen-only 512x3 cold B1 file-to-CPU-score protocol.

Reuse the frozen native inference, power sampler, exclusive guard and independent
audit from the qualified 200-video entry. Adapt only sample-count normalization,
new immutable output root, published source identity and child entry. No neural
forward, precision, preprocessing, sampling, warmup or power gate is changed.
"""
import argparse
import ast
import csv
import hashlib
import inspect
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time

ENTRY = Path(__file__).resolve()
HERE = ENTRY.parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))
from LightGenV2.tasks.t06_video_quality_assessment import measure_mps_e2e_20261024 as base

NEW_PROTOCOL = 'qwen_cold_file_to_cpu_score_B1_MPS50_512x3_20261029'
COUNT = 512
GPU = 'GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55'
ORIGINAL_SOURCES = base.source_map
ORIGINAL_CONFIG = base.config


def source_map():
    return {**ORIGINAL_SOURCES(), ENTRY.relative_to(REPO).as_posix(): base.sha(ENTRY)}


def input_rows(manifest):
    config = ORIGINAL_CONFIG()
    if base.sha(manifest) != config['manifest_sha256']:
        raise ValueError('Original manifest differs')
    with Path(manifest).open(newline='', encoding='utf-8-sig') as stream:
        rows = [row for row in csv.DictReader(stream) if row['split'] == 'test']
    selected = rows[:COUNT]
    if (len(rows) != 558 or len(selected) != COUNT
            or len({r['sample_id'] for r in selected}) != COUNT
            or base.ordered_id_sha(selected[:200]) != config['first200_ids_sha256']):
        raise ValueError('Original ordered first512 TEST videos required')
    return selected


def adapt(original, power_only=False):
    """Scope adapter only: native model/preparation/forward functions untouched."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    class Scope(ast.NodeTransformer):
        def visit_Constant(self, node):
            if type(node.value) is int and node.value in (200, 800):
                return ast.copy_location(ast.Constant({200: COUNT, 800: COUNT * 4}[node.value]), node)
            if isinstance(node.value, str):
                return ast.copy_location(ast.Constant(node.value.replace('200', str(COUNT))), node)
            return node

        def visit_keyword(self, node):
            self.generic_visit(node)
            if node.arg:
                node.arg = node.arg.replace('200', str(COUNT))
            if node.arg == 'protocol_id' and isinstance(node.value, ast.Name) and node.value.id == 'PROTOCOL':
                node.value = ast.Name(id='NEW_PROTOCOL', ctx=ast.Load())
            return node

        def visit_Compare(self, node):
            self.generic_visit(node)
            # The original audit compares report protocol to legacy config.
            if ast.unparse(node).startswith("report['protocol_id'] != PROTOCOL"):
                node.comparators[0] = ast.Name(id='NEW_PROTOCOL', ctx=ast.Load())
            return node

        def visit_Call(self, node):
            self.generic_visit(node)
            # New explicitly labelled power-only acquisition has no utilization
            # or memory observations. Keep them missing, never synthesize zeros.
            if power_only and ast.unparse(node) in (
                    "float(r['utilization_percent'])", "float(r['memory_used_mib'])"):
                return ast.copy_location(ast.Constant(None), node)
            if ast.unparse(node) == "subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()":
                return ast.copy_location(ast.parse("os.environ['MEASUREMENT_PUBLISHED_COMMIT']", mode='eval').body, node)
            return node

        def visit_Assign(self, node):
            self.generic_visit(node)
            if (original.__name__ == 'suite' and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'command'):
                old = node.value
                if not isinstance(old, ast.List) or old.elts[2].value != '-m':
                    raise ValueError('Original worker command AST changed')
                node.value = ast.List(elts=old.elts[:2] + [ast.parse('str(ENTRY)', mode='eval').body] + old.elts[4:], ctx=ast.Load())
            return node
    tree = ast.fix_missing_locations(Scope().visit(tree))
    namespace = dict(original.__globals__, COUNT=COUNT, NEW_PROTOCOL=NEW_PROTOCOL, ENTRY=ENTRY)
    exec(compile(tree, str(ENTRY) + '::' + original.__name__, 'exec'), namespace)
    return namespace[original.__name__]


def configure(root):
    base.CANONICAL = root
    base.source_map = source_map
    base.input_rows = input_rows
    base.summarize = adapt(base.summarize)
    base.audit_point = adapt(base.audit_point)
    base.worker = adapt(base.worker)
    base.suite = adapt(base.suite)
    # Adapted functions need the configured shared namespace, not copied legacy
    # summarize/input_rows/worker/audit references captured before adaptation.
    for function in (base.summarize, base.audit_point, base.worker, base.suite):
        function.__globals__.update(vars(base))
        function.__globals__.update(NEW_PROTOCOL=NEW_PROTOCOL, ENTRY=ENTRY)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('suite', 'worker', 'audit'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--suite-root', type=Path)
    parser.add_argument('--deadline', type=float)
    parser.add_argument('--profile', default='lab', choices=('lab',))
    parser.add_argument('--model', default='qwen', choices=('qwen',))
    parser.add_argument('--cap', type=int, default=50, choices=(50,))
    parser.add_argument('--repeat', type=int, default=1, choices=(1, 2, 3))
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--reproduction-receipt', type=Path)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--gpu-uuid', default=GPU, choices=(GPU,))
    parser.add_argument('--pipe')
    args = parser.parse_args()
    args.output = args.output.resolve()
    root = (args.suite_root or args.output).resolve()
    if (root.parent != base.TASK / 'runs/simulation'
            or not root.name.startswith('qwen_fullscope_20261029_')
            or (args.phase != 'suite' and args.output.parent != root)):
        raise ValueError('New canonical Qwen-only run required')
    args.suite_root = root
    args.pipe = args.pipe or '/DATA/DATA1/guest3/.cache/' + root.name
    args.models = ['qwen']; args.caps = [50]; args.repeats = [1, 2, 3]
    if args.phase != 'audit' and (args.deadline is None or not 0 < args.deadline - time.time() <= 28800):
        raise ValueError('Finite deadline required')
    configure(root)
    if args.phase == 'audit':
        import json
        print(json.dumps(base.audit_point(args.output, args.manifest), ensure_ascii=True))
        return
    commit = os.environ['MEASUREMENT_PUBLISHED_COMMIT']
    for name, digest in source_map().items():
        blob = subprocess.check_output(['git', 'show', commit + ':' + name.replace('\\', '/')], cwd=REPO, timeout=10)
        if hashlib.sha256(blob).hexdigest() != digest:
            raise ValueError('Published source differs: ' + name)
    (base.suite if args.phase == 'suite' else base.worker)(args)


if __name__ == '__main__':
    main()
