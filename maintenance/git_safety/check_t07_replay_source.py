"""Read-only computational-graph audit; private archive audit is explicit."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess


MANIFEST = 'maintenance/storage/T07_OFFLINE_REPLAY_SOURCE_20261004.json'
REPLAY = 'LightGenV2/tasks/t07_abo_image_retrieval/hardware/replay.py'


def function(tree, name):
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def graph(node):
    # Validate the numerical/injection body, excluding the explicit-path wrapper.
    start = next(i for i, child in enumerate(node.body) if isinstance(child, ast.With))
    return ast.Module(body=node.body[start:], type_ignores=[])


class CaptureAdapter(ast.NodeTransformer):
    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == 'capture_stage':
            assert len(node.args) == 7 and not node.keywords
            # Old: bench, out, stage, phase_path, active, ids, orientation.
            node.args = [node.args[index] for index in (2, 4, 5)]
        return node


def identity(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def check(root, commit, audit_archive=False):
    def read(ref, path):
        return subprocess.check_output(['git', '-C', str(root), 'show', ref + ':' + path])
    receipt = json.loads(read(commit, MANIFEST))
    raw = read(commit, REPLAY).replace(b'\r\n', b'\n')
    assert hashlib.sha256(raw).hexdigest() == receipt['published_replay_sha256']
    current = ast.parse(raw)
    assert identity(graph(function(current, 'replay_batch'))) == receipt['adapted_graph_ast_sha256']
    for name in ('pcc', 'snapshot_simulation'):
        assert identity(function(current, name)) == receipt['unchanged_function_ast_sha256'][name]
    if audit_archive:
        originals = {}
        for item in receipt['sources']:
            original = read(receipt['archive_commit'], item['path'])
            assert hashlib.sha256(original).hexdigest() == item['sha256']
            originals[item['role']] = ast.parse(original)
        adapted = CaptureAdapter().visit(graph(function(originals['pipeline'], 'process_batch')))
        assert identity(adapted) == receipt['adapted_graph_ast_sha256']
        for name in ('pcc', 'snapshot_simulation'):
            assert identity(function(originals['geometry'], name)) == receipt['unchanged_function_ast_sha256'][name]
    assert receipt['dataset_queries_evaluated'] is False
    assert receipt['hardware_acquisition_migration_complete'] is False
    return dict(commit=commit, published_graph_identity_checked=True,
                original_archive_graph_compared=audit_archive, devices_opened=False,
                full_hardware_runner_migration_claimed=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit', default='main')
    parser.add_argument('--audit-archive', action='store_true')
    args = parser.parse_args()
    print(json.dumps(check(Path(__file__).resolve().parents[2], args.commit, args.audit_archive), indent=2))
