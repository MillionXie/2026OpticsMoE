"""Check preserved T12 tools in a Git tree; no training/data/model downloads."""
import argparse
import contextlib
import hashlib
import importlib
import inspect
import io
import json
import os
from pathlib import Path
import subprocess
import sys


def run(repository, commit):
    def blob(path):
        return subprocess.check_output(['git', '-C', str(repository), 'show', commit+':'+path])
    helper = 'maintenance/git_safety/check_t08_reverse_backend.py'
    namespace = {'__name__': 'git_tree_importer'}
    exec(compile(blob(helper), helper, 'exec'), namespace)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.dont_write_bytecode = True
    importer = namespace['GitImporter'](repository, commit, Path('/__git_memory_only__'))
    sys.meta_path.insert(0, importer)
    checked = 0
    for name in ('T12_PRESERVED_ENTRY_ADDITIONS_20261004.json',
                 'T12_PRESERVED_ENTRY_DEPENDENCIES_20261004.json'):
        manifest = json.loads(blob('maintenance/storage/'+name))
        for row in manifest['paths']:
            raw = blob(row['path'])
            assert len(raw) == row['bytes']
            assert hashlib.sha256(raw).hexdigest() == row['sha256']
            runtime = Path(manifest['runtime_root'])/row['path']
            original = (runtime.read_bytes() if runtime.is_file() else
                        blob_from_ref(repository, manifest['source_commit'], row['path']))
            assert raw == original
            checked += 1
    task = 'LightGenV2.tasks.t12_text_to_image.'
    training = importlib.import_module(task+'product_global_redesign_training')
    small = importlib.import_module(task+'qwen_mini_small')
    cli = importlib.import_module(task+'qwen_mini_small_run')
    assert callable(training.cache_redesign_latents)
    calls = []
    def train_stub(**kwargs):
        inspect.signature(small.train_qwen_mini_editor).bind(**kwargs)
        calls.append(kwargs)
        return {'contract_only': True}
    def cache_stub(**kwargs):
        inspect.signature(small.build_qwen_embedding_cache).bind(**kwargs)
        calls.append(kwargs)
        return {'contract_only': True}
    cli.train_qwen_mini_editor, cli.build_qwen_embedding_cache = train_stub, cache_stub
    previous_argv = sys.argv
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            sys.argv = ['entry', 'train', '--task', 'unified_expanded', '--data-dir', 'fixture',
                        '--instruction-cache', 'fixture', '--embedding-cache', 'fixture',
                        '--output-dir', 'fixture', '--device', 'cpu', '--learned-source-gate',
                        '--source-gate-loss-weight', '.2', '--image-size', '256']
            assert cli.main() == 0
            sys.argv = ['entry', 'cache', '--instruction-cache', 'fixture',
                        '--qwen-checkpoint', 'fixture', '--output', 'fixture', '--device', 'cpu']
            assert cli.main() == 0
    finally:
        sys.argv = previous_argv
        sys.meta_path.remove(importer)
    assert len(calls) == 2
    assert calls[0]['learned_source_gate'] and calls[0]['source_gate_loss_weight'] == .2
    return dict(commit=commit, exact_server_source_entries=checked, cli_signature_contracts=2,
                imports_from_candidate=importer.loaded, training_invoked=False,
                datasets_read=False, model_weights_loaded=False, hardware_touched=False)


def blob_from_ref(repository, ref, path):
    return subprocess.check_output(['git', '-C', str(repository), 'show', ref+':'+path])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.repository, args.commit), indent=2))
