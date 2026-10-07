"""Opt-in repository-local pre-commit guard; never replaces an existing hook."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def inspect(root):
    root = Path(git(Path(root).resolve(), 'rev-parse', '--show-toplevel'))
    head = git(root, 'rev-parse', 'HEAD')
    branch = git(root, 'branch', '--show-current')
    proc = subprocess.run(['git', '-C', str(root), 'config', '--get', 'core.hooksPath'],
                          capture_output=True, text=True)
    if proc.returncode not in (0, 1):
        raise RuntimeError('Cannot inspect existing hooksPath')
    configured = proc.stdout.strip() or None
    if configured not in (None, '.githooks'):
        raise ValueError('Existing hooksPath must be preserved; no automatic replacement')
    python_proc = subprocess.run(['git', '-C', str(root), 'config', '--local', '--get',
                                  'lightgen.guardPython'], capture_output=True, text=True)
    if python_proc.returncode not in (0, 1):
        raise RuntimeError('Cannot inspect existing guard interpreter')
    default_hook = Path(git(root, 'rev-parse', '--git-path', 'hooks/pre-commit'))
    if not default_hook.is_absolute():
        default_hook = root / default_hook
    if configured is None and (default_hook.exists() or default_hook.is_symlink()):
        raise ValueError('Existing pre-commit hook must be preserved')
    hook = root / '.githooks/pre-commit'
    guard = root / 'maintenance/git_safety/review_git.py'
    if not hook.is_file() or not guard.is_file():
        raise ValueError('Published hook and staged guard must exist in this checkout')
    if branch != 'main':
        raise ValueError('Install from the canonical main checkout only')
    return {'root': str(root), 'head': head, 'branch': branch,
            'previous_local_hooks_path': configured,
            'configured_python': python_proc.stdout.strip() or None,
            'hook_sha256': hashlib.sha256(hook.read_bytes()).hexdigest(),
            'guard_sha256': hashlib.sha256(guard.read_bytes()).hexdigest(),
            'installed': configured == '.githooks',
            'scope': 'Repository-local config; old worktrees without .githooks do not gain this guard',
            'limitations': ['Bypassable with --no-verify or config changes',
                           'Only staged artifacts and narrow credential patterns are checked',
                           'No branch creation, concurrent-writer or runtime-device lock']}


def install(root, expected_head):
    state = inspect(root)
    if state['head'] != expected_head:
        raise ValueError('HEAD changed; do not alter shared Git configuration')
    if git(root, 'diff', '--cached', '--name-only'):
        raise ValueError('Index is occupied; finish the current Git transaction first')
    interpreter = str(Path(sys.executable).resolve())
    if state['configured_python'] not in (None, interpreter):
        raise ValueError('Existing guard interpreter must be preserved; inspect manually')
    # Bind the interpreter that actually passed the guard, not SSH/VSCode PATH.
    subprocess.check_call([interpreter, str(Path(state['root']) / 'maintenance/git_safety/review_git.py'),
                           '--repo', state['root'], '--check-staged'], stdout=subprocess.DEVNULL)
    subprocess.check_call(['git', '-C', state['root'], 'config', '--local',
                           'lightgen.guardPython', interpreter])
    subprocess.check_call(['git', '-C', state['root'], 'config', '--local',
                           'core.hooksPath', '.githooks'])
    result = inspect(root)
    if result['head'] != expected_head or not result['installed']:
        raise RuntimeError('Guard installation verification failed')
    result['configuration_changed'] = state['previous_local_hooks_path'] != '.githooks'
    result['interpreter_configuration_changed'] = state['configured_python'] != interpreter
    result['previous_configured_python'] = state['configured_python']
    result['working_files_refs_and_index_changed'] = False
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', default='.')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--expected-head')
    args = parser.parse_args()
    if args.apply and not args.expected_head:
        parser.error('--apply requires an explicitly verified --expected-head')
    print(json.dumps(install(args.repo, args.expected_head) if args.apply else inspect(args.repo), indent=2))


if __name__ == '__main__':
    main()
