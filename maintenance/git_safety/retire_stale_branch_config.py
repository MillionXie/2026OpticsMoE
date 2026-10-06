"""Remove only archived, absent branch configuration; never remove refs/files.

Default is a read-only plan. --apply requires a fresh private backup directory.
Configuration backup may contain secrets and must never be committed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def git(*args, optional=False):
    result = subprocess.run(['git', *args], capture_output=True, check=False)
    if result.returncode and not (optional and result.returncode == 1):
        raise RuntimeError('Git inspection/config operation failed')
    return result.stdout


def snapshot():
    return {name: hashlib.sha256(git(*args)).hexdigest() for name, args in {
        'head': ('rev-parse', 'HEAD'),
        'refs': ('for-each-ref', '--format=%(refname) %(objectname)'),
        'worktrees': ('worktree', 'list', '--porcelain'),
        'tracked_status': ('status', '--porcelain', '--untracked-files=no'),
        'index': ('ls-files', '--stage', '-z'),
    }.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--backup-dir', type=Path)
    parser.add_argument('--aliases', type=Path, help='Audited old branch to exact existing archive ref/commit mapping')
    args = parser.parse_args()
    config_path = Path(git('rev-parse', '--git-common-dir').decode().strip()) / 'config'
    raw = config_path.read_bytes()
    heads = set(git('for-each-ref', '--format=%(refname:strip=2)', 'refs/heads').decode().splitlines())
    worktrees = git('worktree', 'list', '--porcelain').decode()
    used = {line[len('branch refs/heads/'):] for line in worktrees.splitlines()
            if line.startswith('branch refs/heads/')}
    refs = git('for-each-ref', '--format=%(refname) %(objectname)', 'refs/archive').decode().splitlines()
    aliases = {}
    if args.aliases:
        for item in json.loads(args.aliases.read_text(encoding='utf8'))['aliases']:
            name, ref, commit = item['branch'], item['archive_ref'], item['commit']
            if name in aliases or not ref.startswith('refs/archive/') or not item.get('evidence'):
                raise ValueError('Ambiguous or unscoped alias')
            if ref + ' ' + commit not in refs:
                raise ValueError('Archive alias commit does not match an existing reference')
            aliases[name] = ref + ' ' + commit
    keys = git('config', '--local', '--name-only', '--get-regexp', '^branch\.', optional=True).decode().splitlines()
    names = {key[len('branch.'):].rsplit('.', 1)[0] for key in keys}
    candidates, unbound = [], []
    for name in sorted(names - heads - used):
        matches = [line for line in refs if line.split(' ', 1)[0].endswith('/' + name)]
        if name in aliases:
            matches.append(aliases[name])
        if not matches:
            unbound.append(name)
        else:
            candidates.append({'branch': name, 'archive_identities': matches})
    before = snapshot()
    settings_before = git('config', '--local', '--null', '--list').split(b'\0')
    report = {'read_only': not args.apply, 'kept_branches': sorted(heads),
              'candidates': candidates, 'unbound_preserved': unbound,
              'config_before_sha256': hashlib.sha256(raw).hexdigest(),
              'code_data_refs_removed': False}
    if args.apply:
        if not args.backup_dir:
            raise ValueError('--backup-dir required')
        private = Path('.codex_tmp').resolve()
        destination = args.backup_dir.resolve()
        if private not in destination.parents:
            raise ValueError('Backup must be a fresh directory below .codex_tmp')
        if snapshot() != before or config_path.read_bytes() != raw:
            raise RuntimeError('Repository changed before operation')
        destination.mkdir(parents=True, exist_ok=False)
        shutil.copy2(config_path, destination / 'config.before.private')
        if (destination / 'config.before.private').read_bytes() != raw:
            raise RuntimeError('Backup differs')
        # Git takes its normal config lock; do not rewrite the entire config.
        for item in candidates:
            if snapshot() != before:
                raise RuntimeError('Concurrent repository mutation; stop and retain backup')
            git('config', '--local', '--remove-section', 'branch.' + item['branch'])
        if snapshot() != before:
            raise RuntimeError('Repository state changed during config cleanup')
        remaining = git('config', '--local', '--name-only', '--get-regexp', '^branch\.', optional=True).decode().splitlines()
        expected = [key for key in keys if key[len('branch.'):].rsplit('.', 1)[0]
                    not in {item['branch'] for item in candidates}]
        if remaining != expected:
            raise RuntimeError('Remaining branch configuration keys differ')
        removed_prefixes = [b'branch.' + item['branch'].encode() + b'.' for item in candidates]
        expected_settings = [entry for entry in settings_before
                             if not any(entry.split(b'\n', 1)[0].startswith(prefix)
                                        for prefix in removed_prefixes)]
        if git('config', '--local', '--null', '--list').split(b'\0') != expected_settings:
            raise RuntimeError('Other configuration changed; retain private backup')
        report.update(config_after_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                      preserved_state=before, removed_sections=len(candidates))
        (destination / 'receipt.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
