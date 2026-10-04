"""Generate pinned T09 main additions/replacement plans; no checkout changes."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
SOURCE = '190ceb9da7add8d93ea75c3086ace9ce58019d71'
TASK = 'LightGenV2/tasks/t09_multimodal_matching/'


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])


def main():
    main_head = git('rev-parse', 'main').decode().strip()
    tree = git('ls-tree', '-r', '--name-only', SOURCE, '--', TASK).decode().splitlines()
    paths = [p for p in tree if p.endswith('.py') or p in
             (TASK+'README.md', TASK+'reports/reproduction/README.md')]
    paths += ['LightGenV2/demo_check/pure_optical/models.py',
              'LightGenV2/demo_check/pure_optical/config.json',
              'LightGenV2/demo_check/adrenal_softsign_code_export_20260915_145336/code/optical_reference/__init__.py',
              'LightGenV2/demo_check/adrenal_softsign_code_export_20260915_145336/code/optical_reference/optics.py']
    additions, replacements, identical = [], [], []
    for path in sorted(paths):
        raw = git('show', SOURCE+':'+path)
        digest = hashlib.sha256(raw).hexdigest()
        if path.endswith('.py'):
            compile(raw.decode(), path, 'exec')
        target = git('ls-tree', 'main', '--', path)
        if target:
            old = git('show', 'main:'+path)
            if old == raw:
                identical.append(path)
                continue
            if path != TASK+'vision.py':
                raise RuntimeError('Unexpected differing main dependency: '+path)
            if hashlib.sha256(old).hexdigest() != '630ad5273911f8f6069602788eb5f71cd9ac2f0e9a0040b7c1211a694376761f':
                raise RuntimeError('Changed old vision identity')
            proof = json.loads((ROOT/'maintenance/storage/T09_T16_VISION_COMPATIBILITY_20261004.json').read_text())
            if (proof['overlay_vision_sha256'] != digest
                    or not proof['formal_checkpoint_compatibility']['old_and_new_strict_load']
                    or not proof['formal_checkpoint_compatibility']['synthetic_outputs_identical']):
                raise RuntimeError('Missing formal shared-front compatibility')
            replacements.append(dict(path=path, source_sha256=digest,
                                     expected_target_sha256=hashlib.sha256(old).hexdigest(),
                                     review_reason='Actual server widths metadata support; default state/output exact and actual T16 shared frontend PT strict compatibility verified.'))
        else:
            additions.append(dict(path=path, sha256=digest, bytes=len(raw)))
    outputs = {
        'T09_RUNTIME_ADDITIONS_20261004.json': dict(source_commit=SOURCE, paths=additions,
                                                 existing_identical_dependencies=identical,
                                                 source_runtime='/DATA/DATA1/guest3/demo_reproduction_20260915'),
        'T09_VISION_REPLACEMENT_20261004.json': dict(source_commit=SOURCE, expected_main=main_head,
                                                  task_prefix=TASK, paths=replacements),
    }
    for name, payload in outputs.items():
        target = ROOT/'maintenance/storage'/name
        if target.exists():
            raise FileExistsError(target)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(additions=len(additions), replacements=len(replacements),
                         unchanged_dependencies=identical, main=main_head,
                         candidate_prepared=False, runtime_verified=False)))


if __name__ == '__main__':
    main()
