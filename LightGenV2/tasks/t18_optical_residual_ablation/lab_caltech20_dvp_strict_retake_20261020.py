"""Accuracy-gated acquisition orchestration; physical computation stays unchanged.

Fresh attempts are run namespaces, not engineering copies. Verified passed
prefix files are immutable hard links with an explicit reuse manifest. Older
failed suffixes and their fitted weights remain completely untouched.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ATTEMPT = 'strict8pp_20261020'
HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


def required_correct(depth, prefix):
    assert depth in (4, 6) and 1 <= prefix <= depth + 1
    return math.ceil({4: 346, 6: 354}[depth] - 411 * (1 if prefix == depth + 1 else 8) / 100)


def check_score(report, depth, prefix):
    """Neither 'complete', legacy carry reviews, nor a Boolean can waive count."""
    assert report['state'] == 'complete' and report['frozen_parameters_unchanged']
    result = report['results']['best']
    score = result['TEST_development']
    assert result['strict_CPU'] and score['samples'] == 411
    assert report['selected_correct'] == score['correct']
    assert score['correct'] >= required_correct(depth, prefix), 'Accuracy gate not reached: no next physical layer'
    return result['PT_SHA256']


def link_file(source, target, manifest):
    assert source.is_file() and not target.exists()
    target.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, target)
    manifest.append(dict(source=str(source), target=str(target), bytes=source.stat().st_size))


def prepare(a, m):
    import psutil
    assert not m.LOCK.exists()
    # Do not prepare into a live experiment. Report PIDs only, never credentials.
    for p in psutil.process_iter(['pid', 'name', 'cmdline']):
        if p.pid in (os.getpid(), os.getppid()):
            continue
        cmd = ' '.join(p.info['cmdline'] or [])
        assert not (p.info['name'] and 'python' in p.info['name'].lower()
                    and ('--mode capture' in cmd or '--mode refine' in cmd)
                    and 'caltech20' in cmd), ('Existing capture/refinement', p.pid)
    session = m.session_for(a)
    assert not session.exists()
    old = a.sequence / f'L{a.depth}'
    data, dm = m.data_for(a.sequence)
    hardware = m.session_hardware_digest(a.sequence)
    manifest = []
    session.mkdir()
    if a.depth == 4:
        # Router and E1 passed; their exact measured arrays, not older geometry.
        for stage in ('router', '0'):
            source = old / 'capture' / stage
            assert read(source / 'independent_capture_audit.json')['passed']
            for p in source.rglob('*'):
                if p.is_file():
                    link_file(p, session / 'capture' / stage / p.relative_to(source), manifest)
        parent = '0_head_finish_20261020'
        r = read(old / 'fits' / parent / 'report.json')
        digest = check_score(r, 4, 2)
        assert digest == 'ec2ead5b4c0df87602be4cd5c113b1983cb76ca3436c86c98f118cf302540376'
        _, ids = m.fields_for(old, 2, data, dm, hardware)
        chain = m.validate_fit_ancestry(old, parent, digest, 2,
                                      read(old / 'capture/0/report.json')['PT_SHA256'], ids, dm['sha256'])
        for node in chain:
            source = old / 'fits' / node['fit']
            assert read(source / 'independent_audit.json')['passed']
            for p in source.rglob('*'):
                if p.is_file():
                    link_file(p, session / 'fits' / node['fit'] / p.relative_to(source), manifest)
        _, new_ids = m.fields_for(session, 2, data, dm, hardware)
        assert ids == new_ids
    else:
        parent = 'initial'
    save(session / 'reuse_manifest.json', dict(files=manifest, unchanged_originals=True,
         immutable_links=True, passed_prefix=2 if a.depth == 4 else 0,
         first_retake='1' if a.depth == 4 else 'router', parent=parent,
         data_SHA256=dm['sha256'], hardware_SHA256=hardware, source_git=m.git_head(),
         old_suffix_reused=False, TEST_informed_development=True, TEST_gradients=False, VALID=0))
    # A fresh full native gate binds this published source, not the old source SHA.
    cfg = read(HERE / 'lab_caltech20_dvp_layerwise_profile_20261019.json')
    m.gate(a, data, dm, cfg['models'][str(a.depth)])
    save(session / 'prepared.json', dict(passed=True, source_git=m.git_head(),
         reuse_manifest_SHA256=sha(session / 'reuse_manifest.json'),
         gate_SHA256=sha(session / 'lab_gate.json'), first_retake='1' if a.depth == 4 else 'router'))
    print(json.dumps(dict(prepared=True, depth=a.depth, hardware_used=False)), flush=True)


def acquire_stage(a, m):
    session = m.session_for(a)
    assert read(session / 'prepared.json')['passed'] and not m.LOCK.exists()
    order = ['router'] + [str(i) for i in range(a.depth)]
    prefix = order.index(a.stage)
    assert not (session / 'capture' / a.stage).exists(), 'No automatic reshoot/retry'
    if prefix:
        assert a.parent and a.parent != 'initial'
        fit = session / 'fits' / a.parent
        r, protocol = read(fit / 'report.json'), read(fit / 'protocol.json')
        assert protocol['prefix'] == prefix and read(fit / 'independent_audit.json')['passed']
        digest = check_score(r, a.depth, prefix)
        checkpoint = fit / 'best.pt'
    else:
        assert a.parent == 'initial'
        checkpoint = session / 'initial.pt'
        digest = read(HERE / 'lab_caltech20_dvp_layerwise_profile_20261019.json')['models'][str(a.depth)]['PT_SHA256']
    assert sha(checkpoint) == digest
    out = session / ('stage_' + a.stage)
    assert not out.exists()
    out.mkdir()
    fixed = session / 'shared_fixed' / a.stage
    core = HERE / 'lab_caltech20_dvp_layerwise_20261019.py'
    source_digest = sha(core)
    common = ['--sequence', str(a.sequence), '--depth', str(a.depth), '--attempt', ATTEMPT,
              '--stage', a.stage, '--checkpoint', str(checkpoint), '--digest', digest,
              '--capture-policy', str(HERE / 'lab_caltech20_DVP_geometry20261020_capture_profile.json')]
    commands = [
        [str(core), '--mode', 'fixed-hardware', *common, '--out', str(fixed)],
        [str(core), '--mode', 'capture', *common,
         '--profile', str(HERE / 'lab_caltech20_dvp_layerwise_profile_20261019.json'),
         '--hardware-manifest', str(fixed / 'hardware.json'), '--async-png'],
    ]
    if prefix:
        commands[1] += ['--parent-fit', a.parent]
    save(out / 'protocol.json', dict(source_git=m.git_head(), source_SHA256=source_digest,
         commands=commands, PT_SHA256=digest, prerequisite_target=required_correct(a.depth, prefix) if prefix else None,
         mandatory_next_target=required_correct(a.depth, prefix + 1), automatic_next_layer=False,
         missed_target_action='STOP_AND_DIAGNOSE', old_CCD_suffix_reused=False,
         TRAIN=1600, TEST=411, VALID=0))
    for label, command in zip(('fixed_hardware', 'capture'), commands):
        assert sha(core) == source_digest and not m.LOCK.exists()
        save(out / 'state.json', dict(state=label, started_ns=time.time_ns(), automatic_retry=False))
        with (out / (label + '.log')).open('x', encoding='utf-8') as log:
            result = subprocess.run([sys.executable, *command], cwd=m.ROOT,
                                    stdout=log, stderr=subprocess.STDOUT,
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        assert result.returncode == 0, (label, result.returncode)
    report = read(session / 'capture' / a.stage / 'report.json')
    assert report['state'] == 'complete' and report['sdk_released']
    assert report['TRAIN'] == 1600 and report['TEST'] == 411 and not m.LOCK.exists()
    save(out / 'state.json', dict(state='captured_pending_suffix_fit', report_SHA256=sha(session / 'capture' / a.stage / 'report.json'),
         SDK_released=True, automatic_next_layer=False, accuracy_pass_not_claimed=True))
    print(json.dumps(dict(captured=True, depth=a.depth, stage=a.stage)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('prepare', 'capture'), required=True)
    p.add_argument('--sequence', type=Path, required=True)
    p.add_argument('--depth', type=int, choices=(4, 6), required=True)
    p.add_argument('--stage')
    p.add_argument('--parent', default='initial')
    a = p.parse_args()
    a.attempt = ATTEMPT
    import lab_caltech20_dvp_layerwise_20261019 as m
    m.torch.set_num_threads(2)
    if a.mode == 'prepare':
        prepare(a, m)
    else:
        acquire_stage(a, m)
