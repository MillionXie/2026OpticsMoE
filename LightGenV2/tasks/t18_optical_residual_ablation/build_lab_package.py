"""Build the audited L2 residual simulation reference; no device/network access."""
import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

TASK = Path(__file__).resolve().parent
ROOT = TASK.parents[2]
REL = TASK.relative_to(ROOT).as_posix()
PT_SHA = 'b08923b3b43731e2832fc0251a54af9c52535226176070b92b4954f636614c69'
TS_SHA = '590c3b5cce474faacdc58174315590fb6c185d5c425b80d6220db6875a9c1425'
SOURCES = ['lab_caltech20_rich_model_20261018.py', 'caltech20_alt246_biasfree_model.py',
           'caltech_repair234_model.py', 'caltech_diagnose_model.py', 'caltech_deploy234_model.py']


def digest(data):
    return hashlib.sha256(data).hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def build(output, validation=None):
    pin = git('rev-parse', 'HEAD').decode().strip()
    members, provenance = {}, {}
    paths = [REL + '/' + name for name in SOURCES] + [
        'LightGenV2/tasks/t07_abo_image_retrieval/standalone/optics.py']
    for name in paths:
        data = git('show', pin + ':' + name)
        # Do not silently export an older Git blob over a user's local source edit.
        assert ROOT.joinpath(name).read_bytes().replace(b'\r\n', b'\n') == data.replace(b'\r\n', b'\n'), name
        key = 'src/' + name
        members[key] = data
        provenance[key] = dict(git=pin, repository_path=name)
    for name in ['LightGenV2', 'LightGenV2/tasks', 'LightGenV2/tasks/t07_abo_image_retrieval',
                 'LightGenV2/tasks/t07_abo_image_retrieval/standalone']:
        members['src/' + name + '/__init__.py'] = b''
    example = TASK / 'lab_examples/residual_l2'
    for name in ['example.py', 'README_CN.md']:
        data = (example / name).read_bytes()
        source = (example / name).relative_to(ROOT).as_posix()
        committed = git('show', pin + ':' + source)
        assert data.replace(b'\r\n', b'\n') == committed.replace(b'\r\n', b'\n'), source
        members[name] = committed
        provenance[name] = dict(git=pin, repository_path=source)
    case = TASK / 'runs/hardware/caltech20_residual_sequence_426_20261018/L2'
    checkpoint = (TASK / 'runs/simulation/caltech20_rich20_L2_first8151_head_target818_TRAIN_20261018/case01/best.pt').read_bytes()
    assert digest(checkpoint) == PT_SHA
    members['weights/simulation_L2_rho03.pt'] = checkpoint
    trace = (case / 'source_reference.ts').read_bytes()
    assert digest(trace) == TS_SHA
    members['reference/source_reference.ts'] = trace
    gate = json.loads((case / 'server_gate.json').read_text())
    assert gate['passed'] and gate['PT_SHA256'] == PT_SHA and gate['checks']['test']['correct'] == 337
    members['reference/server_gate.json'] = json.dumps(gate, indent=2).encode()
    if validation:
        proof = json.loads(validation.read_text(encoding='utf-8'))
        assert proof['passed'] and proof['checkpoint_sha256'] == PT_SHA
        for name in paths:
            assert proof['source_sha256'][name] == digest(members['src/' + name]), name
        members['validation.json'] = json.dumps(proof, indent=2).encode()
    members['requirements.txt'] = b'torch>=2.1,<3\nnumpy>=1.24,<3\nPillow>=9\n'
    manifest = dict(format='t18_residual_l2_reference_v1', source_git=pin,
                    checkpoint_sha256=PT_SHA, simulation_correct=337, simulation_samples=411,
                    includes_CCD=False, includes_experimental_adapted_weights=False,
                    provenance=provenance,
                    files={k: dict(bytes=len(v), sha256=digest(v)) for k, v in sorted(members.items())})
    members['manifest.json'] = json.dumps(manifest, ensure_ascii=False, indent=2).encode()
    assert not output.exists(), 'Use a new explicit release filename; do not overwrite'
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(members.items()):
            assert not name.startswith('/') and '..' not in Path(name).parts
            archive.writestr('T18_L2_Residual_Simulation/' + name, data)
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        for name, item in manifest['files'].items():
            assert digest(archive.read('T18_L2_Residual_Simulation/' + name)) == item['sha256']
    checksum = digest(output.read_bytes())
    output.with_suffix(output.suffix + '.sha256').write_text(checksum + '  ' + output.name + '\n')
    print(json.dumps(dict(zip=str(output), bytes=output.stat().st_size, sha256=checksum,
                          members=len(members), source_git=pin), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--validation', type=Path)
    args = parser.parse_args()
    build(args.output.resolve(), args.validation)
