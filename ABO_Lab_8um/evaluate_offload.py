"""Replay all six real CCD stages on a server; no hardware or simulation fallback."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import zipfile

import numpy as np
from PIL import Image
from common import ROOT, CHECKPOINT_SHA, config, hardware_identity, read, sha, write, session_path
from offload import normalized, file_hash, required


def verify_files(root, files):
    for name, expected in files.items():
        if sha(Path(root)/normalized(name)) != expected:
            raise ValueError('Changed file: '+name)


def export_delta(session, base_report, target):
    c, _ = config()
    root = session_path(session)
    state = read(root/'session.json')
    report = read(base_report)
    base = report['manifest']
    if base['session'] != session or base['checkpoint_sha256'] != CHECKPOINT_SHA:
        raise ValueError('Base session/checkpoint mismatch')
    if sha(root/'session.json') != base['files']['session.json']:
        raise ValueError('Session changed since base snapshot')
    if hardware_identity(c) != state['hardware_identity']:
        raise ValueError('Hardware configuration changed')
    for i, (name, expected) in enumerate(base['source_records'].items(), 1):
        if sha(root/normalized(name)) != expected:
            raise ValueError('Upstream capture record changed: '+name)
        if i % 2000 == 0: print('Checked upstream records', i, flush=True)
    for s in state['samples']:
        if not (root/'ccd'/normalized(s['id'])/'language_global.record.json').is_file():
            raise ValueError('Missing last-stage capture: '+s['id'])
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    files = {}
    with zipfile.ZipFile(target, 'x', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for i, s in enumerate(state['samples'], 1):
            sid = normalized(s['id'])
            d = root/'ccd'/sid
            rp = d/'language_global.record.json'
            record = read(rp)
            if (record['stage'] != 'language_global' or record['sample'] != sid
                    or record.get('capture_mode') != 'real'
                    or record['hardware_identity'] != state['hardware_identity']
                    or record['phase_sha256'] != base['phase_sha256']):
                raise ValueError('Wrong last-stage capture identity: '+sid)
            png = d/'language_global.png'
            data = png.read_bytes()
            if file_hash(data) != record['files'][png.name]:
                raise ValueError('Last-stage PNG changed: '+sid)
            for path, content in [(png, data), (rp, rp.read_bytes())]:
                name = 'ccd/'+sid+'/'+path.name
                files[name] = file_hash(content)
                z.writestr(name, content)
            if i % 250 == 0: print('Packed final stage', i, '/', len(state['samples']), flush=True)
        manifest = dict(schema=1, session=session, samples=len(state['samples']),
            base_archive_sha256=report['sha256'], session_sha256=sha(root/'session.json'),
            hardware_identity=state['hardware_identity'], checkpoint_sha256=CHECKPOINT_SHA,
            files=files, integrity_scope='Earlier immutable snapshot is reused after checking current capture record identities. Final-stage PNG hashes checked against real capture records. Raw TIFFs are not consumed, transferred, deleted or rehashed.')
        z.writestr('DELTA.json', json.dumps(manifest, indent=2))
    result = dict(zip=str(target), bytes=target.stat().st_size, sha256=sha(target), manifest=manifest)
    write(target.with_suffix('.report.json'), result)
    print('DELTA_READY', target, result['bytes'], result['sha256'], flush=True)


def evaluate(base_archive, base_dir, archive, out, device):
    start = time.monotonic()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    base_dir = Path(base_dir)
    with zipfile.ZipFile(archive) as z:
        delta = json.loads(z.read('DELTA.json'))
        if sha(base_archive) != delta['base_archive_sha256']:
            raise ValueError('Wrong base archive')
        for name, expected in delta['files'].items():
            name = normalized(name)
            data = z.read(name)
            if file_hash(data) != expected: raise ValueError('Corrupt delta: '+name)
            p = out/'final_stage'/name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
    with zipfile.ZipFile(base_archive) as z:
        base = json.loads(z.read('SNAPSHOT.json'))
    if base['session'] != delta['session'] or base['files']['session.json'] != delta['session_sha256']:
        raise ValueError('Base and delta session mismatch')
    verify_files(base_dir, base['files'])
    verify_files(ROOT, base['runtime_fingerprints'])
    state = read(base_dir/'session.json')
    if (len(state['samples']) != delta['samples'] or state['checkpoint_sha256'] != CHECKPOINT_SHA
            or state['hardware_identity'] != delta['hardware_identity']):
        raise ValueError('Session identity mismatch')
    print('All transferred inputs verified; loading exact fixed model', flush=True)
    import torch
    from backend import create, forward, optical_contract
    from memory import memory_report
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    b = create(device, force_fp32=True)
    b.guard_optics()
    if optical_contract(b) != read(base_dir/'optical_contract.json'):
        raise ValueError('Optical contract changed')
    source_images = {p.stem: p for p in (base_dir/'images').iterdir()}
    images, titles, labels, ids, title_ids = [], [], [], [], []
    for i, sample in enumerate(state['samples'], 1):
        s = dict(sample)
        if s['kind'] == 'image': s['path'] = str(source_images[s['id']])
        measured = {}
        for stage in (*required(s), 'language_global'):
            folder = out/'final_stage' if stage == 'language_global' else base_dir
            d = folder/'ccd'/normalized(s['id'])
            if stage.endswith('router'):
                measured[stage] = read(d/(stage+'.route.json'))
            else:
                with Image.open(d/(stage+'.png')) as im: measured[stage] = np.array(im)
        v = forward(b, s, measured, release=True)
        if s['kind'] == 'title': titles.append(v); title_ids.append(s['id'])
        else: images.append(v); labels.append(s['label']); ids.append(s['id'])
        if i % 100 == 0: print('Evaluated', i, '/', len(state['samples']), 'seconds', round(time.monotonic()-start, 1), flush=True)
    metrics, rows = b.m._metrics(torch.tensor(np.stack(images)), torch.tensor(np.stack(titles)), labels)
    write(out/'metrics.json', dict(mode='real_six_stage_server_fp32', metrics=metrics,
        n_queries=len(images), n_candidates=len(titles), checkpoint_sha256=CHECKPOINT_SHA,
        predictions=[dict(sample_id=s, **r) for s, r in zip(ids, rows)]))
    np.savez(out/'embeddings.npz', queries=np.stack(images), titles=np.stack(titles),
        labels=labels, query_ids=ids, title_ids=title_ids)
    report = dict(session=delta['session'], hardware_identity=state['hardware_identity'],
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        base_archive_sha256=delta['base_archive_sha256'], delta_archive_sha256=sha(archive),
        elapsed_seconds=time.monotonic()-start, torch_version=torch.__version__,
        numpy_version=np.__version__, memory=memory_report(b), integrity_scope=delta['integrity_scope'])
    write(out/'evaluation_report.json', report)
    (out/'README.txt').write_text('Six-stage real CCD replay, exact fixed checkpoint, FP32, TF32 disabled.\nNo training, no simulated CCD fallback. Original TIFFs remain on the lab PC.\nEarlier stages use the verified immutable snapshot; final stage uses the verified delta.\n', encoding='utf-8')
    result = out/'evaluation_results.zip'
    names = ['metrics.json', 'embeddings.npz', 'evaluation_report.json', 'README.txt']
    with zipfile.ZipFile(result, 'x', zipfile.ZIP_DEFLATED) as z:
        for name in names: z.write(out/name, name)
        z.writestr('FILES.json', json.dumps({n: sha(out/n) for n in names}, indent=2))
    print('RESULT', json.dumps(metrics), flush=True)
    print('RESULT_ZIP', result, result.stat().st_size, sha(result), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    e = sub.add_parser('export')
    e.add_argument('--session', required=True)
    e.add_argument('--base-report', type=Path, required=True)
    e.add_argument('--archive', type=Path, required=True)
    r = sub.add_parser('evaluate')
    for arg in ['base-archive', 'base-dir', 'archive', 'out']: r.add_argument('--'+arg, type=Path, required=True)
    r.add_argument('--device', default='cuda')
    a = p.parse_args()
    if a.action == 'export': export_delta(a.session, a.base_report, a.archive)
    else: evaluate(a.base_archive, a.base_dir, a.archive, a.out, a.device)


if __name__ == '__main__': main()
