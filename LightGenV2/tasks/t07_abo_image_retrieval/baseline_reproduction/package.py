"""Build a small baseline-only handoff; no source copying from other projects."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    source = Path(__file__).resolve().parent
    repo = Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],cwd=source,text=True).strip())
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    files = {}
    for name in ['baseline.py', 'test_baseline.py', 'requirements.txt', 'README.md']:
        rel = (source/name).relative_to(repo).as_posix()
        files[name] = subprocess.check_output(['git','show',f'{commit}:{rel}'],cwd=repo)
    paths = {
        'evidence/abo_similarity10_manifest.csv': args.data/'data/abo_similarity10_manifest.csv',
        'evidence/enrolled_protocol.json': args.runs/'abo200_enrolled_protocol_20260913/protocol.json',
        'evidence/legacy/features.pt': args.runs/'frozen_qwen_20260912/features.pt',
        'evidence/legacy/baseline_report.json': args.runs/'frozen_qwen_20260912/baseline_report.json',
        'evidence/legacy/run_manifest.json': args.runs/'frozen_qwen_20260912/run_manifest.json',
        'evidence/enrolled/features.pt': args.runs/'abo200_enrolled_qwen64_20260913/normal_features.pt',
        'evidence/enrolled/final_report.json': args.runs/'abo200_enrolled_qwen64_20260913/final_report.json',
    }
    # Require the actual full-image runs, not just a successful cache audit.
    for protocol in ['legacy_category', 'enrolled_sku']:
        run = args.runs/f'baseline_reproduction_20260914_{protocol}'
        report = json.loads((run/'report.json').read_text())
        expected = .94375 if protocol == 'legacy_category' else .85125
        if report['status'] != 'complete' or report['mode'] != 'infer' or report['metrics_by_dimension']['64']['hit_at_1'] != expected:
            raise ValueError('Independent reproduction incomplete/different: '+protocol)
        for name in ['report.json','predictions_64d.csv','predictions_2048d.csv']:
            paths[f'evidence/independent/{protocol}/{name}'] = run/name
    for name,path in paths.items(): files[name] = path.read_bytes()
    pinned = {
        'evidence/abo_similarity10_manifest.csv': '2949a4035150a9f8718f2a6cace164c17394613d24fb9d0234c553bee8d77c97',
        'evidence/enrolled_protocol.json': 'f1749d5fc22d2dfee6a1333ce2b35e9fa600a070f949eba8420b4def41906dde',
        'evidence/legacy/features.pt': '38f77637e48cf28fb7b1e077119bcf4c1ea37dd3480cbd1b6ce5707f05b38324',
        'evidence/enrolled/features.pt': 'c6eb631c268d2446a2f783854c86d8493cdbcaa04c0669148b16d9785016d8d7',
    }
    for name,expected in pinned.items():
        if hashlib.sha256(files[name]).hexdigest() != expected: raise ValueError('Source artifact SHA mismatch: '+name)
    manifest = dict(source_commit=commit, scope='Frozen baseline reproduction only; NOT a hardware package',
        exclusions=['ABO images','Qwen model weights','optical model','hardware credentials'],
        files={name:dict(bytes=len(value),sha256=hashlib.sha256(value).hexdigest()) for name,value in files.items()})
    files['PACKAGE_MANIFEST.json'] = json.dumps(manifest,indent=2).encode()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,value in files.items(): z.writestr(name,value)
    print(json.dumps(dict(zip=str(args.output.resolve()),bytes=args.output.stat().st_size,
        sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),source_commit=commit)))


if __name__ == '__main__': main()
