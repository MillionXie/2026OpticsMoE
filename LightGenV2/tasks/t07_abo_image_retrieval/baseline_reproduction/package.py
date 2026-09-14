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
    # These optional independently rerun reports are evidence, not alternative cached scores.
    for protocol in ['legacy_category', 'enrolled_sku']:
        run = args.runs/f'baseline_reproduction_20260914_{protocol}'
        for name in ['report.json','predictions_64d.csv','predictions_2048d.csv']:
            if (run/name).is_file(): paths[f'evidence/independent/{protocol}/{name}'] = run/name
    for name,path in paths.items(): files[name] = path.read_bytes()
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
