"""Prepare additional source-disjoint TRAIN1000, without hardware or model selection."""
import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    args = parser.parse_args()
    data = args.project.parent / 'OpenMoji_Lab_SHS_8um/data'
    old = args.project / 'data_train_adapt1000'
    output = args.project / 'data_train_extra1000'
    assert not output.exists(), 'Do not overwrite prepared data'
    train = [json.loads(row) for row in (data / 'train.jsonl').read_text().splitlines()]
    test = [json.loads(row) for row in (data / 'test.jsonl').read_text().splitlines()]
    previous = [json.loads(row) for row in (old / 'capture_train.jsonl').read_text().splitlines()]
    assert len(previous) == len(test) == 1000
    previous_ids = {r['sample_id'] for r in previous}
    test_ids = {r['sample_id'] for r in test}
    excluded_hashes = {sha(data / r['relative_dir'] / r['files']['source']) for r in test + previous}
    groups = {}
    for row in train:
        if row['sample_id'] in previous_ids | test_ids:
            continue
        digest = sha(data / row['relative_dir'] / row['files']['source'])
        if digest not in excluded_hashes:
            groups.setdefault(row['task'], []).append((row, digest))
    rng = random.Random(1002)
    selected, used = [], set(excluded_hashes)
    assert set(groups) == {'add', 'replace', 'move', 'remove'}
    for task, candidates in sorted(groups.items()):
        rng.shuffle(candidates)
        chosen = []
        for row, digest in candidates:
            if digest not in used:
                chosen.append(row)
                used.add(digest)
            if len(chosen) == 250:
                break
        assert len(chosen) == 250, (task, len(chosen))
        selected.extend(chosen)
    ids = {r['sample_id'] for r in selected}
    assert len(ids) == 1000 and not ids & (previous_ids | test_ids)
    output.mkdir()
    for row in selected:
        target = output / row['relative_dir']
        target.mkdir(parents=True, exist_ok=True)
        for source in (data / row['relative_dir']).iterdir():
            if source.is_file():
                shutil.copy2(source, target / source.name)
    shutil.copy2(data / 'token_embeddings_v1.pt', output / 'token_embeddings_v1.pt')
    (output / 'capture_train.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in selected), encoding='utf-8')
    audit = {'status': 'prepared_only', 'seed': 1002, 'capture_count': 1000,
        'fit_ids': sorted(ids), 'test_ids_overlap': 0, 'test_source_overlap': 0,
        'prior_train_ids_overlap': 0, 'prior_train_source_overlap': 0,
        'internal_duplicate_sources': 0, 'tasks': {task: 250 for task in groups},
        'prior_manifest_sha256': sha(old / 'capture_train.jsonl'),
        'train_manifest_sha256': sha(data / 'train.jsonl'), 'test_manifest_sha256': sha(data / 'test.jsonl'),
        'capture_manifest_sha256': sha(output / 'capture_train.jsonl'),
        'hardware_started': False, 'selection': 'TRAIN sources only; no TEST performance selection'}
    (output / 'split_audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
