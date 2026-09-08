"""Reproducible deduplicated scenes and raw frozen word-table lookup (no TF)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import torch
from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.scenes import (
    TASKS, generate_example, _save_sample, prompt_key, validate_assets, load_icons,
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def identity(record):
    grid = record['source_grid']
    return json.dumps([grid.tolist() if hasattr(grid, 'tolist') else grid, record['instruction']], sort_keys=True)


def prepare_embedding_data(settings):
    root = settings.data_dir
    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / 'dataset_summary.json'
    kind = 'openmoji_deduplicated_grid_instruction_v2'
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        if summary.get('type') != kind or summary.get('seed') != settings.seed:
            raise ValueError('Incompatible dataset; choose a separate data directory')
    else:
        if any(root.iterdir()):
            raise RuntimeError('Incomplete dataset directory; inspect it before retrying in a new empty directory')
        assets, icons = validate_assets(settings), load_icons(settings)
        seen = set()
        summary = {'type': kind, 'seed': settings.seed, 'assets': assets, 'validation_split': False}
        for split, count, offset in [('train', settings.train_samples, 0), ('test', settings.test_samples, 20_000_000)]:
            records, rejected = [], 0
            counts = {task: 0 for task in TASKS}
            for index in range(count):
                task = TASKS[index % len(TASKS)]
                for attempt in range(10000):
                    example = generate_example(task, settings.seed * 1_000_003 + offset + index + attempt * 100_000_000, settings)
                    key = identity(example)
                    if key not in seen:
                        seen.add(key)
                        break
                    rejected += 1
                else:
                    raise RuntimeError('Could not generate unique scene/instruction')
                sample_id = f'{split}_{index:06d}'
                relative = Path(split) / sample_id
                metadata = _save_sample(example, root / relative, settings, icons)
                records.append({'sample_id': sample_id, 'split': split, 'relative_dir': relative.as_posix(), **metadata})
                counts[task] += 1
            manifest = root / f'{split}.jsonl'
            manifest.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), encoding='utf-8')
            summary[split] = {'samples': count, 'task_counts': counts, 'duplicates_resampled': rejected, 'sha256': sha256(manifest)}
        summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    records = []
    for split in ('train', 'test'):
        path = root / f'{split}.jsonl'
        if sha256(path) != summary[split]['sha256']:
            raise ValueError('Dataset manifest changed after preparation')
        records.extend(json.loads(line) for line in path.read_text(encoding='utf-8').splitlines())
    if len({identity(r) for r in records}) != len(records):
        raise ValueError('Duplicate source-grid/instruction across or within splits')
    build_token_cache(settings, records, summary)
    return summary


def build_token_cache(settings, records, summary):
    from transformers import AutoTokenizer
    from safetensors import safe_open

    contract = {'type': 'qwen_token_lookup_only_v1', 'native_transformer_blocks': 0,
                'checkpoint': str(settings.qwen_checkpoint), 'max_tokens': settings.max_language_tokens,
                'train_sha256': summary['train']['sha256'], 'test_sha256': summary['test']['sha256']}
    if settings.prompt_cache_path.exists():
        cache = torch.load(settings.prompt_cache_path, map_location='cpu', weights_only=False)
        if any(cache['meta'].get(k) != v for k, v in contract.items()):
            raise ValueError('Wrong prompt cache: raw token lookup is mandatory, contextual TF cache forbidden')
        return cache['meta']
    tokenizer = AutoTokenizer.from_pretrained(settings.qwen_checkpoint, local_files_only=True)
    instructions = sorted({r['instruction'] for r in records})
    ids = {text: tokenizer.encode(text, add_special_tokens=False) for text in instructions}
    if any(not tokens or len(tokens) > settings.max_language_tokens for tokens in ids.values()):
        raise ValueError('Instruction token length outside contract; no silent truncation')
    checkpoint = settings.qwen_checkpoint
    index_path = checkpoint / 'model.safetensors.index.json'
    candidates = []
    if index_path.exists():
        mapping = json.loads(index_path.read_text(encoding='utf-8'))['weight_map']
        candidates = [(checkpoint / file, key) for key, file in mapping.items() if key.endswith('embed_tokens.weight')]
    else:
        for file in checkpoint.glob('*.safetensors'):
            with safe_open(file, framework='pt', device='cpu') as handle:
                candidates.extend((file, key) for key in handle.keys() if key.endswith('embed_tokens.weight'))
    if len(candidates) != 1:
        raise ValueError(f'Expected one Qwen word embedding table, found {len(candidates)}')
    weight_file, key = candidates[0]
    with safe_open(weight_file, framework='pt', device='cpu') as handle:
        table = handle.get_slice(key)
        if table.get_shape()[1] != 2048:
            raise ValueError('Expected Qwen 2B embedding width 2048')
        rows = {i: table[i:i+1].squeeze(0).to(torch.bfloat16).clone() for i in sorted({i for seq in ids.values() for i in seq})}
    prompts = {prompt_key(text): torch.stack([rows[i] for i in tokens]) for text, tokens in ids.items()}
    contract.update(weight_key=key, weight_file_sha256=sha256(weight_file), unique_prompts=len(prompts),
                    instruction_format='plain text, no chat wrapper', frozen=True)
    torch.save({'meta': contract, 'prompts': prompts}, settings.prompt_cache_path)
    settings.prompt_cache_path.with_suffix('.json').write_text(json.dumps(contract, indent=2), encoding='utf-8')
    return contract
