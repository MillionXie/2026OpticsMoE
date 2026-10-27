"""CPU replay of the frozen Qwen merger on verified terminal vision activations.

No image encoder rerun, no additional image branch, no target image input.
Raw deployment must capture the same merger during the ordinary Qwen forward.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

import torch

from .qwen_vl_hybrid_run import checked_features, load_data
from .qwen_vl_turbo_cache import save_with_digest
from .qwen_vl_turbo_data import sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('qwen', 'data', 'features', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    a = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or a.output.exists():
        raise ValueError('CPU-only immutable feature replay required')
    a.output.mkdir(parents=True)
    write_json(a.output / 'execution.json', {'argv': sys.argv, 'pid': os.getpid(),
        'source_commit': os.environ.get('T12_SOURCE_COMMIT'), 'source_sha256': sha256(Path(__file__)),
        'started_unix': time.time(), 'device': 'cpu', 'targets_are_not_inputs': True})
    torch.set_num_threads(4)
    payload = load_data(a.data)
    records = checked_features(a.features, payload)
    from transformers import Qwen3VLForConditionalGeneration
    full = Qwen3VLForConditionalGeneration.from_pretrained(a.qwen, torch_dtype=torch.bfloat16,
        attn_implementation='sdpa', local_files_only=True)
    merger = full.model.visual.merger.eval().requires_grad_(False)
    del full
    gc.collect()
    features, shared = [], {}
    with torch.inference_mode():
        for i, record in enumerate(records):
            source = record['vision_23'][0]
            # Cached equal images may share tensors; use bytes to verify rather
            # than trusting a product ID or a scene name.
            import hashlib
            digest = hashlib.sha256(source.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            if digest not in shared:
                value = merger(source).detach().bfloat16().cpu()
                if value.shape != (64, 2048) or not torch.isfinite(value).all():
                    raise ValueError('Native merger output shape/values differ')
                shared[digest] = value
            features.append(shared[digest])
            if (i + 1) % 96 == 0:
                write_json(a.output / 'progress.json', {'done': i + 1, 'total': len(records), 'unique_images': len(shared)})
    source_sha = json.loads((a.data / 'cache_complete.json').read_text())['sha256']
    saved = {'source_cache_sha256': source_sha, 'sample_ids': [r['sample_id'] for r in payload['rows']],
        'vision': features, 'grid': [8, 8], 'source': 'Frozen original Qwen merger on recorded terminal vision23',
        'input_sha256': [r['input_sha256'] for r in records]}
    digest = save_with_digest(saved, a.output / 'native_vision.pt')
    write_json(a.output / 'report.json', {'complete': True, 'samples': len(features), 'sha256': digest,
        'file': 'native_vision.pt', 'source_cache_sha256': source_sha,
        'terminal_feature_sha256': sha256(a.features / 'teacher_features.pt'),
        'unique_terminal_images': len(shared), 'raw_GPU_identity_pending': True,
        'note': 'Training cache only. Paired labels were not read by the merger. Validate same-raw-Qwen output before a quality claim.'})


if __name__ == '__main__':
    main()
