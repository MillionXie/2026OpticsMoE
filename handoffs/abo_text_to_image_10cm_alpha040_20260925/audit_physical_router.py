"""Record measured optical-router Top-2 decisions without quality gating."""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from export_full_next_stage import ccd, router_scores
from LightGenV2.tasks.t01_object_retrieval.settings import load_settings


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--physical-root', type=Path, required=True)
    p.add_argument('--stage', choices=('vision_router', 'language_router'), required=True)
    a = p.parse_args()
    settings = load_settings(a.config)
    keys = [f'image_{i:04d}' for i in range(2400)]
    if a.stage == 'language_router':
        keys += [f'title_{i:03d}' for i in range(100)]
    output = a.physical_root / (a.stage + '_routing.jsonl')
    usage = Counter()
    with output.open('w', encoding='utf-8') as stream:
        for key in keys:
            image = ccd(a.physical_root, a.stage, key)
            route = router_scores(image[None], settings)
            selected = route['selected_indices'][0].tolist()
            usage.update(selected)
            row = {'key': key, 'selected_experts': selected,
                   'probabilities': route['probabilities'][0].tolist(),
                   'weights': route['weights'][0].tolist(),
                   'detector_energy': route['detector_energy'][0].tolist(),
                   'raw_capture_fraction': float(route['raw_capture_fraction'][0]),
                   'p99_uint8': float(np.percentile(image, 99)),
                   'saturation_fraction': float(np.mean(image == 255))}
            stream.write(json.dumps(row) + '\n')
    summary = {'stage': a.stage, 'samples': len(keys), 'top_k': settings.top_k,
               'expert_selection_counts': {str(k): usage[k] for k in range(4)},
               'no_pcc_or_brightness_rejection': True,
               'routing_manifest': str(output)}
    (a.physical_root / (a.stage + '_routing_summary.json')).write_text(
        json.dumps(summary, indent=2), encoding='utf-8'
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
