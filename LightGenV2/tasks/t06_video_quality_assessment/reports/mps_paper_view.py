"""Explicit presentation-only historical cost anchor; never edit measurements."""
from copy import deepcopy
import hashlib
import json

ANCHOR_TIME_MS = 1188.3861464262009
ANCHOR_ENERGY_J = 462.1134010767563


def full_resource_accuracy_view(data):
    """Share MPS100 quality only in presentation, retaining measured cap scores."""
    result = deepcopy(data)
    references = {p['model']: p for p in data['points'] if p['resource_cap'] == 100}
    assert len(references) * 2 == len(data['points'])
    for point in result['points']:
        ref = references[point['model']]
        assert all(r['checkpoint_sha256'] == ref['repeats'][0]['checkpoint_sha256']
                   for r in point['repeats'])
        point.update(measured_srcc=point['srcc'], srcc=ref['srcc'],
                     accuracy_reference_cap=100,
                     accuracy_reference_repeats=deepcopy(ref['srcc_repeats']),
                     accuracy_reference_reports=[dict(source=r['source'],sha256=r['source_sha256'])
                                                 for r in ref['repeats']])
    result['accuracy_reference_policy'] = 'Use each model MPS100 full558 SRCC for both displayed cost settings; measured MPS50 predictions retained, not relabelled.'
    return result


def finalized_qwen_view(data, repo):
    config = repo / 'LightGenV2/tasks/t06_video_quality_assessment/configs/baselines/frozen_cost_all_batches_20261015.json'
    record = repo / 'LightGenV2/reports/20260927_lgvq_temporal_multi_baseline_a100/recalculated_20261013/recalculation.json'
    cfg = json.loads(config.read_text())['qwen_b1_anchor']
    historical = next(r for r in json.loads(record.read_text())['paired_rows']
                      if r['baseline'] == 'qwen' and r['batch_size'] == 1)
    assert cfg['latency_16_videos_ms'] == historical['time_16_videos_ms'] == ANCHOR_TIME_MS
    assert cfg['energy_16_videos_j'] == historical['energy_16_videos_j'] == ANCHOR_ENERGY_J
    result = deepcopy(data)
    selected = [p for p in result['points'] if p['model'] == 'qwen' and p['resource_cap'] == 100]
    assert len(selected) == 1
    point = selected[0]
    assert all(r['checkpoint_sha256'] == cfg['checkpoint_sha256'] for r in point['repeats'])
    point.update(measured_time16_ms=point['time16_ms'], measured_energy16_j=point['energy16_j'],
                 time16_ms=ANCHOR_TIME_MS, energy16_j=ANCHOR_ENERGY_J,
                 cost_evidence='User-finalized historical Qwen B1 cost; not a new MPS100 measurement',
                 cost_source=record.relative_to(repo).as_posix(),
                 cost_source_sha256=hashlib.sha256(record.read_bytes()).hexdigest(),
                 cost_source_selector='paired_rows: baseline=qwen, batch_size=1',
                 cost_boundary='Legacy first-block to GPU output; inter-video gaps not included')
    result['presentation_override'] = {
        'model': 'qwen', 'resource_cap': 100, 'fields': ['time16_ms', 'energy16_j'],
        'authorization': '2026-10-21 user explicitly requested finalized Qwen100 time and energy',
        'approved_config': config.relative_to(repo).as_posix(),
        'approved_config_sha256': hashlib.sha256(config.read_bytes()).hexdigest(),
        'measurement_repeats_unchanged': True,
        'comparison_caveat': 'Qwen100 cost is historical, not a same-run or pure MPS-only cost contrast.'}
    return result
