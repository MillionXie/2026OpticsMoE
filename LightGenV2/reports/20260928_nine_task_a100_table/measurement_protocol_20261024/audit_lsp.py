"""Independent CPU replay of original LSP labels, neural spans and energy.

Never loads a model or changes raw measurements. Full TEST references and the
new first-200 subset are separate. The original person-box task is retained.
"""
from __future__ import annotations
import argparse
import ast
from bisect import bisect_right
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image
from scipy.io import loadmat
from audit_t07 import REPO, sha, read, csv_rows, close

GPU = 'GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55'
PINS = {
    'ours': '495b9c2c4e3df15d3715f1ce8f2faea7cb9156275b31103ec684f4e96a328518',
    'qwen': '0a4569f288f6de424b1b412452fa804a96e58d8f7e8655efaa23f461ab8cc735',
}
ANNOTATION_SHA = 'ddcfcbe904106bfbf26561c8d11720ca62a11c7a861548064b3bae84cb396b2a'


def target_from_original(joints, margin=1.25, size=224):
    """One-based official coordinates -> original integer crop -> float32 XY."""
    points = np.asarray(joints[:, :2], dtype=np.float32).copy() - np.float32(1.)
    valid = np.isfinite(points).all(1)
    xy = points[valid]
    if not len(xy):
        raise ValueError('No finite original joints')
    center = (xy.min(0) + xy.max(0)) * .5
    side = max(float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1])), 32.) * margin
    left = float(center[0] - side * .5)
    top = float(center[1] - side * .5)
    right, bottom = left + side, top + side
    points[:, 0] = (points[:, 0] - math.floor(left)) * (size / (math.ceil(right)-math.floor(left)))
    points[:, 1] = (points[:, 1] - math.floor(top)) * (size / (math.ceil(bottom)-math.floor(top)))
    valid &= (points >= 0).all(1) & (points < size).all(1)
    points[~valid] = np.nan
    return points, valid, [left, top, right, bottom]


def pose_metric(predicted, targets, visible):
    """Original float32 torso scale and PCK@.2, independently implemented."""
    values = [float(np.linalg.norm(targets[a]-targets[b])) for a, b in ((8, 3), (9, 2))
              if visible[a] and visible[b]]
    values = [v for v in values if math.isfinite(v) and v > 1e-6]
    torso = float(np.mean(values)) if values else float('nan')
    head = float(np.linalg.norm(targets[12]-targets[13])) if visible[12] and visible[13] else float('nan')
    head = 2*head if math.isfinite(head) and head > 1e-6 else float('nan')
    distances = np.linalg.norm(predicted-targets, axis=-1)
    mask = visible & np.isfinite(distances)
    if not math.isfinite(torso) or torso <= 0:
        mask[:] = False
    return int(((distances <= .2*torso) & mask).sum()), int(mask.sum()), torso, head


def board_energy(power, start, end):
    ts = [float(r['host_monotonic_s']) for r in power]
    if not ts or ts[0] > start or ts[-1] < end or any(b <= a for a, b in zip(ts, ts[1:])):
        raise ValueError('Missing full-window power support/order')
    tick = start
    index = bisect_right(ts, start)-1
    energy = max_age = 0.
    while tick < end:
        boundary = min(end, ts[index+1] if index+1 < len(ts) else end)
        age = boundary-ts[index]
        watts = float(power[index]['watts'])
        if age > .25 or not math.isfinite(watts) or not 0 < watts <= 250.1:
            raise ValueError('Invalid board power or support older than250ms')
        energy += watts*(boundary-tick)
        max_age = max(max_age, age)
        tick = boundary
        index += 1
    return energy, max_age


def audit(root, data_root):
    root, data_root = Path(root), Path(data_root)
    state = read(root/'suite_status.json')
    expected_workers = [f'{method}_repeat{i:02d}' for method in ('ours', 'qwen') for i in (1, 2, 3)]
    if state['status'] != 'complete' or not state.get('mps_released') or state['completed'] != expected_workers:
        raise ValueError('Six complete original runs and MPS release required')
    contract = read(root/'cpu_contract/cpu_contract.json')
    if contract['status'] != 'passed' or not contract['bitwise_equal'] or contract['max_abs_difference'] != 0:
        raise ValueError('Original/instrumented real-image CPU contract required')
    annotation = data_root/'joints.mat'
    if sha(annotation) != ANNOTATION_SHA or sha(root/'audit_original/joints.mat') != ANNOTATION_SHA:
        raise ValueError('Wrong original annotation identity')
    original = np.asarray(loadmat(annotation)['joints'], dtype=np.float32)
    if original.shape != (3, 14, 2000):
        raise ValueError('Not canonical original LSP2000')
    original = original.transpose(2, 1, 0)
    ids = [f'lsp_{i:05d}' for i in range(1001, 2001)]
    images = []
    for i in range(1001, 1201):
        path = data_root/'images'/f'im{i:04d}.jpg'
        with Image.open(path) as image:
            size = list(image.size)
        images.append((path.name, sha(path), size))
    rows = []
    stage_details = []
    source_bindings, external_bindings = set(), set()
    input_signature = None
    for method in ('ours', 'qwen'):
        previous = None
        for repeat in (1, 2, 3):
            folder = root/f'{method}_repeat{repeat:02d}'
            identity, summary = read(folder/'identity.json'), read(folder/'summary.json')
            samples, stages = csv_rows(folder/'samples.csv'), csv_rows(folder/'stages.csv')
            settings = identity['resolved_settings']
            if (identity['system'] != method or identity['repeat'] != repeat
                    or identity['checkpoint_sha256'] != PINS[method]
                    or identity['annotation_sha256'] != ANNOTATION_SHA
                    or identity['explicit_warmup_forwards'] != 0 or identity['test_ids'] != ids
                    or settings['image_size'] != 224 or settings['visibility_policy'] != 'coordinates_in_image'
                    or settings['crop_margin'] != 1.25 or summary['status'] != 'complete'
                    or summary['valid_samples'] != 200 or summary['repeat'] != repeat):
                raise ValueError('Wrong checkpoint, annotation, order or protocol')
            command = identity['command']
            if command[command.index('--gpu-uuid')+1] != GPU or GPU not in identity['gpu_admission']['gpu']:
                raise ValueError('Wrong bound physical GPU')
            if '250.00' not in identity['gpu_admission']['gpu']:
                raise ValueError('Wrong A100 power cap')
            for record in read(folder/'files_manifest.json'):
                p = folder/record['file']
                if sha(p) != record['sha256'] or p.stat().st_size != record['bytes']:
                    raise ValueError('Changed original raw artifact')
            if len(samples) != 200 or [r['sample_id'] for r in samples] != ids[:200]:
                raise ValueError('Wrong first200 TEST IDs')
            arrays = np.load(folder/'raw_pose_outputs.npz', allow_pickle=False)
            predicted, targets, visible = (arrays[k] for k in ('predicted', 'targets', 'visible'))
            if (predicted.shape != (200, 14, 2) or targets.shape != predicted.shape or visible.shape != (200, 14)
                    or predicted.dtype != np.float32 or targets.dtype != np.float32 or visible.dtype != bool):
                raise ValueError('Wrong raw pose arrays')
            if not np.isfinite(predicted).all():
                raise ValueError('Invalid predictions')
            if previous is not None and not np.array_equal(predicted, previous):
                raise ValueError('Repeated predictions differ; retain and investigate')
            previous = predicted.copy()
            hits = joints = 0
            signature = []
            for index, (sample, image) in enumerate(zip(samples, images)):
                target, mask, crop = target_from_original(original[1000+index])
                if not np.allclose(target, targets[index], atol=3e-5, rtol=0, equal_nan=True) or not np.array_equal(mask, visible[index]):
                    raise ValueError('Original GT/crop versus saved target differs')
                if (Path(sample['image_path']).name != image[0] or sample['image_sha256'] != image[1]
                        or json.loads(sample['original_size']) != image[2]
                        or json.loads(sample['processed_size']) != [224, 224]
                        or json.loads(sample['pixel_values_shape']) != [196, 1536]
                        or json.loads(sample['image_grid_thw']) != [[1, 14, 14]]):
                    raise ValueError('Wrong original input or dimensions')
                if not np.allclose(json.loads(sample['crop_box']), crop, atol=3e-5, rtol=0):
                    raise ValueError('Original person crop differs')
                for key, array in (('predicted_coordinates', predicted[index]), ('target_coordinates', targets[index])):
                    if not np.array_equal(np.asarray(json.loads(sample[key]), dtype=np.float32), array, equal_nan=True):
                        raise ValueError('NPZ/CSV pose differs')
                if not np.array_equal(ast.literal_eval(sample['visible']), visible[index]):
                    raise ValueError('NPZ/CSV visibility differs')
                hit, count, torso, head = pose_metric(predicted[index], targets[index], visible[index])
                if hit != int(sample['pck_hits']) or count != int(sample['pck_joints']):
                    raise ValueError('Per-sample PCK differs')
                close(torso, float(sample['torso_scale'])); close(head, float(sample['head_scale']))
                hits += hit; joints += count
                signature.append((sample['sample_id'], *image))
            if input_signature is not None and input_signature != signature:
                raise ValueError('Repeated or cross-model input identities differ')
            input_signature = signature
            if hits != summary['pck_hits'] or joints != summary['evaluated_joints']:
                raise ValueError('PCK numerator/denominator differs')
            close(hits/joints, summary['measured200_pck'])
            bindings = [('LightGenV2/reports/20260928_nine_task_a100_table/measurement_protocol_20261024/measure_lsp.py', identity['script_sha256']),
                        ('LightGenV2/reports/20260928_nine_task_a100_table/measurement_protocol_20261024/measure_salicon.py', identity['salicon_scope_sha256']),
                        (identity['config'].split('/2026OpticsMoE/', 1)[1], identity['config_sha256'])]
            for entry in read(folder/'stage_ledger.json'):
                if '/2026OpticsMoE/' in entry['source_file']:
                    bindings.append((entry['source_file'].split('/2026OpticsMoE/', 1)[1], entry['source_sha256']))
                else:
                    external_bindings.add((entry['source_file'], entry['source_sha256']))
            for name, digest in set(bindings):
                blob = subprocess.check_output(['git', 'show', identity['source_commit']+':'+name], cwd=REPO)
                if hashlib.sha256(blob).hexdigest() != digest:
                    raise ValueError('Measured source not bound to published Git')
                source_bindings.add((name, digest))
            events = read(folder/'process_audit.json')
            own = None
            if len(events) < 2:
                raise ValueError('Missing process/MPS occupation evidence')
            for event in events:
                if len(event['clients']) != 1:
                    raise ValueError('Foreign or missing MPS client')
                own = event['clients'][0] if own is None else own
                if event['clients'] != [own] or any(int(p[0]) not in {own, *event['servers']} or p[1].strip() != GPU for p in event['processes']):
                    raise ValueError('Foreign GPU process')
            counts, totals, spans = Counter(), defaultdict(float), defaultdict(list)
            charged = nn_wall = nn_event = 0.
            for stage in stages:
                i = int(stage['sample_index'])
                start, end = float(stage['start_monotonic']), float(stage['end_monotonic'])
                if i not in range(200) or start > end or start < float(samples[i]['input_start']) or end > float(samples[i]['cpu_result_return']):
                    raise ValueError('Stage outside own inference')
                close(1000*(end-start), float(stage['synchronized_wall_ms']))
                spans[i].append((start, end)); counts[stage['stage']] += 1
                totals[stage['stage']] += float(stage['synchronized_wall_ms'])
                if stage['family'] in ('simulation_only', 'parallel_residual', 'layout', 'fusion') and stage['charged'] != 'False':
                    raise ValueError('Excluded arithmetic charged')
                if stage['charged'] == 'True':
                    charged += float(stage['synchronized_wall_ms'])
                    if stage['cuda_event_ms']:
                        nn_wall += float(stage['synchronized_wall_ms']); nn_event += float(stage['cuda_event_ms'])
            for intervals in spans.values():
                if any(a[1] > b[0]+1e-8 for a, b in zip(intervals, intervals[1:])):
                    raise ValueError('Overlapping timing spans')
            for name in ('input.file_person_crop_processor_h2d', 'output.pose_hardargmax', 'output.cpu_coordinates'):
                if counts[name] != 200:
                    raise ValueError('Missing frontend or task output')
            if method == 'ours':
                expected = {'bridge.native_merger': 200, 'vision.router_simulation': 200,
                            'vision.expert_global_simulation': 400, 'vision.parallel_residual1': 200,
                            'vision.parallel_residual2': 200, 'head.pose_heatmap_decoder': 200}
            else:
                expected = {'qwen.full_native_vision_pose_head': 200}
            if any(counts[name] != count for name, count in expected.items()):
                raise ValueError('Wrong native merger/head/three-pass occurrence count')
            for index, sample in enumerate(samples):
                gap = 0. if index == 0 else 1000*(float(sample['input_start'])-float(samples[index-1]['cpu_result_return']))
                close(gap, float(sample['internal_gap_ms'])); charged += gap
            start, end = float(samples[0]['input_start']), float(samples[-1]['cpu_result_return'])
            seconds = end-start
            close(seconds, summary['actual_software_wall_seconds'])
            board, max_age = board_energy(csv_rows(folder/'telemetry.csv'), start, end)
            if method == 'ours':
                optical = 200*3*1.0447
                duration = (charged+optical)/1000
                energy = (121.746*optical+104.230*charged)/1000
                close(charged, summary['charged_electronic_ms_total']); close(optical, summary['optical_ms_total'])
            else:
                optical = 0.; charged = seconds*1000
                duration = seconds; energy = 338.2*seconds+board
                close(energy, summary['power_integration']['energy_j'])
                close(board/seconds, summary['mean_board_power_w'])
            close(duration*5, summary['cost']['sample_ms']); close(energy/200, summary['cost']['sample_j'])
            rows.append(dict(system=method, repeat=repeat, sample_ms=duration*5, sample_j=energy/200,
                             optical_ms=optical/200, charged_electronic_ms=charged/200,
                             subset200_pck=hits/joints, pck_hits=hits, evaluated_joints=joints,
                             actual_board_w=board/seconds, actual_software_wall_s=seconds,
                             actual_gpu_energy_j=board, max_power_support_age_s=max_age,
                             nn_wall_event_ratio=nn_wall/nn_event, summary_sha256=sha(folder/'summary.json')))
            for name in counts:
                example = next(s for s in stages if s['stage'] == name)
                stage_details.append(dict(system=method, repeat=repeat, stage=name, family=example['family'],
                                          charged=example['charged'], occurrences=counts[name], ms_per_sample=totals[name]/200))
    aggregate = {}
    for method in ('ours', 'qwen'):
        selected = [r for r in rows if r['system'] == method]
        aggregate[method] = {k:sum(r[k] for r in selected)/3 for k in
                             ('sample_ms', 'sample_j', 'optical_ms', 'charged_electronic_ms', 'subset200_pck')}
        aggregate[method]['time_weighted_actual_board_w'] = sum(r['actual_gpu_energy_j'] for r in selected)/sum(r['actual_software_wall_s'] for r in selected)
    return dict(status='passed', calls_per_system=600, unique_test_images=200, original_full_test_images=1000,
                original_annotation_sha256=ANNOTATION_SHA, original_image_evidence=input_signature,
                repeats=rows, aggregate=aggregate, stage_details=stage_details,
                source_bindings=sorted(source_bindings), external_dependency_bindings=sorted(external_bindings),
                original_reference_not_subset=True, common_tokens_per_image=196,
                op_count_status='pending matching-input replay; no historical operation count copied')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'data-root', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve previous audit')
    result = audit(args.root, args.data_root)
    with args.output.open('x', encoding='utf8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k:result[k] for k in ('status', 'calls_per_system', 'aggregate')}))
