"""Full 800-query physical evaluation using the accepted four-query calibration.

The hardware is kept open for the whole run.  Four queries are processed per
batch, completed batches are checkpointed, and an interrupted run resumes
without recapturing completed batches.  The simulated 1600-image gallery is
fixed; only the 800 query descriptors pass through the six physical stages.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from transformers import AutoProcessor
import four_image_flow as physical_geometry

from four_image_flow import (
    BEST, PROJECT, Bench, OpticalRetrieval, active_to_native, inputs, orient,
    phase_gray, phase_planes, picture, pcc, save_amplitudes, sha,
    snapshot_simulation, stage_active, write, fuse,
)


STAGE_CALIBRATION = {}


def selected_phase(radians, name):
    spatial = name.split('_')[0]
    return phase_gray(radians, spatial, name.endswith('_inverse'))


def remove_amplitudes(paths):
    for path in paths:
        path.unlink(missing_ok=True)


def capture_stage(bench, out, stage, phase_path, active, ids, orientation):
    paths, scale = save_amplitudes(active, out, stage, ids)
    try:
        raw, receipt = bench.capture(stage, phase_path, paths, ids, orientation, save=True)
    finally:
        remove_amplitudes(paths)
    return torch.from_numpy(raw).to('cuda').float(), receipt, scale


def process_batch(model, processor, bench, out, phase_paths, rows):
    device = torch.device('cuda')
    ids = [r['sample_id'] for r in rows]
    images = [picture(PROJECT/'data'/r['image_path'], model.metadata.get('input_preprocessing', 'contain_white')) for r in rows]
    batch = inputs(processor, images, device)
    with torch.inference_mode():
        # A previous physical batch leaves router injection tensors attached.
        # Clear them before producing this batch's all-simulation reference.
        model.vision.optics.router.measured_ccd = None
        model.language.optics.router.measured_ccd = None
        sim = snapshot_simulation(model, batch)
        measured = {}
        receipts = {}
        scales = {}

        v = model.vision
        patches = model.frontend.patches(batch['pixel_values'], len(ids))
        latent = v.input_norm(v.input_adapter(patches.float()))
        e1 = v.blocks[0](latent)
        amplitude = v.optics.encode(latent)
        ccd, receipts['vision_router'], scales['vision_router'] = capture_stage(
            bench, out, 'vision_router', phase_paths['vision_router'],
            stage_active(amplitude), ids, STAGE_CALIBRATION['vision_router'][1])
        measured['vision_router'] = ccd.cpu()
        v.optics.router.measured_ccd = ccd
        weights = v.optics.router(amplitude)
        vision_router_probabilities = v.optics.router.last['probabilities'].float().cpu()

        ccd, receipts['vision_expert'], scales['vision_expert'] = capture_stage(
            bench, out, 'vision_expert', phase_paths['vision_expert'],
            stage_active(amplitude, weights), ids, STAGE_CALIBRATION['vision_expert'][1])
        measured['vision_expert'] = ccd.cpu()
        o1 = v.optics.decode(ccd, latent.shape[1], latent.dtype, False)
        f1 = fuse(e1, o1, v.block1_optical_fusion_logit, v.alpha_bounds)
        e2 = v.blocks[1](f1)
        global_amp = v.optics.encode(f1)
        ccd, receipts['vision_global'], scales['vision_global'] = capture_stage(
            bench, out, 'vision_global', phase_paths['vision_global'],
            stage_active(global_amp, weights), ids, STAGE_CALIBRATION['vision_global'][1])
        measured['vision_global'] = ccd.cpu()
        o2 = v.optics.decode(ccd, latent.shape[1], latent.dtype, True)
        f2 = fuse(e2, o2, v.block2_optical_fusion_logit, v.alpha_bounds)
        vo = v.output_norm(f2)
        vision = (patches.float() + torch.sigmoid(v.residual_logit) * v.output_adapter(vo)).to(patches.dtype)

        image_features = model.frontend.merge(vision)
        emb = model.frontend.embed(batch['input_ids'])
        mask = batch['input_ids'].eq(model.metadata['image_token_id']).unsqueeze(-1).expand_as(emb)
        emb = emb.masked_scatter(mask, image_features.to(emb.dtype))
        l = model.language
        latent = l.input_norm(l.input_adapter(emb.float()))
        e1 = l.blocks[0](latent)
        amplitude = l.optics.encode(latent)
        ccd, receipts['language_router'], scales['language_router'] = capture_stage(
            bench, out, 'language_router', phase_paths['language_router'],
            stage_active(amplitude), ids, STAGE_CALIBRATION['language_router'][1])
        measured['language_router'] = ccd.cpu()
        l.optics.router.measured_ccd = ccd
        weights = l.optics.router(amplitude)
        language_router_probabilities = l.optics.router.last['probabilities'].float().cpu()

        ccd, receipts['language_expert'], scales['language_expert'] = capture_stage(
            bench, out, 'language_expert', phase_paths['language_expert'],
            stage_active(amplitude, weights), ids, STAGE_CALIBRATION['language_expert'][1])
        measured['language_expert'] = ccd.cpu()
        o1 = l.optics.decode(ccd, latent.shape[1], latent.dtype, False)
        f1 = fuse(e1, o1, l.block1_optical_fusion_logit, l.alpha_bounds)
        e2 = l.blocks[1](f1)
        global_amp = l.optics.encode(f1)
        ccd, receipts['language_global'], scales['language_global'] = capture_stage(
            bench, out, 'language_global', phase_paths['language_global'],
            stage_active(global_amp, weights), ids, STAGE_CALIBRATION['language_global'][1])
        measured['language_global'] = ccd.cpu()
        o2 = l.optics.decode(ccd, latent.shape[1], latent.dtype, True)
        f2 = fuse(e2, o2, l.block2_optical_fusion_logit, l.alpha_bounds)
        lo = l.output_norm(f2)
        descriptor = model.readout(lo, batch['input_ids'].eq(model.metadata['image_token_id'])).float().cpu()

    stage_pcc = {name: [pcc(x, y.numpy()) for x, y in zip(value, sim[name])]
                 for name, value in measured.items()}
    return {
        'ids': ids,
        'descriptor': descriptor,
        'simulation_descriptor': sim['descriptor'],
        'stage_pcc': stage_pcc,
        'descriptor_cosine': F.cosine_similarity(descriptor, sim['descriptor']).tolist(),
        'router_probabilities': {
            'vision': vision_router_probabilities,
            'language': language_router_probabilities,
        },
        'phase_receipts': receipts,
        'amplitude_scales': scales,
    }


def evaluate(descriptor, rows, protocol, bank):
    metadata = {r['sample_id']: r for r in protocol['rows']}
    gallery_indices = [i for i, sid in enumerate(bank['ids']) if metadata[sid]['split'] == 'train']
    gallery_ids = [bank['ids'][i] for i in gallery_indices]
    gallery = F.normalize(bank['vectors'][gallery_indices].float(), dim=-1)
    scores = F.normalize(descriptor.float(), dim=-1) @ gallery.T
    order = scores.argsort(dim=1, descending=True)
    predictions = []
    reciprocal = []
    hit1 = hit5 = hit10 = 0
    for row, ranked in zip(rows, order.tolist()):
        products = [metadata[gallery_ids[i]]['product_id'] for i in ranked]
        ranks = [i + 1 for i, product in enumerate(products) if product == row['product_id']]
        rank = ranks[0]
        hit1 += rank <= 1; hit5 += rank <= 5; hit10 += rank <= 10; reciprocal.append(1.0 / rank)
        top = ranked[0]
        predictions.append({
            'sample_id': row['sample_id'], 'product_id': row['product_id'],
            'top1_sample_id': gallery_ids[top], 'top1_product_id': metadata[gallery_ids[top]]['product_id'],
            'rank_of_first_relevant': rank, 'hit_at_1': int(rank == 1),
        })
    n = len(rows)
    return predictions, {
        'recall_at_1': hit1 / n,
        'recall_at_5': hit5 / n,
        'recall_at_10': hit10 / n,
        'mrr': float(np.mean(reciprocal)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--exposure-us', type=float, default=20000)
    p.add_argument('--wait-ms', type=float, default=240)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--simulation-bank', type=Path, required=True)
    p.add_argument('--physical-gallery', action='store_true',
                   help='Recapture all 1600 gallery and 800 query images; evaluate measured-to-measured.')
    p.add_argument('--max-batches', type=int, default=0,
                   help='Stop after this many batches for hardware smoke; zero runs all.')
    p.add_argument('--calibration-report', type=Path, required=True)
    p.add_argument('--corners-tltrbrbl', type=float, nargs=8,
                   metavar=('TL_X','TL_Y','TR_X','TR_Y','BR_X','BR_Y','BL_X','BL_Y'))
    args = p.parse_args()
    if args.corners_tltrbrbl is not None:
        physical_geometry.BASE_CORNERS = np.asarray(args.corners_tltrbrbl, dtype=np.float32).reshape(4, 2)
    if not 100 <= args.exposure_us <= 20000 or not 150 <= args.wait_ms <= 1000:
        raise ValueError('Unsafe exposure/wait bounds')
    if args.batch_size < 1 or args.batch_size > 8:
        raise ValueError('Batch size must be 1..8')
    if sha(PROJECT/'assets/best.pt') != BEST:
        raise ValueError('Wrong best.pt')
    calibration = json.loads(args.calibration_report.read_text(encoding='utf-8'))
    if calibration.get('status') != 'complete' or calibration.get('checkpoint_sha256') != BEST:
        raise ValueError('Four-query calibration is incomplete or uses another checkpoint')
    if not np.array_equal(np.asarray(calibration['base_corners_screen_TL_TR_BR_BL'], np.float32),
                          physical_geometry.BASE_CORNERS):
        raise ValueError('Four-query calibration uses different camera corners')
    stages = ('vision_router', 'vision_expert', 'vision_global',
              'language_router', 'language_expert', 'language_global')
    if set(calibration['stage_calibration']) != set(stages):
        raise ValueError('Four-query calibration does not cover six stages')
    candidates = {f'{spatial}_{tone}' for spatial in ('none', 'h', 'v', 'hv')
                  for tone in ('normal', 'inverse')}
    orientations = {'identity', 'flip_h', 'flip_v', 'rot180', 'transpose',
                    'rot90', 'rot270', 'anti_transpose'}
    STAGE_CALIBRATION.update({stage: (calibration['stage_calibration'][stage]['phase_candidate'],
                                      calibration['stage_calibration'][stage]['camera_orientation'])
                              for stage in stages})
    if any(phase not in candidates or orientation not in orientations
           for phase, orientation in STAGE_CALIBRATION.values()):
        raise ValueError('Invalid phase/orientation in four-query calibration')
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((PROJECT/'protocol.json').read_text(encoding='utf-8'))
    query_rows = [row for row in protocol['rows'] if row['split'] == 'query']
    gallery_rows = [row for row in protocol['rows'] if row['split'] == 'train']
    if len(query_rows) != 800 or len(gallery_rows) != 1600:
        raise ValueError('Expected 800 queries and 1600 gallery images')
    rows = gallery_rows + query_rows if args.physical_gallery else query_rows
    if len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate sample IDs')

    payload = torch.load(PROJECT/'assets/best.pt', map_location='cpu', weights_only=True)
    model = OpticalRetrieval(payload['metadata'])
    model.load_state_dict(payload['state_dict'], strict=True)
    model.to('cuda').eval().requires_grad_(False)
    processor = AutoProcessor.from_pretrained(str(PROJECT/'assets/processor'), local_files_only=True)
    phases = phase_planes(model); phase_dir = out/'phase'; phase_dir.mkdir(exist_ok=True)
    phase_paths = {}
    for stage, radians in phases.items():
        name = STAGE_CALIBRATION[stage][0]
        path = phase_dir/(stage+'.bmp')
        if not path.exists(): Image.fromarray(selected_phase(radians, name)).save(path)
        phase_paths[stage] = path
    run_contract = {
        'schema': 1, 'query_count': len(rows), 'batch_size': args.batch_size,
        'exposure_us': args.exposure_us, 'wait_ms': args.wait_ms,
        'stage_calibration': STAGE_CALIBRATION, 'checkpoint_sha256': BEST,
        'calibration_report_sha256': sha(args.calibration_report),
        'simulation_bank_sha256': sha(args.simulation_bank),
        'corners_screen_TL_TR_BR_BL': physical_geometry.BASE_CORNERS.tolist(),
        'scope': ('800 physical queries against 1600 physical gallery images'
                  if args.physical_gallery else '800 physical queries against fixed 1600 simulated gallery'),
    }
    contract_path = out/'run_contract.json'
    # Match the representation saved on disk (JSON turns tuples into lists).
    # This preserves strict comparisons of every actual acquisition setting.
    run_contract = json.loads(json.dumps(run_contract))
    if contract_path.exists():
        existing = json.loads(contract_path.read_text(encoding='utf-8'))
        if existing != run_contract:
            raise ValueError('Resume contract differs; refusing to mix hardware runs')
    else:
        write(contract_path, run_contract)

    batch_dir = out/'batch_results'; batch_dir.mkdir(exist_ok=True)
    completed = []
    with Bench(out, args.exposure_us, args.wait_ms, phase_paths) as bench:
        for start in range(0, len(rows), args.batch_size):
            part = rows[start:start+args.batch_size]
            result_path = batch_dir/(f'{start:04d}.pt')
            if result_path.exists():
                result = torch.load(result_path, map_location='cpu', weights_only=True)
                print(json.dumps({'resume_batch': start, 'count': len(part)}), flush=True)
            else:
                result = process_batch(model, processor, bench, out, phase_paths, part)
                result['rows'] = part
                torch.save(result, result_path)
                write(out/'progress.json', {
                    'status': 'running', 'completed_queries': start + len(part),
                    'total_queries': len(rows), 'last_batch': start,
                })
                print(json.dumps({'completed_queries': start+len(part), 'total_queries': len(rows)}), flush=True)
            completed.append(result)
            if args.max_batches and len(completed) >= args.max_batches:
                print(json.dumps({'status': 'smoke_complete', 'completed_images': sum(len(x['ids']) for x in completed),
                                  'total_images': len(rows)}), flush=True)
                return
        capture_rows = bench.rows

    descriptors = torch.cat([x['descriptor'] for x in completed])
    simulation_descriptors = torch.cat([x['simulation_descriptor'] for x in completed])
    if args.physical_gallery:
        bank = {'ids': [r['sample_id'] for r in gallery_rows], 'vectors': descriptors[:1600]}
        predictions, metrics = evaluate(descriptors[1600:], query_rows, protocol, bank)
    else:
        bank = torch.load(args.simulation_bank, map_location='cpu', weights_only=True)
        predictions, metrics = evaluate(descriptors, query_rows, protocol, bank)
    stage_pcc = {stage: [v for result in completed for v in result['stage_pcc'][stage]]
                 for stage in STAGE_CALIBRATION}
    report = {
        'schema': 1, 'status': 'complete', 'metrics': metrics,
        'query_count': len(query_rows), 'gallery_count': 1600,
        'captured_image_count': len(rows), 'physical_gallery': args.physical_gallery,
        'checkpoint_sha256': BEST, 'exposure_requested_us': args.exposure_us,
        'camera_actual': capture_rows[0]['exposure'] if capture_rows else None,
        'settle_delay_ms': args.wait_ms, 'stage_calibration': STAGE_CALIBRATION,
        'calibration_report_sha256': run_contract['calibration_report_sha256'],
        'corners_screen_TL_TR_BR_BL': physical_geometry.BASE_CORNERS.tolist(),
        'mean_stage_pcc_to_simulation': {k: float(np.mean(v)) for k, v in stage_pcc.items()},
        'mean_descriptor_cosine_to_simulation': float(F.cosine_similarity(descriptors, simulation_descriptors).mean()),
        'maximum_saturation_fraction': max((x['saturation_fraction'] for x in capture_rows), default=0.0),
        'predictions': predictions, 'no_per_image_photometric_normalization': True,
        'scope': run_contract['scope'],
    }
    torch.save({'ids': [r['sample_id'] for r in rows], 'descriptors': descriptors,
                'simulation_descriptors': simulation_descriptors}, out/'features.pt')
    write(out/'report.json', report)
    write(out/'progress.json', {'status': 'complete', 'completed_queries': len(rows), 'total_queries': len(rows)})
    print(json.dumps({'status': 'complete', 'metrics': metrics,
                      'mean_stage_pcc': report['mean_stage_pcc_to_simulation']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
