"""Offline six-stage replay of the sealed rank72 laboratory computation.

Extracted from the reviewed Windows snapshot's process_batch and simulation
snapshot, with paths, processor/SLM imports and global calibration removed.
The legacy snapshot main (CUDA and optional simulated gallery) is NOT adopted.
No capture, training, dataset evaluation or device ownership is started here.
"""
import numpy as np
import torch
from torch.nn import functional as F
from ..standalone.model import fuse
from ..standalone.bounded_export import stage_active

STAGES = ('vision_router', 'vision_expert', 'vision_global',
          'language_router', 'language_expert', 'language_global')

def pcc(a,b):
    x=np.asarray(a,dtype=np.float64).ravel();y=np.asarray(b,dtype=np.float64).ravel()
    x-=x.mean();y-=y.mean();d=np.linalg.norm(x)*np.linalg.norm(y)
    return float(x.dot(y)/d) if d else 0.0


def snapshot_simulation(model,batch):
    vector=model(batch).float().cpu();result={'descriptor':vector}
    for mode in ('vision','language'):
        branch=getattr(model,mode).optics
        result[f'{mode}_router']=branch.router.last['intensity'].float().cpu()
        result[f'{mode}_router_probabilities']=branch.router.last['probabilities'].float().cpu()
        result[f'{mode}_expert']=branch.last_ccd['expert'].float().cpu()
        result[f'{mode}_global']=branch.last_ccd['global'].float().cpu()
    return result


def replay_batch(model, batch, ids, capture_stage):
    """Replay the actual six-stage graph with an explicit CCD callback.

    The callback receives (stage, bounded_active_amplitude, sample_ids) and
    returns (CPU CCD tensor, receipt, amplitude_scale). It must not normalize
    CCD values. This module opens no hardware and reads no dataset or files.
    """
    if any(parameter.device.type != 'cpu' for parameter in model.parameters()):
        raise ValueError('The sealed hardware graph requires CPU model replay')
    if len(ids) != batch['input_ids'].shape[0] or len(set(ids)) != len(ids):
        raise ValueError('Sample identities must match the batch and be unique')
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
            'vision_router', stage_active(amplitude), ids)
        measured['vision_router'] = ccd.cpu()
        v.optics.router.measured_ccd = ccd
        weights = v.optics.router(amplitude)
        vision_router_probabilities = v.optics.router.last['probabilities'].float().cpu()

        ccd, receipts['vision_expert'], scales['vision_expert'] = capture_stage(
            'vision_expert', stage_active(amplitude, weights), ids)
        measured['vision_expert'] = ccd.cpu()
        o1 = v.optics.decode(ccd, latent.shape[1], latent.dtype, False)
        f1 = fuse(e1, o1, v.block1_optical_fusion_logit, v.alpha_bounds)
        e2 = v.blocks[1](f1)
        global_amp = v.optics.encode(f1)
        ccd, receipts['vision_global'], scales['vision_global'] = capture_stage(
            'vision_global', stage_active(global_amp, weights), ids)
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
            'language_router', stage_active(amplitude), ids)
        measured['language_router'] = ccd.cpu()
        l.optics.router.measured_ccd = ccd
        weights = l.optics.router(amplitude)
        language_router_probabilities = l.optics.router.last['probabilities'].float().cpu()

        ccd, receipts['language_expert'], scales['language_expert'] = capture_stage(
            'language_expert', stage_active(amplitude, weights), ids)
        measured['language_expert'] = ccd.cpu()
        o1 = l.optics.decode(ccd, latent.shape[1], latent.dtype, False)
        f1 = fuse(e1, o1, l.block1_optical_fusion_logit, l.alpha_bounds)
        e2 = l.blocks[1](f1)
        global_amp = l.optics.encode(f1)
        ccd, receipts['language_global'], scales['language_global'] = capture_stage(
            'language_global', stage_active(global_amp, weights), ids)
        measured['language_global'] = ccd.cpu()
        o2 = l.optics.decode(ccd, latent.shape[1], latent.dtype, True)
        f2 = fuse(e2, o2, l.block2_optical_fusion_logit, l.alpha_bounds)
        lo = l.output_norm(f2)
        image_positions = batch['input_ids'].eq(model.metadata['image_token_id'])
        if model.late_rgb is not None:
            rgb = patches.float().reshape(len(ids), 7, 2, 7, 2, 1024).mean((2, 4)).reshape(len(ids), 49, 1024)
            bypass = model.late_rgb(rgb).reshape(-1, 192)
            lo = lo.clone()
            lo[image_positions] = .5 * lo[image_positions] + .5 * bypass
        descriptor = model.readout(lo, image_positions).float().cpu()

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
