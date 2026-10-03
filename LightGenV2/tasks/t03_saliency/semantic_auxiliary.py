"""Removable training-only semantic classifier; never registered in the student."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def box_coverage_targets(boxes, column):
    """Weak 14x14 area coverage, maximum across same-category boxes, not masks.

    Return raster [196,80]. Partial cells are soft labels; narrow boxes survive.
    Max avoids double-counting overlapping instances, but is not exact union.
    """
    result = np.zeros((14,14,80), dtype=np.float32)
    lower = np.arange(14, dtype=np.float64)/14
    upper = lower + 1/14
    for box in boxes:
        coords = box.get('xyxy')
        if (box.get('category_id') not in column or not isinstance(coords,list) or len(coords)!=4
                or any(type(v) not in (int,float) or not math.isfinite(v) for v in coords)):
            raise ValueError('Invalid spatial semantic box')
        x0,y0,x1,y1 = coords
        if not (0<=x0<x1<=1 and 0<=y0<y1<=1):
            raise ValueError('Spatial semantic box outside unit image')
        x = np.maximum(0, np.minimum(upper,x1)-np.maximum(lower,x0))*14
        y = np.maximum(0, np.minimum(upper,y1)-np.maximum(lower,y0))*14
        index = column[box['category_id']]
        result[:,:,index] = np.maximum(result[:,:,index],y[:,None]*x[None,:])
    return torch.from_numpy(result.reshape(196,80))


class SemanticAuxiliary(nn.Module):
    def __init__(self, settings, dataset):
        super().__init__()
        self.mode = getattr(settings,'semantic_mode','image_presence')
        if self.mode not in ('image_presence','box_coverage'):
            raise ValueError('Unsupported semantic auxiliary mode')
        purpose = ('training_only_auxiliary_object_regions_not_saliency_ground_truth'
                   if self.mode=='box_coverage' else
                   'training_only_auxiliary_object_presence_not_saliency_ground_truth')
        contents = Path(settings.semantic_targets).read_bytes()
        if hashlib.sha256(contents).hexdigest() != settings.semantic_targets_sha256:
            raise ValueError('Semantic targets SHA mismatch')
        raw = json.loads(contents)
        if (raw.get('schema_version') != 1
                or raw.get('purpose') != purpose
                or raw.get('image_manifest_sha256') != dataset.manifest_sha256
                or raw.get('train_overlap') != 0 or raw.get('test_overlap') != 0
                or raw.get('additional_human_semantic_supervision') is not True
                or raw.get('no_saliency_or_fixation_targets_created') is not True):
            raise ValueError('Semantic target provenance mismatch')
        records, categories = raw['records'], raw['categories']
        ids = [row['sample_id'] for row in records]
        if ids != dataset.sample_ids or len(set(ids)) != len(ids) or len(ids) != raw['sample_count']:
            raise ValueError('Semantic target identity/order mismatch')
        category_ids = [row['id'] for row in categories]
        if len(category_ids) != 80 or category_ids != sorted(set(category_ids)):
            raise ValueError('Invalid semantic category order')
        column = {key: i for i,key in enumerate(category_ids)}
        self.targets = torch.zeros(len(ids),80)  # Remains on CPU; not checkpoint data.
        for index,row in enumerate(records):
            positives = row['positive_category_ids']
            if positives != sorted(set(positives)) or any(c not in column for c in positives):
                raise ValueError('Invalid semantic positive category set')
            if row['sample_id'] != f"unlabeled/coco2017/{row['image_id']:012d}":
                raise ValueError('Semantic numeric ID/sample ID mismatch')
            self.targets[index,[column[c] for c in positives]] = 1.
        counts = self.targets.sum(0)
        if counts.long().tolist() != raw['category_positive_image_counts']:
            raise ValueError('Semantic category statistics mismatch')
        if self.mode == 'box_coverage':
            if (raw.get('additional_human_box_supervision') is not True or raw.get('image_resize_contract') !=
                    'direct anisotropic RGB BICUBIC resize to 224x224; no crop/EXIF transpose'):
                raise ValueError('Spatial semantic resize/provenance mismatch')
            self.boxes = [row['boxes_xyxy_unit'] for row in records]
            if sum(map(len,self.boxes)) != raw.get('spatial_box_count'):
                raise ValueError('Spatial semantic box count mismatch')
            self.column = column
            # Accumulate priors once; retain compact boxes, not a 1.25GB dense table.
            counts = torch.zeros(80,dtype=torch.float64)
            for row in records:
                if any(b['category_id'] not in row['positive_category_ids'] for b in row['boxes_xyxy_unit']):
                    raise ValueError('Spatial box category absent from image labels')
                counts += box_coverage_targets(row['boxes_xyxy_unit'],column).double().sum(0)
            counts = counts.float()
        observations = len(ids)*(196 if self.mode=='box_coverage' else 1)
        self.index = {key:i for i,key in enumerate(ids)}
        self.width = int(settings.electronic_width)
        # Keep the global training RNG untouched, including CUDA optical draws.
        with torch.random.fork_rng(devices=[]):
            generator = torch.Generator().manual_seed(settings.random_seed + 2281)
            self.head = nn.Linear(self.width,80)
            nn.init.normal_(self.head.weight,std=.001,generator=generator)
            with torch.no_grad():
                prior = (counts/observations).clamp(.0001,.9999)
                self.head.bias.copy_(torch.logit(prior))
        positive_weight = ((observations-counts)/counts.clamp_min(1)).sqrt().clamp(1,10)
        self.register_buffer('positive_weight',positive_weight,persistent=False)
        self.provenance = {
            'targets_sha256': settings.semantic_targets_sha256,
            'image_manifest_sha256': dataset.manifest_sha256,
            'annotation_sha256': raw['annotation_sha256'],
            'samples': len(ids), 'categories': categories,
            'positive_image_counts': raw['category_positive_image_counts'],
            'mode': self.mode,
            'positive_target_mass': counts.tolist(),
            'target_observations_per_category': observations,
            'positive_weight': positive_weight.tolist(),
            'positive_weight_rule': 'sqrt(negative/positive), clipped to [1,10]',
            'training_only_parameters': sum(p.numel() for p in self.parameters()),
            'inference_parameters_added': 0,
            'additional_human_semantic_supervision': True,
            'feature_source': 'final fused student latent -> spatial mean -> non-affine LN -> Linear(192,80)',
            'checkpoint_state': 'training_only_semantic (live auxiliary head, not EMA inference weights)',
        }
        if self.mode=='box_coverage':
            self.provenance.update({
                'additional_human_box_supervision': True,
                'feature_source': 'final fused [196,192] -> restore Qwen block-major to raster -> per-token LN -> Linear(192,80)',
                'spatial_target': '14x14 fractional cell coverage, max over same-category boxes; NOT segmentation/fixation GT',
                'spatial_boxes': raw['spatial_box_count'],
            })

    def forward(self, groups, sample_ids):
        if not sample_ids or len(set(sample_ids)) != len(sample_ids) or len(groups) != len(sample_ids):
            raise ValueError('Invalid semantic minibatch identities')
        if any(g.ndim != 2 or g.shape[0] == 0 or g.shape[1] != self.width for g in groups):
            raise ValueError('Semantic features must be per-image [tokens,width]')
        if self.mode=='box_coverage':
            if any(g.shape!=(196,self.width) for g in groups):
                raise ValueError('Spatial semantic features require exactly 196 block-major tokens')
            b = len(groups)
            features = (torch.stack(groups).float().reshape(b,7,7,2,2,self.width)
                        .permute(0,1,3,2,4,5).reshape(b,196,self.width))
        else:
            features = torch.stack([g.float().mean(0) for g in groups])
        logits = self.head(F.layer_norm(features,(self.width,))).float()
        if self.mode=='box_coverage':
            labels = torch.stack([box_coverage_targets(self.boxes[self.index[key]],self.column)
                                  for key in sample_ids]).to(logits.device)
        else:
            labels = self.targets[[self.index[key] for key in sample_ids]].to(logits.device)
        return F.binary_cross_entropy_with_logits(logits,labels,pos_weight=self.positive_weight)
