"""Removable training-only semantic classifier; never registered in the student."""
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class SemanticAuxiliary(nn.Module):
    def __init__(self, settings, dataset):
        super().__init__()
        contents = Path(settings.semantic_targets).read_bytes()
        if hashlib.sha256(contents).hexdigest() != settings.semantic_targets_sha256:
            raise ValueError('Semantic targets SHA mismatch')
        raw = json.loads(contents)
        if (raw.get('schema_version') != 1
                or raw.get('purpose') != 'training_only_auxiliary_object_presence_not_saliency_ground_truth'
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
        self.index = {key:i for i,key in enumerate(ids)}
        self.width = int(settings.electronic_width)
        # Keep the global training RNG untouched, including CUDA optical draws.
        with torch.random.fork_rng(devices=[]):
            generator = torch.Generator().manual_seed(settings.random_seed + 2281)
            self.head = nn.Linear(self.width,80)
            nn.init.normal_(self.head.weight,std=.001,generator=generator)
            with torch.no_grad():
                prior = (counts/len(ids)).clamp(.0001,.9999)
                self.head.bias.copy_(torch.logit(prior))
        positive_weight = ((len(ids)-counts)/counts.clamp_min(1)).sqrt().clamp(1,10)
        self.register_buffer('positive_weight',positive_weight,persistent=False)
        self.provenance = {
            'targets_sha256': settings.semantic_targets_sha256,
            'image_manifest_sha256': dataset.manifest_sha256,
            'annotation_sha256': raw['annotation_sha256'],
            'samples': len(ids), 'categories': categories,
            'positive_image_counts': counts.long().tolist(),
            'positive_weight': positive_weight.tolist(),
            'positive_weight_rule': 'sqrt(negative/positive), clipped to [1,10]',
            'training_only_parameters': sum(p.numel() for p in self.parameters()),
            'inference_parameters_added': 0,
            'additional_human_semantic_supervision': True,
            'feature_source': 'final fused student latent -> spatial mean -> non-affine LN -> Linear(192,80)',
            'checkpoint_state': 'training_only_semantic (live auxiliary head, not EMA inference weights)',
        }

    def forward(self, groups, sample_ids):
        if not sample_ids or len(set(sample_ids)) != len(sample_ids) or len(groups) != len(sample_ids):
            raise ValueError('Invalid semantic minibatch identities')
        if any(g.ndim != 2 or g.shape[0] == 0 or g.shape[1] != self.width for g in groups):
            raise ValueError('Semantic features must be per-image [tokens,width]')
        features = torch.stack([g.float().mean(0) for g in groups])
        logits = self.head(F.layer_norm(features,(self.width,))).float()
        labels = self.targets[[self.index[key] for key in sample_ids]].to(logits.device)
        return F.binary_cross_entropy_with_logits(logits,labels,pos_weight=self.positive_weight)
