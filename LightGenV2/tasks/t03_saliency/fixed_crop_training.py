"""Matched actual-versus-proxy crop teacher targets; training only, no model edits."""
import math
from pathlib import Path
import random
import torch
import torch.nn.functional as F
from .fixed_crop_teacher import BOX, SIZE, VIEW, crop_image, check_pixels, validate_payload
from .training_support import warp_density


def configure(settings, raw, base):
    options = dict(raw or {})
    settings.fixed_crop_distillation = options
    if not options:
        return
    if set(options) != {'mode', 'cache', 'cache_sha256', 'apply_probability', 'end_epoch'}:
        raise ValueError('Fixed crop requires the exact audited option set')
    if options['mode'] not in {'actual', 'proxy'}:
        raise ValueError('Fixed crop mode must be actual or proxy')
    sha = options['cache_sha256']
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
        raise ValueError('Fixed crop requires a complete cache SHA256')
    p = options['apply_probability']
    if type(p) not in (int, float) or not math.isfinite(p) or not 0 < p <= 1:
        raise ValueError('Invalid fixed crop probability')
    if type(options['end_epoch']) is not int or not 1 <= options['end_epoch'] < settings.student_epochs:
        raise ValueError('Fixed crop requires a final original-image polishing stage')
    if (settings.augmentation_enabled or settings.feature_hint_initial_weight or settings.unlabeled_weight
            or settings.semantic_weight or settings.teacher_only_epochs or settings.first_stage_supervision
            or settings.feature_pretraining.get('enabled', False) or settings.relational_distillation
            or settings.masked_distillation or settings.reset_fusion_on_warmstart
            or settings.sam_rho <= 0 or not settings.initialization_checkpoint
            or settings.distillation_initial_weight <= 0 or settings.distillation_final_weight <= 0
            or settings.distillation_loss != 'spatial_cc' or settings.fusion_alpha_min < .4
            or settings.kl_weight <= 0 or settings.cc_weight <= 0 or settings.top_k != 2
            or settings.lightgen_model_variant != 'optical_router_scale_matched_moe'
            or settings.image_size != SIZE or settings.electronic_width != 192):
        raise ValueError('Fixed crop trial requires isolated original GT+KD/SAM optical MoE')
    if not isinstance(options['cache'], str) or not options['cache']:
        raise ValueError('Fixed crop cache path required')
    options['cache'] = str((Path(base) / options['cache']).resolve())


class FixedCropTargets:
    def __init__(self, settings, records):
        from .recheck_aligned import load_hashed_checkpoint
        from .modeling import sha256_file
        from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency.datasets import _annotation_path
        options = settings.fixed_crop_distillation
        self.payload, digest = load_hashed_checkpoint(options['cache'])
        if digest != options['cache_sha256']:
            raise ValueError('Fixed crop cache bytes SHA mismatch')
        validate_payload(self.payload, records, settings.distillation_teacher_sha256,
                         sha256_file(_annotation_path(settings.data_root, 'train')))
        self.index = {r.sample_id: i for i, r in enumerate(records)}
        self.provenance = dict(self.payload['manifest'], **options,
            student_inference_requires_teacher=False, inference_parameters_added=0,
            original_teacher_cache=str(settings.distillation_cache),
            original_teacher_cache_sha256=sha256_file(settings.distillation_cache),
            fixation_transform='nearest; fall back to original view if cropped fixation sum is zero',
            ground_truth_transform='crop, bilinear align_corners=False, renormalize density mass',
            teacher_target_used=('teacher(cropped RGB)' if options['mode']=='actual'
                                 else 'crop/resize original teacher probability then log; approximate target'))


class FixedCropLoader:
    """Same images/GT/random view choices for both arms; only KD targets differ.

    Two SAM forwards reuse the current target. No test transforms, fabricated
    fixations, color edits, pixel-dependent target choice, or GT-based selection.
    """
    def __init__(self, loader, settings, teacher, targets):
        if teacher is None or teacher.aligned_weak or teacher.aligned_flip:
            raise ValueError('Fixed crop requires an original-image teacher cache')
        if teacher.index != targets.index:
            raise ValueError('Original and crop teacher ordered identities differ')
        self.loader, self.teacher, self.targets = loader, teacher, targets
        self.options = settings.fixed_crop_distillation
        self.rng = random.Random(settings.random_seed + 6703)
        self.enabled = True
        self.epoch_images = self.epoch_augmented_images = self.epoch_fallback_images = 0
        teacher.fixed_crop = True

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        self.epoch_images = self.epoch_augmented_images = self.epoch_fallback_images = 0
        for original in self.loader:
            ids = list(original['sample_ids'])
            if (not ids or len(set(ids)) != len(ids) or any(s not in self.targets.index for s in ids)
                    or len(original['images']) != len(ids)
                    or original['density'].shape != (len(ids), 1, SIZE, SIZE)
                    or original['fixation'].shape != original['density'].shape):
                raise ValueError('Fixed crop training batch identity/geometry mismatch')
            self.epoch_images += len(ids)
            raw = self.teacher.get_raw(ids)
            images, densities, fixations, logits = [], [], [], []
            for i, (sid, image) in enumerate(zip(ids, original['images'])):
                index = self.targets.index[sid]
                crop = crop_image(image)
                check_pixels(self.targets.payload, index, image, crop)
                apply = self.enabled and self.rng.random() < self.options['apply_probability']
                density = original['density'][i:i+1]
                fixation = original['fixation'][i:i+1]
                target = raw[i:i+1]
                if apply:
                    left, top, right, bottom = BOX
                    crop_fix = fixation[..., top:bottom, left:right]
                    if crop_fix.sum() <= 0:
                        self.epoch_fallback_images += 1
                        apply = False
                if apply:
                    self.epoch_augmented_images += 1
                    image = crop
                    density = warp_density(density, BOX, False)
                    fixation = F.interpolate(crop_fix.float(), size=(SIZE, SIZE), mode='nearest')
                    if self.options['mode'] == 'actual':
                        target = self.targets.payload['logits'][index:index+1].float()
                    else:
                        probability = target.flatten(1).softmax(-1).reshape_as(target)
                        target = warp_density(probability, BOX, False).clamp_min(1e-30).log()
                images.append(image); densities.append(density); fixations.append(fixation); logits.append(target)
            batch = dict(original, images=images, density=torch.cat(densities), fixation=torch.cat(fixations))
            self.teacher.batch_augmented_logits = (ids, torch.cat(logits))
            try:
                yield batch
            finally:
                self.teacher.batch_augmented_logits = None
