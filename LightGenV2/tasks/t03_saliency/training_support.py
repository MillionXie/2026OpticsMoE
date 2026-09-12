"""Training-only EMA and identity-checked, train-only teacher maps."""
from contextlib import contextmanager
import json
import torch


def use_spawn_workers(loader):
    """Configure an unstarted T03 loader without inheriting parent CUDA state.

    Preserve dataset, sampler/generator, batching and persistent-worker policy.
    This is not a guarantee of cleanup after an arbitrary external signal.
    """
    if loader.num_workers:
        if getattr(loader, '_iterator', None) is not None:
            raise RuntimeError('Configure spawn before starting the data loader')
        loader.multiprocessing_context = 'spawn'
    return loader


def supervision_for_epoch(settings, epoch):
    """Train-only curriculum; never mutate the evaluation/base GT weights.

    Teacher-only means task supervision only: optical/router regularizers stay
    active. The same training images and cached teacher maps are used throughout.
    """
    from copy import copy
    if epoch < 1:
        raise ValueError("Supervision epochs are one-based")
    count = getattr(settings, "teacher_only_epochs", 0)
    active = count > 0 and epoch <= count
    current = settings
    if active:
        if not 0 < settings.map_kd_weight < float("inf"):
            raise ValueError("Teacher-only stage requires an active finite KD weight")
        current = copy(settings)
        for name in ("kl_weight", "cc_weight", "sim_weight", "nss_weight"):
            setattr(current, name, 0.0)
    return current, {
        "supervision_stage": "teacher_only" if active else "ground_truth_plus_teacher",
        **{f"effective_{name}": getattr(current, name)
           for name in ("kl_weight", "cc_weight", "sim_weight", "nss_weight")},
    }


class ModelEMA:
    def __init__(self, model, decay):
        self.modules = {"core": model.core, "head": model.head}
        self.decay = decay
        self.shadow = {k: {n: v.detach().clone() for n, v in m.state_dict().items()}
                       for k, m in self.modules.items()}

    @torch.no_grad()
    def update(self, *_):
        for k, module in self.modules.items():
            for name, value in module.state_dict().items():
                target = self.shadow[k][name]
                if target.is_floating_point():
                    target.lerp_(value.detach(), 1-self.decay)
                else:
                    target.copy_(value)

    @contextmanager
    def applied(self):
        live = {k: {n: v.detach().clone() for n, v in m.state_dict().items()}
                for k, m in self.modules.items()}
        try:
            for k, m in self.modules.items():
                m.load_state_dict(self.shadow[k], strict=True)
            yield
        finally:
            for k, m in self.modules.items():
                m.load_state_dict(live[k], strict=True)


def distillation_weight(initial, end_epoch, epoch, final=0.0):
    return final + (initial-final) * max(0., 1-(epoch-1)/max(1, end_epoch-1))


class AlignedFlipLoader:
    """Train-only mirror equivariance, applied after loading unaugmented data.

    Runs in the main process, so teacher transform identity cannot drift due to
    DataLoader prefetch. Teacher logits are mirrored, not re-inferred: this is
    an equivariance training target, not a claim of exact Qwen equivariance.
    """
    def __init__(self, loader, probability, seed, teacher=None):
        self.loader, self.probability, self.teacher = loader, probability, teacher
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        from PIL import Image
        for original in self.loader:
            batch = dict(original)
            ids = batch["sample_ids"]
            if any(not key.startswith("train/") for key in ids):
                raise ValueError("AlignedFlipLoader is restricted to training identities")
            flags = torch.rand(len(ids), generator=self.generator) < self.probability
            batch["images"] = [image.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if flag else image
                               for image, flag in zip(batch["images"], flags.tolist())]
            for name in ("density", "fixation"):
                value = batch[name].clone()
                value[flags] = value[flags].flip(-1)
                batch[name] = value
            if self.teacher is not None:
                self.teacher.batch_transform = (list(ids), flags.clone())
            yield batch


def spatial_correlation_distillation(student_logits, teacher_logits):
    """Per-image density correlation, inspired by DIST (NeurIPS22), not full DIST.

    No cross-image relation: pixel positions are not classification categories.
    Uniform teachers contain no spatial preference and contribute zero loss.
    Unit-mean scaling avoids near-zero norms of 224x224 probability densities.
    """
    from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency.objectives import density_from_logits
    if student_logits.shape != teacher_logits.shape:
        raise ValueError('Student and teacher saliency map shapes must match')
    student = density_from_logits(student_logits).flatten(1)
    teacher = density_from_logits(teacher_logits.detach()).flatten(1)
    size = student.shape[1]
    x = (student - student[:,:1]) * size
    y = (teacher - teacher[:,:1]) * size
    x = x - x.mean(1,keepdim=True)
    y = y - y.mean(1,keepdim=True)
    valid = torch.linalg.vector_norm(y,dim=1) > 1e-6
    cc = torch.nn.functional.cosine_similarity(x,y,dim=1,eps=1e-6)
    return ((1-cc.clamp(-1,1))*valid).sum()/valid.sum().clamp_min(1)


def task_saliency_loss(logits, target, fixation, settings, *, teacher_logits=None):
    """Opt-in train-only loss; historical KL path returns the original function."""
    from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency import objectives
    mode = getattr(settings,'distillation_loss','kl')
    if mode == 'kl':
        return objectives.saliency_loss(logits,target,fixation,settings,teacher_logits=teacher_logits)
    if mode != 'spatial_cc' or settings.map_kd_temperature != 1:
        raise ValueError('Only temperature1 spatial_cc or historical KL is supported')
    base, pieces = objectives.saliency_loss(logits,target,fixation,settings,teacher_logits=None)
    kd = logits.new_zeros(()) if teacher_logits is None else spatial_correlation_distillation(logits,teacher_logits)
    reliability = getattr(settings, 'teacher_reliability', {})
    if reliability and teacher_logits is not None:
        from .teacher_reliability import loss as reliable_teacher_loss
        kd, diagnostics = reliable_teacher_loss(logits, teacher_logits, target, reliability)
        pieces.update(diagnostics)
    reference = logits.new_zeros(())
    if teacher_logits is not None:
        with torch.no_grad():
            reference = objectives.kl_divergence(objectives.density_from_logits(logits),
                                                objectives.density_from_logits(teacher_logits))
    pieces.update(map_kd=kd, map_kd_kl_reference=reference)
    total = base + float(settings.map_kd_weight)*kd
    pyramid = getattr(settings, 'pyramid_cc', {})
    if pyramid:
        from .pyramid_supervision import loss as pyramid_loss
        extra, diagnostics = pyramid_loss(logits, target, pyramid['sizes'])
        total = total + pyramid['weight'] * extra
        pieces.update(diagnostics)
        pieces['pyramid_cc_loss'] = extra.detach()
    options = getattr(settings, 'hard_example_cc', {})
    if options:
        from .hard_example_cc import loss as hard_cc_loss
        extra, extra_pieces = hard_cc_loss(logits, target, options)
        total = total + options['weight']*extra
        pieces.update(extra_pieces)
    return total, pieces


class TrainTeacherMaps:
    def __init__(self, settings, records):
        self.aligned_flip = getattr(settings, "augmentation_mode", "legacy") == "aligned_flip" and settings.augmentation_enabled
        self.aligned_weak = getattr(settings, "augmentation_mode", "legacy") == "aligned_weak" and settings.augmentation_enabled
        self.batch_transform = None
        self.batch_augmented_logits = None
        if settings.augmentation_enabled and not (self.aligned_flip or self.aligned_weak):
            raise ValueError("Cached KD requires unaugmented, exactly aligned images")
        payload = torch.load(settings.distillation_cache, map_location="cpu", weights_only=False)
        ids = [r.sample_id for r in records]
        if payload["sample_ids"] != ids or any(not k.startswith("train/") for k in ids):
            raise ValueError("Teacher cache must contain exactly the ordered training identities")
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate teacher identities")
        self.manifest = payload["manifest"]
        if self.manifest["checkpoint_sha256"] != settings.distillation_teacher_sha256:
            raise ValueError("Teacher checkpoint SHA mismatch")
        if self.manifest["image_size"] != settings.image_size or self.manifest["augmentation"]:
            raise ValueError("Teacher preprocessing mismatch")
        self.values = payload["logits"]
        if tuple(self.values.shape) != (len(ids), 1, settings.image_size, settings.image_size):
            raise ValueError("Teacher cache shape mismatch")
        if not torch.isfinite(self.values).all():
            raise ValueError("Nonfinite teacher cache")
        self.index = {k: i for i, k in enumerate(ids)}

    def get_raw(self, sample_ids):
        return self.values[[self.index[k] for k in sample_ids]].float()

    def get(self, sample_ids, device):
        if self.aligned_weak or getattr(self, 'fixed_crop', False):
            if self.batch_augmented_logits is None or list(sample_ids) != self.batch_augmented_logits[0]:
                raise ValueError("Missing or mismatched weak-augmentation teacher identity")
            return self.batch_augmented_logits[1].to(device)
        values = self.get_raw(sample_ids)
        if self.aligned_flip:
            if self.batch_transform is None or list(sample_ids) != self.batch_transform[0]:
                raise ValueError("Missing or mismatched teacher augmentation identity")
            flags = self.batch_transform[1]
            values[flags] = values[flags].flip(-1)
        return values.to(device)


def warp_density(value, box, flip):
    """Crop/rescale a probability map, NOT its logits; preserve unit mass."""
    import torch.nn.functional as F
    left, top, right, bottom = box
    result = F.interpolate(value[..., top:bottom, left:right], size=value.shape[-2:],
                           mode="bilinear", align_corners=False).clamp_min(0)
    if flip:
        result = result.flip(-1)
    mass = result.sum(dim=(-2, -1), keepdim=True)
    if not torch.isfinite(result).all() or (mass <= 0).any():
        raise ValueError("Invalid/empty augmented density")
    return result / mass


class AlignedWeakLoader:
    """Training-only weak view consistency, with one transform per identity.

    Cached teacher density is geometrically transformed, not re-inferred on
    the augmented image. This is an approximate equivariance target; brightness
    and contrast invariance are also training assumptions, not measured facts.
    No transform or teacher is used in evaluation/inference.
    """
    def __init__(self, loader, settings, teacher=None):
        import random
        self.loader, self.settings, self.teacher = loader, settings, teacher
        self.rng = random.Random(settings.random_seed + 1703)
        self.apply_probability = getattr(settings, "augmentation_apply_probability", 1.0)
        if not 0 <= self.apply_probability <= 1:
            raise ValueError("Invalid weak-augmentation probability")
        # Separate RNG: the default p=1 consumes exactly the legacy view stream.
        self.apply_rng = random.Random(settings.random_seed + 4703)
        self.epoch_images = self.epoch_augmented_images = 0
        self.enabled = True

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        from PIL import Image, ImageEnhance
        import torch.nn.functional as F
        self.epoch_images = self.epoch_augmented_images = 0
        for original in self.loader:
            batch = dict(original)
            ids = list(batch["sample_ids"])
            if any(not key.startswith("train/") for key in ids) or len(set(ids)) != len(ids):
                raise ValueError("AlignedWeakLoader requires unique training identities")
            self.epoch_images += len(ids)
            raw_teacher = self.teacher.get_raw(ids) if self.teacher is not None else None
            if not self.enabled:
                if self.teacher is not None:
                    self.teacher.batch_augmented_logits = (ids, raw_teacher)
                yield batch
                continue
            images, densities, fixations, targets = [], [], [], []
            h, w = batch["density"].shape[-2:]
            for i, image in enumerate(batch["images"]):
                if image.size != (w, h):
                    raise ValueError("Augmentation image/target geometry mismatch")
                apply = self.apply_probability == 1 or (
                    self.apply_probability > 0 and self.apply_rng.random() < self.apply_probability)
                if not apply:
                    images.append(image)
                    densities.append(batch["density"][i:i+1])
                    fixations.append(batch["fixation"][i:i+1])
                    if raw_teacher is not None:
                        targets.append(raw_teacher[i:i+1])
                    continue
                self.epoch_augmented_images += 1
                scale = self.rng.uniform(self.settings.crop_scale_min, 1.0)
                ch, cw = max(1, round(h * scale)), max(1, round(w * scale))
                left, top = self.rng.randrange(w-cw+1), self.rng.randrange(h-ch+1)
                box = (left, top, left+cw, top+ch)
                flip = self.rng.random() < self.settings.horizontal_flip_probability
                density = warp_density(batch["density"][i:i+1], box, flip)
                # If the crop drops every fixation, use the whole original view;
                # never manufacture a fixation from the teacher or density peak.
                fix = batch["fixation"][i:i+1, :, top:top+ch, left:left+cw]
                if fix.sum() <= 0:
                    box = (0, 0, w, h)
                    density = warp_density(batch["density"][i:i+1], box, flip)
                    fix = batch["fixation"][i:i+1]
                fix = F.interpolate(fix.float(), size=(h,w), mode="nearest")
                if flip:
                    fix = fix.flip(-1)
                image = image.crop(box).resize((w,h), Image.Resampling.BILINEAR)
                if flip:
                    image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                image = ImageEnhance.Brightness(image).enhance(self.rng.uniform(
                    1-self.settings.brightness_jitter, 1+self.settings.brightness_jitter))
                image = ImageEnhance.Contrast(image).enhance(self.rng.uniform(
                    1-self.settings.contrast_jitter, 1+self.settings.contrast_jitter))
                images.append(image); densities.append(density); fixations.append(fix)
                if raw_teacher is not None:
                    prob = raw_teacher[i:i+1].flatten(1).softmax(-1).reshape(1,1,h,w)
                    targets.append(warp_density(prob, box, flip).clamp_min(1e-30).log())
            batch.update(images=images, density=torch.cat(densities), fixation=torch.cat(fixations))
            if self.teacher is not None:
                self.teacher.batch_augmented_logits = (ids, torch.cat(targets))
            yield batch
