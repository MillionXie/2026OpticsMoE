"""Training-only EMA and identity-checked, train-only teacher maps."""
from contextlib import contextmanager
import json
import torch


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


class TrainTeacherMaps:
    def __init__(self, settings, records):
        self.aligned_flip = getattr(settings, "augmentation_mode", "legacy") == "aligned_flip" and settings.augmentation_enabled
        self.batch_transform = None
        if settings.augmentation_enabled and not self.aligned_flip:
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

    def get(self, sample_ids, device):
        values = self.values[[self.index[k] for k in sample_ids]].float()
        if self.aligned_flip:
            if self.batch_transform is None or list(sample_ids) != self.batch_transform[0]:
                raise ValueError("Missing or mismatched teacher augmentation identity")
            flags = self.batch_transform[1]
            values[flags] = values[flags].flip(-1)
        return values.to(device)
