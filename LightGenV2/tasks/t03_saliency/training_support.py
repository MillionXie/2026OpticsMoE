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


def distillation_weight(initial, end_epoch, epoch):
    return initial * max(0., 1-(epoch-1)/max(1, end_epoch-1))


class TrainTeacherMaps:
    def __init__(self, settings, records):
        if settings.augmentation_enabled:
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
        return self.values[[self.index[k] for k in sample_ids]].float().to(device)
