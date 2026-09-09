from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from LightGenV2.tasks.t03_saliency.modeling import (
    architecture_label, configure_spatial_kernel, expand_spatial_checkpoint,
)
from LightGenV2.tasks.t03_saliency.settings import load_settings
from LightGenV2.tasks.t03_saliency.training import staged_epoch
from LightGenV2.tasks.t03_saliency.training_support import (
    AlignedFlipLoader, TrainTeacherMaps, distillation_weight,
)

TASK = Path(__file__).resolve().parents[1]


def test_center_padded_kernel_preserves_function_and_receives_rim_gradients():
    blocks = []
    for _ in range(2):
        block = torch.nn.Module()
        block.token_depthwise = torch.nn.Conv2d(192, 192, 3, padding=1, groups=192, bias=False)
        blocks.append(block)
    hybrid = torch.nn.Module()
    hybrid.blocks = torch.nn.ModuleList(blocks)
    x = torch.randn(2, 192, 14, 14)
    before = hybrid.blocks[0].token_depthwise(x).detach()
    count = sum(p.numel() for p in hybrid.parameters())
    configure_spatial_kernel(hybrid, 5)
    assert sum(p.numel() for p in hybrid.parameters()) - count == 6144
    after = hybrid.blocks[0].token_depthwise(x)
    torch.testing.assert_close(after, before, atol=1e-6, rtol=1e-5)
    after.square().mean().backward()
    assert hybrid.blocks[0].token_depthwise.weight.grad[:, :, 0, :].abs().sum() > 0


def test_warmstart_shape_transfer_is_strict():
    source = {f"hybrid.blocks.{i}.token_depthwise.weight": torch.randn(192, 1, 3, 3) for i in range(2)}
    source["other"] = torch.ones(3)
    target = {k: torch.zeros(192, 1, 5, 5) if k != "other" else v for k, v in source.items()}
    out = expand_spatial_checkpoint(source, target)
    for i in range(2):
        key = f"hybrid.blocks.{i}.token_depthwise.weight"
        torch.testing.assert_close(out[key][:, :, 1:4, 1:4], source[key])
        assert out[key][:, :, 0, :].abs().sum() == 0
    with pytest.raises(RuntimeError):
        expand_spatial_checkpoint(source, {**target, "other": torch.zeros(4)})


def test_flip_image_targets_and_teacher_are_identical_transform(tmp_path):
    path = tmp_path / "teacher.pt"
    logits = torch.arange(8.).reshape(2, 1, 2, 2)
    ids = ["train/1", "train/2"]
    torch.save({"sample_ids": ids, "logits": logits,
                "manifest": {"checkpoint_sha256": "abc", "image_size": 2, "augmentation": False}}, path)
    settings = SimpleNamespace(augmentation_enabled=True, augmentation_mode="aligned_flip",
        distillation_cache=path, distillation_teacher_sha256="abc", image_size=2)
    teacher = TrainTeacherMaps(settings, [SimpleNamespace(sample_id=k) for k in ids])
    with pytest.raises(ValueError):
        teacher.get(ids, "cpu")
    image = Image.fromarray(np.array([[0, 100], [50, 200]], dtype=np.uint8)).convert("RGB")
    batch = {"sample_ids": ids, "images": [image, image], "density": logits.clone(), "fixation": logits.clone()}
    flipped = next(iter(AlignedFlipLoader([batch], 1.0, 42, teacher)))
    torch.testing.assert_close(flipped["density"], logits.flip(-1))
    torch.testing.assert_close(flipped["fixation"], logits.flip(-1))
    torch.testing.assert_close(teacher.get(ids, "cpu"), logits.flip(-1))
    assert np.array_equal(np.asarray(flipped["images"][0]), np.asarray(image)[:, ::-1])
    torch.testing.assert_close(batch["density"], logits)
    torch.testing.assert_close(teacher.values, logits)
    with pytest.raises(ValueError):
        teacher.get(ids[::-1], "cpu")
    batch["sample_ids"] = ["validation/1", "validation/2"]
    with pytest.raises(ValueError):
        next(iter(AlignedFlipLoader([batch], .5, 42)))


def test_constant_kd_and_true_electronic_freeze():
    assert [distillation_weight(.2, 30, e, .2) for e in (1, 10, 30, 100)] == [.2]*4
    settings = load_settings(TASK / "configs/moe_alpha40_rfstage_staged.yaml")
    parameter = torch.nn.Parameter(torch.ones(1))
    phase = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.AdamW([{"name": "electronic", "params": [parameter], "lr": .01},
                            {"name": "feature_phase", "params": [phase], "lr": .01}])
    staged_epoch(opt, settings, 1)
    assert not parameter.requires_grad and phase.requires_grad
    phase.sum().backward()
    opt.step()
    assert torch.equal(parameter.detach(), torch.ones(1))
    assert parameter not in opt.state  # No hidden momentum accumulation during freeze.
    staged_epoch(opt, settings, 6)
    assert parameter.requires_grad and opt.param_groups[0]["lr"] == .01


def test_four_configs_share_checkpoint_and_optical_contract():
    configs = [load_settings(TASK / f"configs/moe_alpha40_rfstage_{name}.yaml")
               for name in ("control", "flip", "kernel5", "staged")]
    for s in configs:
        assert s.initialization_checkpoint == configs[0].initialization_checkpoint
        assert s.initialization_checkpoint_sha256 == configs[0].initialization_checkpoint_sha256
        assert s.student_epochs == 30 and s.test_interval_epochs == 2
        assert s.top_k == 2 and s.router_backend == "optical" and s.fusion_alpha_min == .4
        assert s.language_optical_phase_zero_order_intensity_min == .2
        assert s.language_optical_phase_zero_order_intensity_max == .3
        assert s.active_size == 478 and s.expert_size == 224
    assert [s.augmentation_enabled for s in configs] == [False, True, False, False]
    assert [s.staged_warmup_epochs for s in configs] == [0, 0, 0, 5]
    assert architecture_label(configs[2]) == architecture_label(configs[0]) + "_ek5"
