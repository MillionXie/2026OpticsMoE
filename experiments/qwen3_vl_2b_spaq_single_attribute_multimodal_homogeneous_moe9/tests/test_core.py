from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn

from experiments.qwen3_vl_2b_spaq_single_attribute_multimodal_homogeneous_moe9.datasets import load_spaq
from experiments.qwen3_vl_2b_spaq_single_attribute_multimodal_homogeneous_moe9.features import pool_answer_hidden_state
from experiments.qwen3_vl_2b_spaq_single_attribute_multimodal_homogeneous_moe9.modeling import build_head
from experiments.qwen3_vl_2b_spaq_single_attribute_multimodal_homogeneous_moe9.optics.geometry import MoEGeometry
from experiments.qwen3_vl_2b_spaq_single_attribute_multimodal_homogeneous_moe9.optics.moe import (
    HomogeneousMoEOpticalCore, LanguageDeepStackHomogeneousMoE, VisionDeepStackHomogeneousMoE,
)
from experiments.qwen3_vl_2b_spaq_single_attribute_multimodal_homogeneous_moe9.optics.replacement import DeepStackMultimodalReplacement
from experiments.qwen3_vl_2b_spaq_single_attribute_multimodal_homogeneous_moe9.processor_cache import collate_processor_samples
from experiments.qwen3_vl_2b_spaq_single_attribute_multimodal_homogeneous_moe9.settings import load_settings


ROOT = Path("experiments/qwen3_vl_2b_spaq_single_attribute_multimodal_homogeneous_moe9")
OPTICAL = ROOT / "configs" / "spaq_mos_vision_language_optical_smoke.json"
ELECTRONIC = ROOT / "configs" / "spaq_mos_vision_electronic_language_smoke.json"


def test_configs_select_independent_language_modes() -> None:
    optical = load_settings(OPTICAL); electronic = load_settings(ELECTRONIC)
    assert optical.task_name == "MOS" and optical.student_language_mode == "optical_moe"
    assert electronic.student_language_mode == "electronic"
    assert optical.classification_prompt.endswith("Score:")
    assert optical.max_visual_tokens == optical.max_language_tokens == 120
    assert tuple(optical.vision_tap_stages) == (1, 3, 4)


def test_dataset_supports_mos_and_rgb(tmp_path: Path) -> None:
    root = tmp_path / "SPAQ"; images = root / "images"; images.mkdir(parents=True)
    rows = ["Image name,MOS,Brightness,Colorfulness,Contrast"]
    for index in range(10):
        name = f"i{index}.jpg"; Image.new("RGB", (8, 8), (index, 2, 3)).save(images / name)
        rows.append(f"{name},{50+index},40,30,20")
    (root / "scores.csv").write_text("\n".join(rows), encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"dataset": "spaq_single_attribute", "task_name": "MOS",
        "data_root": str(root), "download": False, "output_dir": str(tmp_path / "run"),
        "classification_prompt": "Rate quality. Score:"}), encoding="utf-8")
    bundle = load_spaq(load_settings(config)); image, target = bundle.train[0]
    assert image.mode == "RGB" and 0 <= target <= 1 and bundle.metadata["language_model_used"] is True


def _encoder(hidden_size: int = 8, max_tokens: int = 120) -> HomogeneousMoEOpticalCore:
    module = HomogeneousMoEOpticalCore.__new__(HomogeneousMoEOpticalCore); nn.Module.__init__(module)
    module.hidden_size = hidden_size; module.max_tokens = max_tokens; module.geometry = MoEGeometry()
    module.input_adapter = nn.Linear(hidden_size, 120); module.input_norm = nn.LayerNorm(120); module.nonnegative = nn.Softplus()
    return module


def test_strict_token_row_mapping_is_nonnegative_and_zero_padded() -> None:
    encoder = _encoder(); field = encoder.encode_groups([torch.randn(60, 8)])
    assert field.shape == (1, 120, 120); assert torch.all(field >= 0); assert torch.count_nonzero(field[:, 60:]) == 0
    with pytest.raises(RuntimeError, match="visual token count 121"):
        encoder.encode_groups([torch.randn(121, 8)])


def test_language_overflow_is_explicit() -> None:
    language = LanguageDeepStackHomogeneousMoE.__new__(LanguageDeepStackHomogeneousMoE); nn.Module.__init__(language)
    language.core = SimpleNamespace(max_tokens=120)
    with pytest.raises(RuntimeError, match="language sequence length 121"):
        language.set_attention_mask(torch.ones(1, 121))


def test_cached_multimodal_batch_padding_and_pixel_concatenation() -> None:
    rows = [{"input_ids": torch.tensor([1, 2]), "sequence_length": 2,
             "pixel_values": torch.ones(3, 4), "image_grid_thw": torch.tensor([1, 1, 3])},
            {"input_ids": torch.tensor([3, 4, 5]), "sequence_length": 3,
             "pixel_values": torch.ones(2, 4), "image_grid_thw": torch.tensor([1, 1, 2])}]
    batch = collate_processor_samples(rows, {"padding_side": "left", "pad_token_id": 0})
    assert batch["input_ids"].tolist() == [[0, 1, 2], [3, 4, 5]]
    assert batch["attention_mask"].tolist() == [[0, 1, 1], [1, 1, 1]]
    assert batch["pixel_values"].shape == (5, 4)


def test_answer_position_uses_last_valid_token() -> None:
    hidden = torch.arange(2 * 4 * 3).reshape(2, 4, 3).float(); mask = torch.tensor([[0, 1, 1, 0], [1, 1, 1, 1]])
    answer, positions = pool_answer_hidden_state(hidden, mask)
    assert positions.tolist() == [2, 3]; assert torch.equal(answer[0], hidden[0, 2])


class Block(nn.Module):
    def forward(self, hidden_states, **kwargs): return hidden_states


class FakeSurrogate(nn.Module):
    def __init__(self, stages=5):
        super().__init__(); self.core = SimpleNamespace(expert_layers=[None] * stages); self.weight = nn.Parameter(torch.ones(()))
    def set_attention_mask(self, mask): self.mask = mask


def test_replacement_maps_native_deepstack_taps_and_language_modes() -> None:
    visual = SimpleNamespace(blocks=nn.ModuleList([Block() for _ in range(24)]), deepstack_visual_indexes=[5, 11, 17])
    language = SimpleNamespace(layers=nn.ModuleList([Block() for _ in range(28)]), norm=nn.Identity())
    core = SimpleNamespace(visual=visual, language_model=language)
    model = nn.Module(); model.model = core
    vision = FakeSurrogate(); language_surrogate = FakeSurrogate()
    replacement = DeepStackMultimodalReplacement(model, vision, language_surrogate, "optical_moe")
    replacement.use_student()
    assert replacement.vision_blocks[0].__class__.__name__ == "VisionStartBlock"
    assert [replacement.vision_blocks[i].slot for i in (5, 11, 17, 23)] == [0, 1, 2, 3]
    assert [replacement.language_layers[i].stage for i in range(5)] == list(range(5))
    replacement.language_mode = "electronic"; replacement.use_student()
    assert replacement.language_layers[0] is replacement.original_language[0]
    replacement.close()


def test_small_text_regression_head_backward() -> None:
    settings = load_settings(OPTICAL); head = build_head(settings, 2048)
    prediction = head(torch.randn(4, 2048)); assert prediction.shape == (4,)
    torch.nn.functional.smooth_l1_loss(prediction, torch.rand(4), beta=0.1).backward()
    assert all(parameter.grad is not None for parameter in head.parameters())
