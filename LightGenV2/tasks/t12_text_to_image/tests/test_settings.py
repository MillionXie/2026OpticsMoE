from pathlib import Path

from LightGenV2.tasks.t12_text_to_image.compare import DEFAULT_PROMPTS
from LightGenV2.tasks.t12_text_to_image.settings import load_settings


TASK_DIR = Path(__file__).resolve().parents[1]


def test_formal_profiles_share_qwen_vae_and_shape_contract() -> None:
    lightgen = load_settings(TASK_DIR / "configs" / "lightgen_parallel.yaml")
    baseline = load_settings(TASK_DIR / "configs" / "qwen_vae_baseline.yaml")
    assert lightgen.qwen_model == baseline.qwen_model
    assert lightgen.vae_model == baseline.vae_model
    assert lightgen.image_size == baseline.image_size == 224
    assert lightgen.latent_size == baseline.latent_size == 28
    assert lightgen.token_grid == baseline.token_grid == 14
    assert lightgen.variant == "lightgen_parallel"
    assert baseline.variant == "qwen_vae_baseline"
    assert lightgen.optical_backend == "audited_dc20"
    assert baseline.optical_backend == "none"


def test_matched_prompts_follow_the_frozen_abo_categories() -> None:
    assert len(DEFAULT_PROMPTS) == 4
    assert all(any(category in prompt for category in ("shoe", "chair", "lamp", "table")) for prompt in DEFAULT_PROMPTS)
