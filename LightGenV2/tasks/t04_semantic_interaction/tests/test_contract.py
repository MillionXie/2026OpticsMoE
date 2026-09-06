from pathlib import Path

import torch

from LightGenV2.tasks.t04_semantic_interaction.settings import load_settings
from LightGenV2.tasks.t04_semantic_interaction.training import (
    _semantic_router_code_loss,
)


TASK = Path(__file__).resolve().parents[1]


def test_optical_router_profile_contract() -> None:
    settings = load_settings(
        TASK / "configs" / "moe_optical_router_scale_matched_dc20.yaml"
    )
    assert settings.lightgen_model_variant == "optical_router_scale_matched_moe"
    assert settings.train_samples == 5000
    assert settings.test_samples == 1000
    assert settings.router_hard_load_balance_weight == 0.50
    assert settings.router_importance_weight == 0.10
    assert settings.router_semantic_code_weight == 0.05
    assert settings.phase_dc_weight == 0.005


def test_d2nn_profile_has_no_router_penalty() -> None:
    settings = load_settings(
        TASK / "configs" / "d2nn_active_expert_matched_dc20.yaml"
    )
    assert settings.lightgen_model_variant == "d2nn_active_expert_matched"
    assert settings.router_balance_weight == 0.0
    assert settings.router_importance_weight == 0.0
    assert settings.router_hard_load_balance_weight == 0.0
    assert settings.router_semantic_code_weight == 0.0


def test_no_validation_split() -> None:
    settings = load_settings(
        TASK / "configs" / "moe_optical_router_scale_matched_dc20.yaml"
    )
    assert settings.train_manifest.name == "train.jsonl"
    assert settings.test_manifest.name == "test.jsonl"


def test_semantic_router_code_uses_physical_detector_energy() -> None:
    class Core:
        last_routing = {
            "probabilities": torch.full((2, 4), 0.25),
            "detector_energy_fraction": torch.tensor(
                [[0.45, 0.45, 0.05, 0.05], [0.45, 0.05, 0.45, 0.05]],
                requires_grad=True,
            ),
        }

    class Branch:
        core = Core()

    class Language:
        optical_branch = Branch()

    class Model:
        router_backend = "optical"
        language_core = Language()

    loss = _semantic_router_code_loss(Model(), ["add", "replace"])
    expected = -torch.log(torch.tensor(0.45))
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert Core.last_routing["detector_energy_fraction"].grad is not None
    assert Core.last_routing["probabilities"].grad is None
