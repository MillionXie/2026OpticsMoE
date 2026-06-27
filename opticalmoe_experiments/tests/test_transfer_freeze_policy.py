import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_switching.scripts.train_dataset_switching import build_model
from test_dataset_switching_model import tiny_config
from transfer_adaptation.scripts.transfer_utils import add_transfer_target_task, apply_transfer_freeze_policy


def test_transfer_freeze_policy_only_trains_target_prompt(tmp_path):
    tasks = ["mnist", "fashionmnist", "emnist_letters"]
    nums = {"mnist": 10, "fashionmnist": 10, "emnist_letters": 26}
    model = build_model(tiny_config("learnable_route_moe"), tasks, nums)
    add_transfer_target_task(
        model,
        "usps",
        10,
        init_from_source_prompt="mnist",
        target_head_config={"readout_type": "optical_only", "input_norm": "none", "norm_affine": False},
    )
    summary = apply_transfer_freeze_policy(
        model,
        "usps",
        {
            "train_target_prompt": True,
            "train_target_readout": False,
            "freeze_target_readout": True,
            "allow_extra_trainable_params": False,
        },
        tmp_path,
    )
    trainable = summary["trainable_parameter_names"]
    assert trainable == [
        "prompt_bank.prompts.usps.amplitude_logits",
        "prompt_bank.prompts.usps.phase_biases",
    ]
    assert summary["trainable_electronic_params"] == 0
    assert all(not p.requires_grad for p in model.expert_layers.parameters())
    assert all(not p.requires_grad for p in model.global_fc.parameters())
    assert (tmp_path / "parameter_freeze" / "trainable_parameter_names.txt").exists()

