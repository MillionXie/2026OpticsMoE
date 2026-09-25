"""Read-only count of deployed checkpoint components and unused packaged state."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _count(state):
    return sum(tensor.numel() for tensor in state.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--small-checkpoint", type=Path, required=True)
    parser.add_argument("--large-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    small = torch.load(args.small_checkpoint, map_location="cpu", weights_only=False)
    large = torch.load(args.large_checkpoint, map_location="cpu", weights_only=False)
    s = small["model"]
    def prefix_count(state, prefix):
        return sum(value.numel() for key, value in state.items() if key.startswith(prefix))
    small_components = {
        "qwen_mini_text": prefix_count(s, "text."),
        "rgb_stem": prefix_count(s, "stem."),
        "rgb_down": sum(prefix_count(s, f"down{i}.") for i in range(1, 4)),
        "parallel_bottleneck_electronic": prefix_count(s, "bottleneck.electronic."),
        "parallel_bottleneck_optical_and_fusion": (
            prefix_count(s, "bottleneck.") - prefix_count(s, "bottleneck.electronic.")),
        "optical_phase_masks_only": sum(value.numel() for key, value in s.items()
                                        if key.startswith("bottleneck.optical.") and key.endswith("_phase")),
        "rgb_up": sum(prefix_count(s, f"up{i}.") for i in range(1, 4)),
        "rgb_delta_head": prefix_count(s, "to_delta."),
        "source_gate": prefix_count(s, "source_gate."),
    }
    unet = large["unet"]
    frontend = large["text_frontend"]
    router_key = next((key for key in ("attribute_router", "design_router", "region_router")
                       if key in large), None)
    large_components = {
        "qwen_mini_text": _count(frontend["text"]),
        "qwen_mini_to_2048_bridge": _count(frontend["bridge"]),
        "unet_all": _count(unet),
        "unet_mid_optical_block": prefix_count(unet, "mid_block."),
        "unet_mid_phase_masks_only": sum(value.numel() for key, value in unet.items()
                                           if key.startswith("mid_block.optical.") and key.endswith("_phase")),
        "unet_other_electronic": _count(unet) - prefix_count(unet, "mid_block."),
        "condition_adapter_all_state_tensors": _count(large["adapter"]),
        "condition_adapter_trainable_tensors": prefix_count(large["adapter"], "predictor."),
        "condition_adapter_fixed_buffers": (_count(large["adapter"]) -
                                            prefix_count(large["adapter"], "predictor.")),
        "packaged_router": _count(large[router_key]) if router_key else 0,
        "packaged_router_key": router_key,
        "vae_encoder_including_quant": 34_163_664,
        "vae_decoder_including_post_quant": 49_490_199,
    }
    result = {"small": {"reported_counted": small["counted_parameters"],
                        "checkpoint_tensor_count": _count(s), "components": small_components,
                        "checkpoint_keys": list(small.keys())},
              "large": {"reported_counted": large["counted_parameters"],
                        "components": large_components,
                        "checkpoint_keys": list(large.keys()),
                        "text_frontend_kind": frontend.get("kind"),
                        "training_noise_scale": large["training_config"].get("noise_scale"),
                        "training_residual_scale": large["training_config"].get("residual_scale")}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
