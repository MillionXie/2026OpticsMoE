"""Post-training expert utilization diagnostic for the LSP pose optical MoE16.

Loads a trained student checkpoint and measures, over the test set:
  - per-expert selection frequency (load) and probability mass (importance)
  - dead / over-used experts
  - routing entropy distribution
  - per-expert phase statistics

Run from the repository root:
    CUDA_VISIBLE_DEVICES=0 python \
      experiments/qwen3_vl_embedding_2b_lsp_pose_optical_moe16/tools_expert_utilization.py \
      --config experiments/qwen3_vl_embedding_2b_lsp_pose_optical_moe16/configs/lsp_pose_opt.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.datasets import prepare_lsp
from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.modeling import (
    build_student,
    load_vision_backbone,
    preprocess_vision,
)
from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.settings import load_settings
from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.training import (
    build_loaders,
    seed_everything,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/student_best_train_loss.pt")
    parser.add_argument("--max-batches", type=int, default=40)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    settings = load_settings(args.config)
    settings.device = "cuda:0"
    seed_everything(settings.random_seed)

    bundle = prepare_lsp(settings, persist=False)
    _, test_loader = build_loaders(bundle, settings, training=False)

    print(f"[util] loading {settings.model_id} on {device}", flush=True)
    loaded = load_vision_backbone(settings, device)
    model = build_student(loaded, settings)

    ckpt_path = settings.output_dir / args.checkpoint
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.core.load_state_dict(payload["core"])
    model.head.load_state_dict(payload["head"])
    print(f"[util] loaded {ckpt_path} epoch={payload.get('epoch')}", flush=True)
    model.eval()

    num_experts = settings.num_experts
    selected_counts = torch.zeros(num_experts)
    prob_sum = torch.zeros(num_experts)
    weight_sum = torch.zeros(num_experts)
    entropy_values = []
    samples_seen = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(test_loader, 1):
            if batch_index > args.max_batches:
                break
            inputs = preprocess_vision(loaded.processor, batch["images"], device)
            model(**inputs)
            routing = model.core.last_routing
            selected = routing["selected_mask"].float().cpu()
            probs = routing["probabilities"].float().cpu()
            weights = routing["weights"].float().cpu()
            ent = routing["normalized_entropy"].float().cpu()
            selected_counts += selected.sum(0)
            prob_sum += probs.sum(0)
            weight_sum += weights.sum(0)
            entropy_values.append(ent.reshape(-1))
            samples_seen += selected.shape[0]

    load = selected_counts / max(samples_seen, 1)
    importance = prob_sum / max(samples_seen, 1)
    mean_weight = weight_sum / max(samples_seen, 1)
    entropy_all = torch.cat(entropy_values) if entropy_values else torch.zeros(0)

    print(f"\n[util] ==== expert utilization over {samples_seen} test samples ====", flush=True)
    print(f"  {'expert':>6} {'load(sel%)':>10} {'importance':>10} {'mean_weight':>12}", flush=True)
    for e in range(num_experts):
        flag = ""
        if load[e] == 0:
            flag = "  <-- DEAD"
        elif load[e] > 0.5:
            flag = "  <-- over-used"
        print(
            f"  {e:6d} {100*load[e]:9.2f}% {importance[e]:10.4f} {mean_weight[e]:12.4f}{flag}",
            flush=True,
        )
    print(f"  dead experts: {int((load == 0).sum())}/16", flush=True)
    print(f"  load std: {load.std():.4f} (0 = perfectly balanced)", flush=True)
    print(f"  normalized entropy mean={entropy_all.mean():.4f} "
          f"min={entropy_all.min():.4f} max={entropy_all.max():.4f}", flush=True)

    # Phase statistics.
    phases = []
    for i in range(num_experts):
        p = model.core.expert_layers[0].experts[i].raw_phase.detach().float()
        phases.append(p.std().item())
    print(f"  per-expert phase std: min={min(phases):.4f} max={max(phases):.4f} "
          f"mean={sum(phases)/len(phases):.4f}", flush=True)

    model.restore_native()
    loaded.model.to("cpu")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
