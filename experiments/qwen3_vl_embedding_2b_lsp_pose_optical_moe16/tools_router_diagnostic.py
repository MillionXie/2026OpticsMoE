"""Router gradient diagnostic for the LSP pose optical MoE16 student.

Measures, over a few training batches, the gradient norm of the router gate
versus the optical phase planes / head, using the previously trained student
checkpoint.  Purpose: verify whether the router receives meaningful task
gradient, or whether it is starved (hypothesis behind the uniform routing).

Run from the repository root:
    python experiments/qwen3_vl_embedding_2b_lsp_pose_optical_moe16/tools_router_diagnostic.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.datasets import prepare_lsp
from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.losses import (
    masked_coordinate_loss,
    masked_heatmap_mse,
)
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
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optics.physical import (
    phase_dc_loss,
)


def _grad_norm(tensor: torch.Tensor) -> float:
    if tensor.grad is None:
        return 0.0
    return float(tensor.grad.detach().float().norm())


def _grad_list_norm(grads: list, params: list) -> float:
    total = 0.0
    for g, p in zip(grads, params):
        if g is None:
            continue
        total += float(g.detach().float().norm() ** 2)
    return total ** 0.5


def main() -> int:
    device = torch.device("cuda:0")
    config = REPO / "experiments/qwen3_vl_embedding_2b_lsp_pose_optical_moe16/configs/lsp_pose.yaml"
    settings = load_settings(config)
    settings.device = "cuda:0"
    seed_everything(settings.random_seed)

    print(f"[diag] dataset root = {settings.data_root}", flush=True)
    bundle = prepare_lsp(settings, persist=False)
    train_loader, _ = build_loaders(
        bundle, settings, training=True, train_batch_size=4,
    )

    print(f"[diag] loading {settings.model_id} on {device}", flush=True)
    loaded = load_vision_backbone(settings, device)
    model = build_student(loaded, settings)

    # Load the previously trained student checkpoint.
    ckpt_path = (
        settings.output_dir / "checkpoints" / "student_best_train_loss.pt"
    )
    if ckpt_path.is_file():
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.core.load_state_dict(payload["core"])
        model.head.load_state_dict(payload["head"])
        print(f"[diag] loaded checkpoint epoch={payload.get('epoch')}", flush=True)
    else:
        print("[diag] no checkpoint; running with freshly initialized weights", flush=True)

    model.train()

    grad_groups = {
        "router.gate": list(model.core.router.parameters()),
        "input_adapter": [model.core.input_adapter.weight, model.core.input_adapter.bias,
                          model.core.input_norm.weight, model.core.input_norm.bias],
        "expert_phases": list(model.core.expert_layers.parameters()),
        "global_phase": list(model.core.global_phase.parameters()),
        "pose_head": list(model.head.parameters()),
    }

    totals = {name: 0.0 for name in grad_groups}
    counts = {name: sum(p.numel() for p in params) for name, params in grad_groups.items()}
    # Decomposed router gradients: task vs balance vs importance vs phase_dc.
    router_params = list(model.core.router.parameters())
    router_grad = {
        "task": 0.0, "balance": 0.0, "importance": 0.0, "phase_dc": 0.0, "full": 0.0,
    }
    n_batches = 0
    entropy_sum = 0.0
    balance_sum = 0.0
    for batch_index, batch in enumerate(train_loader, 1):
        if batch_index > 4:
            break
        n_batches += 1
        model.zero_grad(set_to_none=True)
        inputs = preprocess_vision(loaded.processor, batch["images"], device)
        target_heatmaps = batch["heatmaps"].to(device, non_blocking=True)
        keypoints = batch["keypoints"].to(device, non_blocking=True)
        visible = batch["visible"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=True,
        ):
            predictions = model(**inputs)[0]
            heatmap_loss = masked_heatmap_mse(predictions, target_heatmaps, visible)
            coordinate_loss = masked_coordinate_loss(
                predictions, keypoints, visible, settings.image_size,
            )
            balance, importance = model.router_losses()
            dc = phase_dc_loss(model) if settings.phase_dc_weight > 0.0 else predictions.new_zeros(())
            task_loss = (
                settings.heatmap_loss_weight * heatmap_loss
                + settings.coordinate_loss_weight * coordinate_loss
            )
            loss = (
                task_loss
                + settings.router_balance_weight * balance
                + settings.router_importance_weight * importance
                + settings.phase_dc_weight * dc
            )
        # Reuse one graph to decompose the router gradient by loss term.
        g_task = torch.autograd.grad(
            task_loss, router_params, retain_graph=True, allow_unused=True,
        )
        g_bal = torch.autograd.grad(
            settings.router_balance_weight * balance, router_params,
            retain_graph=True, allow_unused=True,
        )
        g_imp = torch.autograd.grad(
            settings.router_importance_weight * importance, router_params,
            retain_graph=True, allow_unused=True,
        )
        g_dc = torch.autograd.grad(
            settings.phase_dc_weight * dc, router_params,
            retain_graph=True, allow_unused=True,
        )
        g_full = torch.autograd.grad(loss, router_params, allow_unused=True)
        router_grad["task"] += _grad_list_norm(g_task, router_params)
        router_grad["balance"] += _grad_list_norm(g_bal, router_params)
        router_grad["importance"] += _grad_list_norm(g_imp, router_params)
        router_grad["phase_dc"] += _grad_list_norm(g_dc, router_params)
        router_grad["full"] += _grad_list_norm(g_full, router_params)

        for name, params in grad_groups.items():
            norm = sum(_grad_norm(p) ** 2 for p in params) ** 0.5
            totals[name] += norm
            if name == "router.gate":
                w = model.core.router.router.gate.weight
                print(
                    f"[diag] batch {batch_index} router_gate_grad_norm={norm:.4f} "
                    f"gate_w_std={float(w.detach().std()):.4f} "
                    f"entropy={float(model.core.last_routing['normalized_entropy']):.4f} "
                    f"balance={float(balance.detach()):.4f}",
                    flush=True,
                )
        entropy_sum += float(model.core.last_routing["normalized_entropy"])
        balance_sum += float(balance.detach())

    print("\n[diag] ==== average per-batch gradient norms (total over params) ====", flush=True)
    for name in grad_groups:
        per_batch = totals[name] / n_batches
        per_param = per_batch / max(counts[name], 1)
        print(
            f"  {name:16s} params={counts[name]:>10,}  "
            f"grad_norm/batch={per_batch:10.4f}  grad_norm/param={per_param:.6f}",
            flush=True,
        )
    print("\n[diag] ==== router gradient decomposition (per-batch, L2 over gate params) ====", flush=True)
    for name in ("task", "balance", "importance", "phase_dc", "full"):
        print(f"  router_grad[{name:10s}] = {router_grad[name] / n_batches:.4f}", flush=True)
    print(f"[diag] avg router entropy = {entropy_sum / n_batches:.4f}", flush=True)
    print(f"[diag] avg router balance = {balance_sum / n_batches:.4f}", flush=True)

    model.restore_native()
    loaded.model.to("cpu")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
