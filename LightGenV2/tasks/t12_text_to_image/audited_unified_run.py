"""Audit, fine-tune and infer the versioned live-language DC20 editor.

Only frozen token embeddings may be cached. Live language/optics run per batch.
This entry point does not accept legacy pooled-text caches as model inputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .audited_unified import ARCHITECTURE, architecture_report, migrate, optical_diagnostics, reduce_condition_rank, prune_text_mlp, add_decoder_refinement


def load_model(args):
    if args.checkpoint:
        sealed = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if sealed.get("construction"):
            if sealed.get("architecture") != ARCHITECTURE:
                raise ValueError("Unsupported sealed architecture")
            from .sealed_editor import build_sealed
            model = build_sealed(sealed)
            report = architecture_report(model)
            if not report["within_limit"]:
                raise ValueError("Sealed checkpoint exceeds parameter budget")
            return model.to(args.device), report
    payload = torch.load(args.source, map_location="cpu", weights_only=False)
    model = migrate(payload, initial_unet=args.initial_unet, vae_checkpoint=args.vae_checkpoint,
                    adapter_checkpoint=args.adapter_checkpoint)
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if saved["architecture"] != ARCHITECTURE:
            raise ValueError("Not a compatible audited checkpoint")
        if saved.get("adapter_rank") is not None:
            reduce_condition_rank(model, saved["adapter_rank"])
        if saved.get("text_mlp_indices") is not None:
            prune_text_mlp(model, len(saved["text_mlp_indices"][0]), saved["text_mlp_indices"])
        if saved.get("decoder_refinement", False):
            add_decoder_refinement(model)
        model.load_state_dict(saved["model"], strict=True)
    if args.expand_decoder:
        add_decoder_refinement(model)
    if args.text_mlp_width is not None:
        if getattr(model, "text_mlp_indices", None) is not None:
            raise ValueError("Do not prune an already-pruned checkpoint twice")
        prune_text_mlp(model, args.text_mlp_width)
    if args.adapter_rank is not None:
        reduce_condition_rank(model, args.adapter_rank)
    report = architecture_report(model)
    from .sealed_editor import construction_metadata
    model.construction = construction_metadata(model, payload, args)
    if args.checkpoint:
        report.update({"loaded_checkpoint": str(args.checkpoint),
                       "loaded_checkpoint_step": saved.get("step", saved.get("report", {}).get("best_step")),
                       "status": "trained v2 checkpoint; quality must be checked on held-out data"})
    if not report["within_limit"]:
        raise ValueError(f"Budget exceeded, including fixed condition matrices: {report['parameters_plus_fixed_condition_values']}")
    return model.to(args.device), report


def noise_for(model, reference, generator=None):
    shape = reference.shape if model.kind == "small" else (len(reference), 4, 32, 32)
    return torch.randn(shape, device=reference.device, generator=generator)


def seal(model, report, args):
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"architecture": ARCHITECTURE, "construction": model.construction,
                "model": model.cpu().state_dict(), "text_mlp_indices": getattr(model,"text_mlp_indices",None),
                "decoder_refinement": getattr(model,"decoder_refinement",False), "report": report,
                "provenance_checkpoint": str(args.checkpoint)}, args.output)
    return report


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")


class DetailFeatures(torch.nn.Module):
    """Frozen ImageNet ResNet18 early features; training/validation only."""
    def __init__(self):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        net = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.layers = torch.nn.ModuleList([
            torch.nn.Sequential(net.conv1, net.bn1, net.relu),
            torch.nn.Sequential(net.maxpool, net.layer1), net.layer2])
        self.register_buffer("mean", torch.tensor([.485, .456, .406])[None,:,None,None])
        self.register_buffer("std", torch.tensor([.229, .224, .225])[None,:,None,None])
        self.eval().requires_grad_(False)

    def forward(self, image):
        x = ((image+1)/2-self.mean)/self.std
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return features


class DetailDiscriminator(torch.nn.Module):
    """Reference-conditional PatchGAN, never serialized into inference weights."""
    def __init__(self):
        super().__init__()
        from torch.nn.utils import spectral_norm
        layers = []
        channels = 6
        for width in (32, 64, 128, 128):
            layers.extend([spectral_norm(torch.nn.Conv2d(channels, width, 4, 2, 1)),
                           torch.nn.LeakyReLU(.2)])
            channels = width
        layers.append(spectral_norm(torch.nn.Conv2d(channels, 1, 3, 1, 1)))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, reference, image):
        return self.net(torch.cat([reference, image], dim=1))


def feature_distance(extractor, prediction, target):
    with torch.no_grad():
        expected = extractor(target)
    actual = extractor(prediction)
    return sum(F.l1_loss(a, b) for a, b in zip(actual, expected))/len(actual)


def reliable_teacher_detail(prediction, teacher, target):
    """Distill only patches where teacher is closer to GT; never use GT at inference."""
    from .small_fullframe import _edge
    with torch.no_grad():
        student_error = F.avg_pool2d((prediction-target).square().mean(1, keepdim=True), 9, 1, 4)
        teacher_error = F.avg_pool2d((teacher-target).square().mean(1, keepdim=True), 9, 1, 4)
        weight = (teacher_error < student_error).to(prediction.dtype)
    pixel = ((prediction-teacher).abs()*weight).mean()
    detail = ((_edge(prediction)-_edge(teacher)).abs()*weight).mean()
    return pixel + detail


def detail_patches(reference, target, *images, size=96):
    """Two nonidentical detail crops per image; GT selection is train/eval only."""
    with torch.no_grad():
        dx = F.pad((target[:,:,:,1:]-target[:,:,:,:-1]).abs().mean(1, keepdim=True), (0,1,0,0))
        dy = F.pad((target[:,:,1:,:]-target[:,:,:-1,:]).abs().mean(1, keepdim=True), (0,0,0,1))
        changed = (target-reference).abs().mean(1, keepdim=True).clamp(0, .5)
        # Restrict centers to the product region; backgrounds must not dominate.
        score = F.avg_pool2d((dx+dy)*(1+4*changed), size, stride=16)
        rows = []
        for batch in range(len(target)):
            candidates = score[batch,0].clone()
            candidates[:,:2] = -1; candidates[:,-2:] = -1
            for _ in range(2):
                index = int(candidates.argmax())
                y, x = divmod(index, candidates.shape[-1])
                rows.append((batch, 16*y, 16*x))
                candidates[max(0,y-2):y+3,max(0,x-2):x+3] = -1
    return [torch.stack([value[b,:,y:y+size,x:x+size] for b,y,x in rows])
            for value in (target, *images)]


def local_detail_loss(features, reference, prediction, target, teacher=None):
    crops = detail_patches(reference, target, prediction, *([] if teacher is None else [teacher]))
    gt, student = crops[:2]
    perceptual = feature_distance(features, student, gt)
    # Signed RGB gradients retain direction and chromatic texture, unlike edge magnitude alone.
    texture = (F.l1_loss(student[:,:,:,1:]-student[:,:,:,:-1], gt[:,:,:,1:]-gt[:,:,:,:-1])
               +F.l1_loss(student[:,:,1:,:]-student[:,:,:-1,:], gt[:,:,1:,:]-gt[:,:,:-1,:]))
    loss = perceptual + texture
    if teacher is not None:
        loss = loss + .5*reliable_teacher_detail(student, crops[2], gt)
    return loss


def audit(model, report, args):
    device = torch.device(args.device)
    model.eval()
    torch.manual_seed(42)
    reference = torch.randn(1, 3, 256, 256, device=device).clamp(-1, 1)
    embeddings = torch.randn(1, 8, model.text.config.input_width, device=device)
    mask = torch.ones(1, 8, dtype=torch.bool, device=device)
    calls = {}
    shared = {}
    hooks = []
    # Check identity of the branch INPUT tensors, not just output shapes.
    def before(name):
        def hook(module, inputs):
            calls[name] = calls.get(name, 0)+1
            if name in ("language_e1", "language_e2"):
                shared[name] = inputs[0].detach().clone()
        return hook
    hooks += [model.text.layers[i].register_forward_pre_hook(before(f"language_e{i+1}")) for i in range(2)]
    spatial = model.editor.bottleneck if model.kind == "small" else model.unet.mid_block.hybrid
    for i in range(2):
        def spatial_before(module, inputs, stage=i+1):
            shared[f"vision_e{stage}"] = inputs[0].detach().clone()
        hooks.append(getattr(spatial, f"electronic{i+1}").register_forward_pre_hook(spatial_before))
    for label, path in (("language", model.text.optical), ("vision", spatial.optical)):
        hooks.append(path.core.router.register_forward_pre_hook(before(label+"_router")))
        # Propagation builds phase modulation using these modules.
        hooks.append(path.core.global_phase.register_forward_pre_hook(before(label+"_global")))
    original = model.text.optical.run_expert_block
    original_global = model.text.optical.encode_global_input
    vision_expert = spatial.optical.run_expert_block
    vision_global = spatial.optical.encode_global_input
    parallel_checks = {}
    def expert(value, padding):
        parallel_checks["language_stage1_same_input"] = torch.equal(value, shared["language_e1"].float())
        return original(value, padding)
    def global_input(value, padding, routing):
        parallel_checks["language_stage2_same_input"] = torch.equal(value, shared["language_e2"].float())
        return original_global(value, padding, routing)
    model.text.optical.run_expert_block = expert
    model.text.optical.encode_global_input = global_input
    def expert_vision(value, padding):
        parallel_checks["vision_stage1_same_input"] = torch.equal(value, shared["vision_e1"].float())
        return vision_expert(value, padding)
    def global_vision(value, padding, routing):
        parallel_checks["vision_stage2_same_input"] = torch.equal(value, shared["vision_e2"].float())
        return vision_global(value, padding, routing)
    spatial.optical.run_expert_block = expert_vision
    spatial.optical.encode_global_input = global_vision
    try:
        prediction = model(reference, embeddings, mask, noise_for(model, reference))
        loss = prediction.square().mean()
        loss.backward()
        gradients = {}
        for name, parameter in model.named_parameters():
            if "raw_phase" in name or "raw_router_phase" in name:
                gradients[name] = None if parameter.grad is None else float(parameter.grad.abs().sum())
        report.update({"output_shape": list(prediction.shape), "finite_output": bool(prediction.isfinite().all()),
                       "actual_calls": calls, "parallel_input_checks": parallel_checks,
                       "phase_gradient_l1": gradients,
                       "all_phase_gradients_present": all(v is not None for v in gradients.values())})
        assert prediction.shape == reference.shape and bool(prediction.isfinite().all())
        assert all(parallel_checks.values())
        assert all(calls.get(name, 0) == 1 for name in ("language_e1", "language_e2", "language_router", "language_global", "vision_router", "vision_global"))
        assert report["all_phase_gradients_present"], gradients
        # Top-2 routing deliberately gives unselected experts zero gradients.
        # Require router/global and exactly the selected expert set to be live.
        for path in (model.text.optical, spatial.optical):
            assert path.core.router.raw_router_phase.grad.abs().sum() > 0
            assert path.core.global_phase.phase.raw_phase.grad.abs().sum() > 0
            assert sum(bool(expert.raw_phase.grad.abs().sum() > 0)
                       for expert in path.core.expert_layers[0].experts) == 2
        # The legacy RGB head starts nonzero in trained weights. Randomly
        # initialized zero output heads must not be accepted by this audit.
        report["audit_passed"] = True
        if getattr(model, "decoder_refinement", False):
            norms = {name: float(module.conv2.weight.detach().abs().sum())
                     for name,module in model.named_modules()
                     if ".details." in name and hasattr(module, "conv2")}
            report["decoder_detail_output_weight_l1"] = norms
            report["all_seven_detail_blocks_trained"] = len(norms) == 7 and all(v > 0 for v in norms.values())
        report["optical_diagnostics"] = optical_diagnostics(model)
    finally:
        model.text.optical.run_expert_block = original
        model.text.optical.encode_global_input = original_global
        spatial.optical.run_expert_block = vision_expert
        spatial.optical.encode_global_input = vision_global
        for hook in hooks:
            hook.remove()
        model.zero_grad(set_to_none=True)
    write_json(args.output, report)
    return report


def train(model, report, args):
    from .qwen_mini_small import PromptEmbeddingLookup, QwenMiniTextEncoder
    from .small_fullframe import build_dataset
    if args.output.exists():
        raise FileExistsError("Use a new directory; legacy weights are never overwritten")
    args.output.mkdir(parents=True)
    lookup = PromptEmbeddingLookup(args.embedding_cache)
    mentor = QwenMiniTextEncoder(model.text.config)
    mentor_state = {key: value for key,value in model.text.state_dict().items() if key in mentor.state_dict()}
    mentor.load_state_dict(mentor_state)
    mentor = mentor.to(args.device).eval().requires_grad_(False)
    features = DetailFeatures().to(args.device) if args.perceptual_weight or args.local_detail_weight else None
    image_teacher = None
    if args.image_teacher_checkpoint:
        import copy
        teacher_args = copy.copy(args)
        teacher_args.source = args.image_teacher_source
        teacher_args.checkpoint = args.image_teacher_checkpoint
        teacher_args.adapter_rank = None
        teacher_args.text_mlp_width = None
        teacher_args.expand_decoder = False
        image_teacher, _ = load_model(teacher_args)
        image_teacher.eval().requires_grad_(False)
    datasets = {split: build_dataset("unified_expanded", args.data_dir, split, 256, args.instruction_cache)
                for split in ("train", "val")}
    loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, persistent_workers=args.num_workers > 0)
    # Fine-tune only new paths and language initially. Existing image weights
    # and VAE remain frozen, but decoder gradients still reach the new paths.
    model.requires_grad_(False)
    model.text.requires_grad_(True)
    spatial = model.editor.bottleneck if model.kind == "small" else model.unet.mid_block.hybrid
    spatial.requires_grad_(True)
    if args.joint_training:
        model.requires_grad_(True)
        if model.vae is not None:
            model.vae.requires_grad_(False)
    detail_parameters = [p for n,p in model.named_parameters() if p.requires_grad and ".details." in n]
    main_parameters = [p for n,p in model.named_parameters() if p.requires_grad and ".details." not in n]
    optimizer = torch.optim.AdamW([
        {"params": main_parameters, "lr": args.learning_rate},
        {"params": detail_parameters, "lr": args.learning_rate*args.decoder_lr_multiplier}], lr=args.learning_rate)
    discriminator = DetailDiscriminator().to(args.device) if args.adversarial_weight else None
    discriminator_optimizer = (torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate,
                                               betas=(.5, .999)) if discriminator is not None else None)
    gate_values = {}
    gate_hook = None
    if args.source_gate_weight:
        if model.kind != "small" or model.editor.source_gate is None:
            raise ValueError("Source gate supervision requires the small gated editor")
        gate_hook = model.editor.source_gate.register_forward_hook(
            lambda module, inputs, output: gate_values.update(logits=output))
    torch.manual_seed(args.seed)
    history = []
    validation_history = []
    best_mse = float("inf")
    from torch.utils.data import Subset
    val_count = min(args.validation_samples, len(datasets["val"]))
    indices = torch.linspace(0, len(datasets["val"])-1, val_count).long().tolist()
    validation_loader = DataLoader(Subset(datasets["val"], indices), batch_size=args.batch_size, num_workers=0)
    def validate(step):
        nonlocal best_mse
        model.eval()
        total, detail_total, count = 0., 0., 0
        generator = torch.Generator(device=args.device).manual_seed(args.seed+1000)
        with torch.no_grad():
            for batch in validation_loader:
                reference, target = batch["reference"].to(args.device), batch["target"].to(args.device)
                embeddings, mask, _ = lookup.batch(list(batch["prompt"]), torch.device(args.device))
                prediction = model(reference, embeddings.float(), mask, noise_for(model, reference, generator))
                total += float(F.mse_loss(prediction, target))*len(reference)
                if features is not None:
                    from .small_fullframe import _edge
                    detail_total += float(args.perceptual_weight*feature_distance(features, prediction, target)
                                          +args.edge_loss_weight*F.l1_loss(_edge(prediction), _edge(target)))*len(reference)
                    if args.local_detail_weight:
                        detail_total += float(args.local_detail_weight*local_detail_loss(features, reference, prediction, target))*len(reference)
                count += len(reference)
        mse = total/count
        score = mse + detail_total/count
        validation_history.append({"step": step, "mse": mse, "quality_score": score, "samples": count})
        print(json.dumps({"validation_step": step, "mse": mse, "quality_score": score, "best_before": best_mse}), flush=True)
        if score < best_mse:
            best_mse = score
            torch.save({"architecture": ARCHITECTURE, "source": str(args.source),
                        "model": {k:v.detach().cpu() for k,v in model.state_dict().items()},
                        "adapter_rank": model.adapter.basis.shape[0] if model.adapter is not None else None,
                        "text_mlp_indices": getattr(model, "text_mlp_indices", None),
                        "decoder_refinement": getattr(model, "decoder_refinement", False),
                        "construction": model.construction,
                        "step": step, "validation_mse": mse}, args.output/"best_model.pt")
        write_json(args.output/"progress.json", {"validation": validation_history, "best_mse": best_mse})
        model.train()
    validate(0)
    model.train()
    step = 0
    while step < args.steps:
        for batch in loader:
            if args.clean_optics:
                # Eval mode disables inherited DC20 random perturbations only;
                # gradients and phase learning remain enabled, geometry unchanged.
                model.text.optical.eval()
                spatial.optical.eval()
            reference, target = batch["reference"].to(args.device), batch["target"].to(args.device)
            embeddings, mask, _ = lookup.batch(list(batch["prompt"]), torch.device(args.device))
            prediction = model(reference, embeddings.float(), mask, noise_for(model, reference))
            loss = F.mse_loss(prediction, target) + .1 * F.l1_loss(prediction, target)
            if args.source_gate_weight:
                difference = (target-reference).abs().mean(1, keepdim=True)
                retention = ((.10-difference)/.08).clamp(0, 1)
                loss += args.source_gate_weight*F.binary_cross_entropy_with_logits(gate_values["logits"], retention)
            if args.changed_region_weight:
                changed = ((target-reference).abs().mean(1, keepdim=True) > .08).float()
                loss += args.changed_region_weight*((prediction-target).abs()*changed).sum()/(3*changed.sum()+1e-6)
            if discriminator is not None:
                discriminator.train().requires_grad_(True)
                discriminator_optimizer.zero_grad(set_to_none=True)
                real_logits = discriminator(reference, target)
                fake_logits = discriminator(reference, prediction.detach())
                discriminator_loss = F.relu(1-real_logits).mean()+F.relu(1+fake_logits).mean()
                discriminator_loss.backward()
                discriminator_optimizer.step()
                discriminator.eval().requires_grad_(False)
                ramp = min(1., (step+1)/args.adversarial_warmup)
                loss -= args.adversarial_weight*ramp*discriminator(reference, prediction).mean()
            if model.kind == "large" and args.latent_loss_weight:
                with torch.no_grad():
                    target_latent = model.vae.encode(target).latent_dist.mode()*model.vae.config.scaling_factor
                loss += args.latent_loss_weight*F.mse_loss(model.current_latent, target_latent)
            if args.edge_loss_weight:
                from .small_fullframe import _edge
                loss += args.edge_loss_weight*F.l1_loss(_edge(prediction), _edge(target))
            if features is not None:
                loss += args.perceptual_weight*feature_distance(features, prediction, target)
            if image_teacher is not None:
                with torch.no_grad():
                    teacher_image = image_teacher(reference, embeddings.float(), mask,
                                                  noise_for(image_teacher, reference))
                loss += args.image_distillation_weight*reliable_teacher_detail(prediction, teacher_image, target)
            if args.local_detail_weight:
                loss += args.local_detail_weight*local_detail_loss(features, reference, prediction, target,
                                                                  teacher_image if image_teacher is not None else None)
            with torch.no_grad():
                teacher_hidden = mentor.hidden(embeddings.float(), mask)
            student_hidden = model.text.current_hidden
            text_loss = F.mse_loss(student_hidden, teacher_hidden) + (1-F.cosine_similarity(student_hidden, teacher_hidden).mean())
            loss = loss + args.text_distillation_weight*text_loss
            if not bool(loss.isfinite()):
                raise FloatingPointError("Nonfinite loss; no checkpoint is promoted")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            history.append(float(loss.detach()))
            step += 1
            if step == 1 or step % 100 == 0:
                print(json.dumps({"step": step, "loss": history[-1]}), flush=True)
            if step % args.validate_every == 0:
                validate(step)
            if step >= args.steps:
                break
    if step % args.validate_every:
        validate(step)
    best = torch.load(args.output/"best_model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best["model"])
    best_step = best["step"]
    del best
    model.eval()
    totals, count = 0., 0
    previews = []
    validation_generator = torch.Generator(device=args.device).manual_seed(args.seed+1000)
    with torch.no_grad():
        for batch in validation_loader:
            reference, target = batch["reference"].to(args.device), batch["target"].to(args.device)
            embeddings, mask, _ = lookup.batch(list(batch["prompt"]), torch.device(args.device))
            prediction = model(reference, embeddings.float(), mask, noise_for(model, reference, validation_generator))
            for index in range(len(reference)):
                if len(previews) < 12:
                    previews.append((batch["prompt"][index], reference[index].cpu(), target[index].cpu(), prediction[index].cpu()))
            totals += float(F.mse_loss(prediction, target))*len(reference)
            count += len(reference)
    report.update({"status": "joint recovery candidate; preview and held-out test required", "steps": step,
                   "validation_mse": totals/count, "validation_samples": count, "loss_history": history,
                   "validation_history": validation_history, "best_step": best_step,
                   "joint_training": args.joint_training, "clean_optics_training": args.clean_optics,
                   "perceptual_weight": args.perceptual_weight,
                   "image_teacher_checkpoint": str(args.image_teacher_checkpoint),
                   "image_distillation_weight": args.image_distillation_weight,
                   "adversarial_weight": args.adversarial_weight,
                   "discriminator_training_only": discriminator is not None,
                   "source_gate_weight": args.source_gate_weight, "changed_region_weight": args.changed_region_weight,
                   "local_detail_weight": args.local_detail_weight, "detail_crop_size": 96,
                   "decoder_lr_multiplier": args.decoder_lr_multiplier,
                   "selection_metric": "MSE + weighted ResNet18 perceptual + weighted edge L1 + weighted local detail" if features else "MSE",
                   "legacy_source": str(args.source), "modes": ["background", "object", "joint"]})
    report["optical_diagnostics"] = optical_diagnostics(model)
    torch.save({"architecture": ARCHITECTURE, "source": str(args.source), "model": model.cpu().state_dict(),
                "adapter_rank": model.adapter.basis.shape[0] if model.adapter is not None else None,
                "text_mlp_indices": getattr(model, "text_mlp_indices", None),
                "decoder_refinement": getattr(model, "decoder_refinement", False),
                "construction": model.construction,
                "report": report}, args.output/"adapted_model.pt")
    if gate_hook is not None:
        gate_hook.remove()
    from PIL import Image, ImageDraw
    canvas = Image.new("RGB", (320+3*256, 256*len(previews)), "white")
    draw = ImageDraw.Draw(canvas)
    for row, (prompt, reference, target, predicted) in enumerate(previews):
        draw.text((4, row*256+4), "input | target | generated", fill="black")
        for line in range(5):
            draw.text((4, row*256+24+line*16), prompt[line*43:(line+1)*43], fill="black")
        for column, value in enumerate((reference, target, predicted)):
            array = value.clamp(-1, 1).add(1).mul(127.5).byte().permute(1, 2, 0).numpy()
            canvas.paste(Image.fromarray(array), (320+column*256, row*256))
    canvas.save(args.output/"validation_preview.jpg", quality=95)
    write_json(args.output/"report.json", report)
    return report


def infer(model, report, args):
    if args.checkpoint is None:
        raise ValueError("Inference requires an adapted v2 checkpoint, not an initialized migration")
    from PIL import Image
    from .qwen_mini_infer import _load_rgb
    from .half_qwen import load_half_qwen_text_encoder
    from .feature_cache import _qwen_prompts
    language, processor, _ = load_half_qwen_text_encoder(args.qwen_checkpoint, torch.device(args.device), keep_layers=1)
    encoded = _qwen_prompts(processor, [args.prompt])
    ids = encoded["input_ids"][:, -model.text.config.max_length:].to(args.device)
    mask = encoded["attention_mask"][:, -model.text.config.max_length:].to(args.device)
    with torch.no_grad():
        embeddings = language.embed_tokens(ids).float()
    del language, processor
    reference = _load_rgb(args.input_image, 256, torch.device(args.device))
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    model.eval()
    with torch.no_grad():
        generated = model(reference, embeddings, mask, noise_for(model, reference, generator))
    image = generated[0].clamp(-1, 1).add(1).mul(127.5).byte().permute(1, 2, 0).cpu().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(args.output)
    report.update({"prompt": args.prompt, "seed": args.seed, "hard_pixel_composite": False,
                   "generator_calls": 1, "output": str(args.output)})
    write_json(args.output.with_suffix(".json"), report)
    return report


@torch.no_grad()
def evaluate(model, report, args):
    from PIL import Image, ImageDraw
    from .product_unified_edit_data_v2 import ExpandedUnifiedProductEditDataset
    from .qwen_mini_small import PromptEmbeddingLookup
    from .small_fullframe import _edge
    from torch.utils.data import Subset
    model.eval()
    dataset = ExpandedUnifiedProductEditDataset(args.data_dir, args.split, 256, args.instruction_cache)
    count = min(args.validation_samples, len(dataset))
    indices = torch.linspace(0, len(dataset)-1, count).long().tolist()
    lookup = PromptEmbeddingLookup(args.embedding_cache)
    generator = torch.Generator(device=args.device).manual_seed(args.seed+1000)
    metrics = {}
    previews = {}
    routing_totals = {label: {"selected": torch.zeros(4), "probabilities": torch.zeros(4), "samples": 0}
                      for label in ("language", "vision")}
    fid = kid = None
    detail_features = DetailFeatures().to(args.device) if args.detail_evaluation else None
    local_total = 0.
    if args.distribution_metrics:
        from .evaluate_unified_editor import TorchvisionInceptionFeatures
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.kid import KernelInceptionDistance
        extractor = TorchvisionInceptionFeatures().to(args.device)
        fid = FrechetInceptionDistance(feature=extractor).to(args.device)
        kid = KernelInceptionDistance(feature=extractor, subset_size=min(100, count), subsets=10).to(args.device)
    for batch in DataLoader(Subset(dataset, indices), batch_size=args.batch_size, num_workers=args.num_workers):
        reference, target = batch["reference"].to(args.device), batch["target"].to(args.device)
        embeddings, mask, _ = lookup.batch(list(batch["prompt"]), torch.device(args.device))
        output = model(reference, embeddings.float(), mask, noise_for(model, reference, generator))
        if detail_features is not None:
            local_total += float(local_detail_loss(detail_features, reference, output, target))*len(reference)
        spatial = model.editor.bottleneck if model.kind == "small" else model.unet.mid_block.hybrid
        for label, branch in (("language", model.text), ("vision", spatial)):
            entry = routing_totals[label]
            routing = branch.last_routing
            entry["samples"] += len(reference)
            entry["selected"] += routing["selected_mask"].detach().float().sum(0).cpu()
            entry["probabilities"] += routing["probabilities"].detach().float().sum(0).cpu()
        if fid is not None:
            real = target.add(1).mul(127.5).clamp(0, 255).byte()
            fake = output.add(1).mul(127.5).clamp(0, 255).byte()
            fid.update(real, real=True); fid.update(fake, real=False)
            kid.update(real, real=True); kid.update(fake, real=False)
        for i, mode in enumerate(batch["mode"]):
            key = batch["category"][i]+":"+mode
            values = {"mse": float(F.mse_loss(output[i], target[i])),
                      "l1": float(F.l1_loss(output[i], target[i])),
                      "edge_l1": float(F.l1_loss(_edge(output[i:i+1]), _edge(target[i:i+1])))}
            for group in ("overall", mode, key):
                entry = metrics.setdefault(group, {"count": 0, "mse": 0., "l1": 0., "edge_l1": 0.})
                entry["count"] += 1
                for name, value in values.items():
                    entry[name] += value
            previews.setdefault(key, [])
            if len(previews[key]) < 2:
                previews[key].append((batch["prompt"][i], reference[i].cpu(), target[i].cpu(), output[i].cpu()))
    import math
    for entry in metrics.values():
        for name in ("mse", "l1", "edge_l1"):
            entry[name] /= entry["count"]
        entry["psnr_db"] = 10*math.log10(4/max(entry["mse"], 1e-12))
    items = [(key, *item) for key, values in sorted(previews.items()) for item in values]
    canvas = Image.new("RGB", (320+3*256, len(items)*256), "white")
    draw = ImageDraw.Draw(canvas)
    for row, (key, prompt, reference, target, output) in enumerate(items):
        draw.text((4, row*256+4), key+" | input | GT | generated", fill="black")
        for line in range(5):
            draw.text((4, row*256+28+line*16), prompt[line*43:(line+1)*43], fill="black")
        for column, value in enumerate((reference, target, output)):
            array = value.clamp(-1, 1).add(1).mul(127.5).byte().permute(1, 2, 0).numpy()
            canvas.paste(Image.fromarray(array), (320+column*256, row*256))
    args.output.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output/"heldout_preview.jpg", quality=95)
    overview_items = [(key, *values[0]) for key, values in sorted(previews.items())]
    overview = Image.new("RGB", (320+3*256, len(overview_items)*256), "white")
    for row, item in enumerate(overview_items):
        key = item[0]
        original_row = next(i for i, original in enumerate(items) if original[0] == key)
        overview.paste(canvas.crop((0, original_row*256, canvas.width, (original_row+1)*256)), (0, row*256))
    overview.save(args.output/"overview.jpg", quality=95)
    report.update({"split": args.split, "samples": count, "metrics": metrics,
                   "checkpoint": str(args.checkpoint), "optical_diagnostics": optical_diagnostics(model)})
    report["expert_distribution"] = {
        label: {"samples": entry["samples"], "selection_share": (entry["selected"]/(2*entry["samples"])).tolist(),
                "mean_probability": (entry["probabilities"]/entry["samples"]).tolist()}
        for label, entry in routing_totals.items()}
    if fid is not None:
        mean, std = kid.compute()
        report["distribution_metrics"] = {"fid_imagenet_inception_variant": float(fid.compute()),
                                           "kid_mean": float(mean), "kid_std": float(std),
                                           "extractor": "torchvision ImageNet Inception-v3, not canonical TensorFlow FID",
                                           "caveat": "Closed target catalogue and composed scenes; not open-set image generation"}
    if detail_features is not None:
        report["local_detail_evaluation"] = {
            "samples": count, "crops_per_image": 2, "crop_size": 96,
            "resnet18_perceptual_plus_signed_rgb_gradient_l1": local_total/count,
            "note": "Internal GT-selected crop metric, also used in training; not independent LPIPS"}
    write_json(args.output/"evaluation.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("audit", "train", "infer", "evaluate", "seal"))
    for name in ("source", "initial-unet", "vae-checkpoint", "adapter-checkpoint", "checkpoint",
                 "output", "embedding-cache", "data-dir", "instruction-cache", "qwen-checkpoint", "input-image"):
        parser.add_argument("--"+name, type=Path, required=name == "output")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--validation-samples", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--text-distillation-weight", type=float, default=.1)
    parser.add_argument("--joint-training", action="store_true")
    parser.add_argument("--clean-optics", action="store_true")
    parser.add_argument("--edge-loss-weight", type=float, default=0.)
    parser.add_argument("--latent-loss-weight", type=float, default=0.)
    parser.add_argument("--perceptual-weight", type=float, default=0.)
    parser.add_argument("--local-detail-weight", type=float, default=0.)
    parser.add_argument("--image-teacher-source", type=Path)
    parser.add_argument("--image-teacher-checkpoint", type=Path)
    parser.add_argument("--image-distillation-weight", type=float, default=.1)
    parser.add_argument("--adversarial-weight", type=float, default=0.)
    parser.add_argument("--adversarial-warmup", type=int, default=200)
    parser.add_argument("--text-mlp-width", type=int)
    parser.add_argument("--expand-decoder", action="store_true")
    parser.add_argument("--decoder-lr-multiplier", type=float, default=1.)
    parser.add_argument("--source-gate-weight", type=float, default=0.)
    parser.add_argument("--changed-region-weight", type=float, default=0.)
    parser.add_argument("--adapter-rank", type=int)
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--distribution-metrics", action="store_true")
    parser.add_argument("--detail-evaluation", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt")
    args = parser.parse_args()
    if args.image_teacher_checkpoint and not args.image_teacher_source:
        parser.error("--image-teacher-checkpoint requires --image-teacher-source")
    if args.adversarial_warmup < 1:
        parser.error("--adversarial-warmup must be positive")
    try:
        model, report = load_model(args)
        result = {"audit": audit, "train": train, "infer": infer, "evaluate": evaluate, "seal": seal}[args.command](model, report, args)
        print(json.dumps({k: v for k, v in result.items() if k not in ("phase_gradient_l1", "phase_tensors", "loss_history")}, indent=2))
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
