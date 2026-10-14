"""Bounded same-input frozen diagnostics. Never supplies paper cost or SRCC."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

from . import evaluate_temporal_baseline_batches as frozen


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(-1).view(__import__("torch").uint8).numpy().tobytes()).hexdigest()


def input_records(inputs, count):
    """Fingerprint valid text and image tensors per video, ignoring text padding."""
    import torch
    if isinstance(inputs, torch.Tensor):
        return [{"frames": tensor_hash(part)} for part in inputs.reshape(count, 4, *inputs.shape[1:])]
    result = [{} for _ in range(count)]
    for key in ("input_ids", "attention_mask"):
        if key in inputs:
            for index in range(count):
                result[index][key] = tensor_hash(inputs[key][index][inputs["attention_mask"][index].bool()])
    if "pixel_values_videos" in inputs:
        grid = inputs["video_grid_thw"]
        if len(grid) != count:
            raise RuntimeError("Unexpected video grid grouping")
        start = 0
        for index, shape in enumerate(grid):
            length = int(shape.prod())
            result[index]["grid"] = tensor_hash(shape)
            result[index]["pixels"] = tensor_hash(inputs["pixel_values_videos"][start:start+length])
            start += length
        if start != len(inputs["pixel_values_videos"]):
            raise RuntimeError("Video pixel grouping incomplete")
    elif "images" in inputs:
        for index in range(count):
            result[index]["pixels"] = tensor_hash(inputs["images"][index])
    else:
        raise RuntimeError("No recognized image tensor for exact input comparison")
    return result


def diagnostic_forward(args, model, head, inputs, fp32):
    import torch
    dtype = torch.float32 if fp32 else torch.bfloat16
    with torch.autocast("cuda", dtype=dtype, enabled=not fp32):
        if args.baseline == "qwen":
            hidden = model.model(**inputs, return_dict=True, use_cache=False).last_hidden_state
        else:
            embeddings = model.prepare_inputs_embeds(**inputs)
            hidden = model.language.model(inputs_embeds=embeddings,
                attention_mask=inputs.attention_mask, use_cache=False, return_dict=True).last_hidden_state
    mask = inputs["attention_mask"].bool()
    indices = torch.arange(mask.shape[1], device=hidden.device).expand_as(mask)
    last = indices.masked_fill(~mask, -1).amax(1)
    feature = hidden[torch.arange(len(hidden), device=hidden.device), last].float()
    if args.baseline == "qwen":
        _native_logits = model.lm_head(feature.to(model.lm_head.weight.dtype))
    else:
        feature = feature.half().float()  # retain deployed cache/readout contract
    logits = head(feature).float()
    return feature.cpu(), logits.cpu()


def run(args):
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    if os.environ.get("CUDA_VISIBLE_DEVICES") != frozen.A100_UUID:
        raise RuntimeError("Wrong physical GPU")
    frozen.process_audit()
    args.output.mkdir(parents=True, exist_ok=False)
    identity = {"status": "running", "baseline": args.baseline,
                "checkpoint_sha256": frozen.sha256(args.checkpoint),
                "manifest_sha256": frozen.sha256(args.manifest),
                "pid": os.getpid(), "gpu_uuid": frozen.A100_UUID,
                "scope": "first eight fixed TEST videos, no selection or metric optimization",
                "source_sha256": frozen.sha256(Path(__file__))}
    frozen.write_json(args.output / "status.json", identity)
    timer, handles = None, []
    try:
        with torch.inference_mode():
            model, head, rows, prepare, transfer, forward, timer, handles = frozen.load_runtime(args)
            rows = rows[:8]
            records, inputs_by_batch = {}, {}
            for batch in (1, 2, 4):
                groups, hashes, positions = [], [], []
                for start in range(0, len(rows), batch):
                    cpu, where = prepare(rows[start:start+batch])
                    hashes.extend(input_records(cpu, batch))
                    positions.extend(where)
                    groups.append(cpu)
                records[str(batch)] = {"input_fingerprints": hashes, "selected_frame_positions": positions}
                inputs_by_batch[batch] = groups
            for batch in (2, 4):
                records[str(batch)]["inputs_equal_to_single"] = records[str(batch)]["input_fingerprints"] == records["1"]["input_fingerprints"]
                records[str(batch)]["frames_equal_to_single"] = records[str(batch)]["selected_frame_positions"] == records["1"]["selected_frame_positions"]
            probes = {}
            for mode in ("native", "native_repeat", "native_math_sdpa", "cast_deployed_weights_fp32_math_sdpa"):
                frozen.process_audit()
                fp32 = mode.startswith("cast_")
                if fp32:
                    model.float()
                mode_results = {}
                for batch, groups in inputs_by_batch.items():
                    features, logits, predictions, formal_deltas = [], [], [], []
                    for cpu in groups:
                        inputs = transfer(cpu)
                        if fp32:
                            # Floating inputs only; token indices and masks unchanged.
                            for key, value in inputs.items():
                                if torch.is_tensor(value) and value.is_floating_point():
                                    inputs[key] = value.float()
                        if mode == "native":
                            reference, _, _ = forward(inputs)
                        context = sdpa_kernel(SDPBackend.MATH) if "math_sdpa" in mode else __import__("contextlib").nullcontext()
                        with context:
                            feature, logit = diagnostic_forward(args, model, head, inputs, fp32)
                        value = logit.to(head.weight.device).softmax(-1) @ torch.load(args.checkpoint, map_location="cpu", weights_only=False)["level_scores"].float().to(head.weight.device)
                        value = value.cpu()
                        if mode == "native":
                            delta = max(abs(a-b) for a, b in zip(reference, value.tolist()))
                            formal_deltas.append(delta)
                            if delta > 0.0002:
                                raise RuntimeError(f"Diagnostic forward differs from formal path: {delta}")
                        features.append(feature)
                        logits.append(logit)
                        predictions.extend(value.tolist())
                        del inputs
                    feature, logit = torch.cat(features), torch.cat(logits)
                    # Keep small intermediate evidence, never a model checkpoint.
                    torch.save({"sample_ids": [row["sample_id"] for row in rows],
                                "features": feature, "logits": logit},
                               args.output / f"{mode}_b{batch}.pt")
                    mode_results[str(batch)] = {"predictions": predictions,
                        "formal_forward_max_abs_delta": max(formal_deltas) if formal_deltas else None}
                single = mode_results["1"]["predictions"]
                for batch in (1, 2, 4):
                    mode_results[str(batch)]["max_abs_mos_delta_vs_b1"] = max(abs(a-b) for a,b in zip(single, mode_results[str(batch)]["predictions"]))
                probes[mode] = mode_results
                frozen.write_json(args.output / "diagnosis.json", {**identity, "inputs": records, "probes": probes,
                    "fp32_note": "FP32 casts deployed BF16 weights, does not recover discarded checkpoint mantissa bits; DeepSeek retains FP16 feature quantization; math SDPA only affects operations using PyTorch SDPA"})
            frozen.write_json(args.output / "status.json", {**identity, "status": "complete"})
    except BaseException as error:
        frozen.write_json(args.output / "status.json", {**identity, "status": "failed", "error": repr(error)})
        raise
    finally:
        if timer is not None:
            timer.close()
        for handle in handles:
            handle.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", choices=("qwen", "deepseek"), required=True)
    for name in ("manifest", "checkpoint", "model", "output", "reproduction-receipt"):
        parser.add_argument(f"--{name}", type=Path, required=name != "reproduction-receipt")
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    frozen.preflight_files(args)
    run(args)


if __name__ == "__main__":
    main()
