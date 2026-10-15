"""Bounded operator-level causal probes; never edits formal results or weights.

CPU: replay saved Qwen BF16 operands, testing specific reduction hypotheses.
GPU: isolated Qwen linear replay/profiler, or DeepSeek native attention tracing
and an explicitly labelled FP32-primitives numerical control. No training.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

from . import diagnose_batch_layers as layers
from . import diagnose_batch_numerics as numerics
from . import evaluate_temporal_baseline_batches as frozen

TASK = Path(__file__).resolve().parent
OWNER = TASK / "runs/simulation/lgvq_batch_layers_a100_20261015"
FOCUSED = OWNER / "focused_merger_routes03"
BATCHES = (1, 2, 4)
REFERENCE_LIMIT = 2 * 1024**3
WATCHED = {"aten.mm.default", "aten.addmm.default", "aten.bmm.default",
           "aten.linear.default", "aten.matmul.default", "aten.softmax.int",
           "aten._softmax.default", "aten.add.Tensor", "aten.mul.Tensor",
           "aten.div.Tensor", "aten.cos.default", "aten.sin.default"}


def difference(value, reference):
    import torch
    if value.shape != reference.shape:
        raise ValueError(f"Unaligned operator output: {value.shape}, {reference.shape}")
    delta = value.double() - reference.double()
    # Attention masks may contain -inf; identical infinities are not differences.
    equal = value == reference
    delta = torch.where(equal, torch.zeros_like(delta), delta)
    if not torch.isfinite(delta).all():
        raise ValueError("Nonfinite unmatched values")
    return dict(max_abs=float(delta.abs().max()), changed=int((~equal).sum()),
                entries=value.numel(), rms=float(delta.square().mean().sqrt()))


def qwen_operands():
    import torch
    from safetensors import safe_open
    command = json.loads((FOCUSED / "gpu_qwen.command.json").read_text())
    model = Path(command[command.index("--model") + 1])
    prefix = "model.visual.deepstack_merger_list.0.linear_fc1"
    index = model / "model.safetensors.index.json"
    mapping = json.loads(index.read_text())["weight_map"] if index.exists() else {}
    values = []
    for suffix in ("weight", "bias"):
        key = prefix + "." + suffix
        with safe_open(str(model / mapping.get(key, "model.safetensors")),
                       framework="pt", device="cpu") as source:
            values.append(source.get_tensor(key).to(torch.bfloat16))
    captures = {}
    x = None
    for batch in (1, 4):
        data = torch.load(FOCUSED / f"gpu_qwen/native_b{batch}_focused_boundaries.pt",
                          map_location="cpu", weights_only=False)["tensors"]
        current = data[prefix + ":input/video0"]
        if x is not None and not torch.equal(x, current):
            raise ValueError("Saved native inputs differ")
        x = current
        captures[batch] = data[prefix + ":output/video0"]
    return x.to(torch.bfloat16), *values, captures


def qwen_cpu(output):
    import torch
    x, weight, bias, captured = qwen_operands()
    reference = x.double() @ weight.double().T + bias.double()
    results = {"single_round": (reference.to(torch.bfloat16))}
    # Fixed mechanistic hypotheses, not a performance/metric parameter search.
    for partitions in (2, 4, 8, 16):
        widths = x.shape[1] // partitions
        partials = [(x[:, i:i+widths].double() @ weight[:, i:i+widths].double().T)
                    .to(torch.bfloat16).double() for i in range(0, x.shape[1], widths)]
        results[f"split{partitions}_partial_bf16_fp64_sum"] = (sum(partials) + bias.double()).to(torch.bfloat16)
        total = torch.zeros_like(partials[0], dtype=torch.bfloat16)
        for part in partials:
            total = (total.double() + part).to(torch.bfloat16)
        results[f"split{partitions}_partial_and_running_bf16"] = (total.double() + bias.double()).to(torch.bfloat16)
    frozen.write_json(output / "qwen_cpu_mechanisms.json", dict(
        operand_sha256={k: numerics.tensor_hash(v) for k, v in
                        (("input_bf16", x), ("weight_bf16", weight), ("bias_bf16", bias))},
        hypotheses={name: {f"captured_b{b}": difference(v, target) for b, target in captured.items()}
                    for name, v in results.items()},
        limits="CPU emulation can falsify specified reductions; no CUDA-kernel identity claim"))


def qwen_gpu(output):
    import torch
    from .reproduce_temporal_baseline import cuda_guard
    cuda_guard()
    x, weight, bias, captured = qwen_operands()
    x, weight, bias = [v.cuda() for v in (x, weight, bias)]
    torch.backends.cuda.matmul.allow_tf32 = False
    results, saved = {}, {}
    for mode in ("native_linear", "matmul_then_bias", "fp32_linear", "reduction_disabled"):
        previous_flag = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        if mode == "reduction_disabled":
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        try:
            rows = {}
            single = None
            for batch in BATCHES:
                frozen.process_audit()
                inputs = x.repeat(batch, 1)
                def operation():
                    if mode == "matmul_then_bias":
                        return inputs @ weight.T + bias
                    if mode == "fp32_linear":
                        return torch.nn.functional.linear(inputs.float(), weight.float(), bias.float()).to(torch.bfloat16)
                    return torch.nn.functional.linear(inputs, weight, bias)
                # Only a small isolated operator. These are NOT model cost measurements.
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA]) as prof:
                    value = operation()
                    torch.cuda.synchronize()
                first = value[:len(x)].cpu()
                if single is None:
                    single = first
                rows[str(batch)] = dict(vs_own_b1=difference(first, single),
                    vs_original_capture=difference(first, captured[1 if batch==1 else 4]),
                    cuda_kernel_names=sorted({e.name for e in prof.events()
                        if e.device_type == torch.autograd.DeviceType.CUDA}))
                saved[f"{mode}_b{batch}_video0"] = first
                frozen.write_json(output / "qwen_operator_gpu.json", dict(results={**results, mode: rows},
                    limits="Repeated saved video0 rows isolate matrix shape, not full-model accuracy or latency"))
            results[mode] = rows
        finally:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = previous_flag
    torch.save(saved, output / "qwen_operator_outputs.pt")
    frozen.process_audit()


class OperatorCapture:
    """Compare full video0 operator outputs; persist bounded diagnostic slices."""
    def __init__(self):
        self.references, self.stats, self.samples, self.events = {}, {}, {}, []
        self.batch, self.reference, self.byte_count = 1, True, 0
        self.counts = {}

    def reset(self, batch):
        self.batch, self.reference = batch, batch == 1
        self.counts, self.stats, self.samples, self.events = {}, {}, {}, []

    def record(self, name, tensor):
        import torch
        index = self.counts.get(name, 0)
        self.counts[name] = index + 1
        key = f"{name}/{index}"
        value = tensor.detach()
        if value.ndim and value.shape[0] % self.batch == 0:
            value = value[:value.shape[0] // self.batch]
        value = value.cpu().clone()
        self.events.append(key)
        if self.reference:
            self.byte_count += value.numel() * value.element_size()
            if self.byte_count > REFERENCE_LIMIT:
                raise MemoryError("Operator CPU-reference memory bound exceeded")
            self.references[key] = value
        elif key in self.references:
            self.stats[key] = difference(value, self.references[key])
        else:
            raise ValueError(f"Operator sequence changed at {key}")
        # Large attention matrices: retain fixed query rows, not just aggregate stats.
        sample = value
        if sample.numel() > 8_000_000:
            if sample.ndim < 3:
                raise MemoryError("Unexpected large non-attention operator")
            indices = sorted({0, 1, 31, 127, 255, 511, 1023, sample.shape[-2]-1})
            indices = [i for i in indices if i < sample.shape[-2]]
            sample = sample.index_select(-2, torch.tensor(indices))
        self.samples[key] = dict(value=sample, full_shape=list(value.shape),
                                full_sha256=numerics.tensor_hash(value), dtype=str(value.dtype))


def attention_mode(capture, controlled=False, record=True):
    """Observe native ATen calls, optionally FP32-evaluate attention GEMM only."""
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    class Mode(TorchDispatchMode):
        def __torch_dispatch__(self, function, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            name = str(function)
            arithmetic = name in {"aten.mm.default", "aten.addmm.default", "aten.bmm.default",
                                  "aten.linear.default", "aten.matmul.default"}
            if controlled and arithmetic:
                dtype = next(x.dtype for x in args if torch.is_tensor(x) and x.is_floating_point())
                converted = tuple(x.float() if torch.is_tensor(x) and x.is_floating_point() else x for x in args)
                with torch.autocast("cuda", enabled=False):
                    out = function(*converted, **kwargs).to(dtype)
            else:
                out = function(*args, **kwargs)
            if record and name in WATCHED and torch.is_tensor(out) and out.is_floating_point():
                capture.record(name, out)
            return out
    return Mode()


def deepseek_gpu(output):
    import torch
    from . import run_batch_layer_diagnostics as suite
    from .reproduce_temporal_baseline import cuda_guard
    cuda_guard()
    command = suite.child_command("deepseek", "gpu", output)
    # Extract only file options explicitly; no dependence on positional command layout.
    options = {key: Path(command[command.index("--"+key.replace("_", "-"))+1])
               for key in ("manifest", "checkpoint", "model", "previous_diagnostic", "reproduction_receipt")}
    args = argparse.Namespace(baseline="deepseek", output=output, **options)
    # preflight requires a nonexisting output; source assets were already receipt-bound.
    preflight = argparse.Namespace(**vars(args)); preflight.output = output / "preflight_unused"
    frozen.preflight_files(preflight)
    old = json.loads((args.previous_diagnostic / "diagnosis.json").read_text())
    model, head, rows, prepare, transfer, _, timer, timer_handles = frozen.load_runtime(args)
    layers.detach_cost_instrumentation(timer, timer_handles)
    scores = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["level_scores"].float()
    attentions = [(name, m) for name, m in model.named_modules()
                  if name.startswith("language.model.layers.") and name.endswith(".self_attn")]
    if not attentions or attentions[0][1].__class__.__name__ != "LlamaAttention":
        raise ValueError("Actual first attention class changed")
    results = {}
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for mode in ("native", "attention_fp32_gemm_control"):
            capture, contexts, handles = OperatorCapture(), [], []
            for index, (name, module) in enumerate(attentions):
                def enter(_module, positional, keyword, index=index):
                    if index == 0 and capture.record_group:
                        for key in ("attention_mask", "position_ids", "cache_position"):
                            value = keyword.get(key)
                            if torch.is_tensor(value):
                                capture.record("actual_argument_"+key, value)
                    context = attention_mode(capture, mode != "native", index == 0 and capture.record_group)
                    context.__enter__(); contexts.append(context)
                def leave(_module, positional, result):
                    contexts.pop().__exit__(None, None, None)
                handles += [module.register_forward_pre_hook(enter, with_kwargs=True),
                            module.register_forward_hook(leave)]
            mode_results = {}
            try:
                for batch in BATCHES:
                    capture.reset(batch)
                    predictions, features, logits = [], [], []
                    for start in range(0, 8, batch):
                        frozen.process_audit()
                        cpu, _ = prepare(rows[start:start+batch])
                        inputs = layers.verified_transfer(cpu, batch,
                            old["inputs"][str(batch)]["input_fingerprints"][start:start+batch], transfer)
                        capture.record_group = start == 0
                        feature, logit = numerics.diagnostic_forward(args, model, head, inputs, False)
                        if start == 0 and not all(any(event.startswith(name+"/") for event in capture.events)
                            for name in ("aten.linear.default", "aten.matmul.default", "aten.softmax.int")):
                            raise ValueError("Missing native composite attention operators; no causal conclusion")
                        values = (logit.softmax(-1) @ scores).tolist()
                        if mode == "native" and max(abs(a-b) for a,b in zip(values,
                            old["probes"]["native"][str(batch)]["predictions"][start:start+batch])) > .001:
                            raise ValueError("Instrumentation changed native predictions")
                        predictions.extend(values); features.append(feature); logits.append(logit)
                        del inputs
                    first = next((key for key in capture.events if capture.stats.get(key, {}).get("changed")), None)
                    mode_results[str(batch)] = dict(predictions=predictions, first_divergent_observed_operator=first,
                        statistics=capture.stats, events=capture.events, reference_bytes=capture.byte_count)
                    torch.save(dict(operators=capture.samples, features=torch.cat(features), logits=torch.cat(logits)),
                               output / f"{mode}_b{batch}_operators.pt")
                    frozen.write_json(output / "deepseek_operators.json", dict(results={**results, mode:mode_results},
                        scope="Same eight TEST; operator comparisons full video0, large evidence tensors query-sliced; numerical controls not formal scores"))
                    print(json.dumps(dict(mode=mode, batch=batch, first_operator=first)), flush=True)
                results[mode] = mode_results
            finally:
                while contexts: contexts.pop().__exit__(None, None, None)
                for handle in handles: handle.remove()
    finally:
        timer.close()
    frozen.process_audit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--baseline", choices=("qwen", "deepseek"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output = args.output.resolve()
    if not args.output.is_relative_to(OWNER.resolve()):
        raise ValueError("Canonical diagnostic run required")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != ("-1" if args.phase=="cpu" else frozen.A100_UUID):
        raise ValueError("Wrong device visibility")
    args.output.mkdir(parents=True, exist_ok=False)
    identity = dict(status="running", pid=os.getpid(), phase=args.phase, baseline=args.baseline,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_sha256=frozen.sha256(Path(__file__)), scope="Diagnostic only; original weights/data/costs untouched")
    frozen.write_json(args.output / "status.json", identity)
    try:
        import torch
        torch.set_num_threads(2)
        with torch.inference_mode():
            if args.phase=="cpu":
                if args.baseline!="qwen": raise ValueError("CPU mechanism phase only supports saved Qwen operands")
                qwen_cpu(args.output)
            elif args.baseline=="qwen": qwen_gpu(args.output)
            else: deepseek_gpu(args.output)
        frozen.write_json(args.output / "status.json", {**identity, "status":"complete"})
    except BaseException as error:
        frozen.write_json(args.output / "status.json", {**identity, "status":"failed", "error":repr(error)})
        raise


if __name__ == "__main__":
    main()
