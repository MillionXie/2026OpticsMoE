"""Auditable, mutually exclusive neural spans in the unchanged model forward.

No torch import on the CPU-contract path. No dummy forward or warm-up. CUDA
events and synchronized wall are paired, with wall fixed as the primary metric.
AST instrumentation wraps existing statements, never substitutes their math.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
from copy import deepcopy
import functools
import hashlib
import inspect
import textwrap
import time
import types


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for b in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


class StageMeter:
    def __init__(self, cuda=None):
        self.cuda = cuda
        self.rows = []
        self.ledger = []
        self.sample_index = -1
        self.valid_samples = 1
        self._depth = 0
        self._restore = []

    @contextmanager
    def scope(self, name, family="neural", charged=True, gpu=True):
        # Outer NN subgraphs own the interval. Children are not summed again.
        if self._depth:
            yield
            return
        self._depth += 1
        try:
            start_event = end_event = None
            if gpu and self.cuda is not None:
                start_event = self.cuda.Event(enable_timing=True)
                end_event = self.cuda.Event(enable_timing=True)
                self.cuda.synchronize()
            start = time.perf_counter()
            if start_event is not None:
                start_event.record()
            try:
                yield
            finally:
                if end_event is not None:
                    end_event.record()
                    end_event.synchronize()
                end = time.perf_counter()
                self.rows.append({
                    "sample_index": self.sample_index,
                    "valid_samples": self.valid_samples,
                    "stage": name,
                    "family": family,
                    "charged": charged,
                    "start_monotonic": start,
                    "end_monotonic": end,
                    "synchronized_wall_ms": (end - start) * 1000,
                    "cuda_event_ms": None if start_event is None else float(start_event.elapsed_time(end_event)),
                })
        finally:
            self._depth -= 1

    def wrap(self, owner, method, name, *, family="neural", charged=True, gpu=True):
        original = getattr(owner, method)

        @functools.wraps(original)
        def measured(*args, **kwargs):
            with self.scope(name, family, charged, gpu):
                return original(*args, **kwargs)

        setattr(owner, method, measured)
        self._restore.append((owner, method, original))
        self.ledger.append(source_record(original, name, family, charged))

    def patch_statements(self, owner, method, groups):
        """groups=(stage,family,charged,first-index,last-index), inclusive.

        All statements are preserved. Groups refer to ORIGINAL top-level AST
        positions, and are validated to be non-overlapping. This allows a pad
        or field scatter to stay outside a surrounding learned encoding span.
        """
        original = getattr(owner, method)
        function = original.__func__
        if function.__code__.co_freevars:
            raise ValueError("Closure methods require an explicit non-AST adapter")
        source, first_line = inspect.getsourcelines(function)
        parsed = ast.parse(textwrap.dedent("".join(source)))
        node = next(n for n in parsed.body if isinstance(n, ast.FunctionDef))
        body = node.body
        occupied = set()
        starts = {}
        for name, family, charged, first, last in groups:
            if not 0 <= first <= last < len(body):
                raise ValueError(f"Invalid original AST span {name}: {first}..{last}/{len(body)}")
            indices = set(range(first, last + 1))
            if occupied & indices:
                raise ValueError("Neural spans overlap")
            occupied |= indices
            starts[first] = (name, family, charged, last)
            record = source_record(function, name, family, charged)
            record.update({
                "first_source_line": first_line + body[first].lineno - 1,
                "last_source_line": first_line + body[last].end_lineno - 1,
                "original_statements": [ast.unparse(n) for n in body[first:last + 1]],
            })
            self.ledger.append(record)
        new_body, index = [], 0
        while index < len(body):
            if index not in starts:
                new_body.append(body[index])
                index += 1
                continue
            name, family, charged, last = starts[index]
            call = ast.Call(func=ast.Attribute(value=ast.Name(id="_lightgen_measurement_meter", ctx=ast.Load()), attr="scope", ctx=ast.Load()),
                            args=[ast.Constant(name), ast.Constant(family), ast.Constant(charged)], keywords=[])
            new_body.append(ast.With(items=[ast.withitem(context_expr=call)], body=body[index:last + 1]))
            index = last + 1
        node.body = new_body
        node.decorator_list = []
        # Default objects and annotations are copied from the original function.
        node.returns = None
        for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
            arg.annotation = None
        node.args.defaults = []
        node.args.kw_defaults = [None] * len(node.args.kwonlyargs)
        namespace = dict(function.__globals__)
        namespace["_lightgen_measurement_meter"] = self
        ast.fix_missing_locations(parsed)
        exec(compile(parsed, function.__code__.co_filename, "exec"), namespace)
        patched = namespace[function.__name__]
        patched.__defaults__, patched.__kwdefaults__ = function.__defaults__, function.__kwdefaults__
        functools.update_wrapper(patched, function)
        setattr(owner, method, types.MethodType(patched, owner))
        self._restore.append((owner, method, original))

    def close(self):
        for owner, method, original in reversed(self._restore):
            setattr(owner, method, original)
        self._restore.clear()


def source_record(function, stage, family, charged):
    function = getattr(function, "__func__", function)
    path = inspect.getsourcefile(function)
    lines, first = inspect.getsourcelines(function)
    return {"stage": stage, "family": family, "charged": charged,
            "method": function.__qualname__, "source_file": path,
            "source_sha256": file_sha256(path), "first_source_line": first,
            "last_source_line": first + len(lines) - 1}


def statements(owner, method):
    """Read-only helper for explicit, reviewable per-task span selection."""
    function = getattr(getattr(owner, method), "__func__", getattr(owner, method))
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    return [ast.unparse(n) for n in node.body]


def assign_index(items, name):
    candidates = [i for i, item in enumerate(items) if item.startswith(name + " = ")]
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one assignment to {name}, got {candidates}")
    return candidates[0]


def optical_energy_j(optical_seconds, electronic_seconds):
    if min(optical_seconds, electronic_seconds) < 0:
        raise ValueError("Negative accounted interval")
    return 121.746 * optical_seconds + 104.230 * electronic_seconds


def equivalent_cost(total_seconds, total_joules, valid_samples, frames_per_sample=1):
    if min(valid_samples, frames_per_sample) < 1:
        raise ValueError("Positive work denominator required")
    return {"sample_ms": total_seconds * 1000 / valid_samples,
            "sample_j": total_joules / valid_samples,
            "equivalent_frame_ms": total_seconds * 1000 / (valid_samples * frames_per_sample),
            "equivalent_frame_j": total_joules / (valid_samples * frames_per_sample)}
