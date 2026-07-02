from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any, TypeVar


T = TypeVar("T")


def utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def log_event(stage: str, message: str, **fields: Any) -> None:
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    suffix = f" {details}" if details else ""
    print(f"[{utc_now_iso()}] [{stage}] {message}{suffix}", flush=True)


def progress_iter(
    iterable: Iterable[T],
    *,
    description: str,
    enabled: bool,
    total: int | None = None,
    unit: str = "batch",
) -> Iterable[T]:
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:
        log_event("progress", "tqdm is unavailable; continuing without a progress bar")
        return iterable
    return tqdm(
        iterable,
        desc=description,
        total=total,
        unit=unit,
        dynamic_ncols=True,
        mininterval=0.5,
        smoothing=0.1,
    )


def progress_total(iterable: object) -> int | None:
    try:
        return len(iterable)  # type: ignore[arg-type]
    except TypeError:
        return None
