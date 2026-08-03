"""Python 3.11 client for the out-of-process Python 3.8 CARLA service."""

from __future__ import annotations

import time
from multiprocessing.connection import Client
from typing import Any

import numpy as np


class RemoteCarlaEnv:
    """Small Gymnasium-compatible proxy; this process never imports CARLA."""

    def __init__(self, settings: Any) -> None:
        self.address = (settings.carla_bridge_host, settings.carla_bridge_port)
        self.authkey = settings.carla_bridge_authkey.encode("utf-8")
        deadline = time.monotonic() + settings.carla_bridge_timeout_seconds
        last_error: BaseException | None = None
        while time.monotonic() < deadline:
            try:
                self.connection = Client(self.address, authkey=self.authkey)
                break
            except (ConnectionError, OSError) as exc:
                last_error = exc
                time.sleep(1.0)
        else:
            raise RuntimeError(
                f"Could not connect to Python-3.8 CARLA bridge at {self.address} "
                f"within {settings.carla_bridge_timeout_seconds}s: {last_error}. "
                "Start start_carla_bridge_rfl.sh before --phase sac_train."
            )
        hello = self._call({"op": "hello"})
        if hello.get("protocol_version") != 1:
            raise RuntimeError(f"Unsupported CARLA bridge handshake: {hello}")

    def reset(self, *, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        result = self._call({"op": "reset", "seed": seed})
        return _normalize_observation(result["observation"]), result.get("info", {})

    def step(
        self, action: np.ndarray | list[float]
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        values = np.asarray(action, dtype=np.float32)
        if values.shape != (3,) or not np.isfinite(values).all():
            raise ValueError(f"CARLA control must be a finite [3] vector, got {values}")
        result = self._call({"op": "step", "action": values.tolist()})
        return (
            _normalize_observation(result["observation"]),
            float(result["reward"]),
            bool(result["terminated"]),
            bool(result["truncated"]),
            dict(result["info"]),
        )

    def close(self) -> None:
        connection = getattr(self, "connection", None)
        if connection is None:
            return
        try:
            self._call({"op": "close"})
        except (EOFError, OSError, RuntimeError):
            pass
        finally:
            connection.close()
            self.connection = None

    def _call(self, request: dict[str, Any]) -> dict[str, Any]:
        self.connection.send(request)
        try:
            response = self.connection.recv()
        except EOFError as exc:
            raise RuntimeError("CARLA bridge closed the connection unexpectedly") from exc
        if not isinstance(response, dict):
            raise RuntimeError(f"Invalid CARLA bridge response type {type(response)!r}")
        if not response.get("ok", False):
            raise RuntimeError(
                "CARLA bridge operation failed:\n" + str(response.get("error", response))
            )
        return response.get("result", {})


def _normalize_observation(value: dict[str, Any]) -> dict[str, Any]:
    observation = dict(value)
    image = np.asarray(observation["rgb_front"], dtype=np.uint8)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise RuntimeError(f"Remote rgb_front must be HWC RGB, got {image.shape}")
    observation["rgb_front"] = np.ascontiguousarray(image)
    observation["speed"] = float(observation["speed"])
    observation["command"] = int(observation["command"])
    observation["target_point"] = np.asarray(
        observation["target_point"], dtype=np.float32
    )
    return observation


def make_remote_carla_env(settings: Any) -> RemoteCarlaEnv:
    return RemoteCarlaEnv(settings)

