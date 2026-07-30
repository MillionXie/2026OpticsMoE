from __future__ import annotations

import copy
import importlib
import math
import random
from collections import deque
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image

from .datasets_bench2drive import normalized_driving_state
from .io_utils import append_csv, atomic_torch_save, torch_load, write_json
from .modeling import (
    OpticalDrivingPolicy,
    TwinQCritic,
    decode_normalized_action,
    preprocess_vision,
)
from .objectives import shaped_reward


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.values: deque[tuple[Any, ...]] = deque(maxlen=int(capacity))

    def add(
        self,
        observation: Any,
        action: np.ndarray,
        reward: float,
        next_observation: Any,
        done: bool,
    ) -> None:
        self.values.append(
            (
                _compact_copy(observation),
                np.asarray(action, dtype=np.float32),
                float(reward),
                _compact_copy(next_observation),
                bool(done),
            )
        )

    def sample(self, count: int) -> tuple[list[Any], ...]:
        rows = random.sample(self.values, count)
        observations, actions, rewards, next_observations, dones = zip(*rows)
        return (
            list(observations),
            list(actions),
            list(rewards),
            list(next_observations),
            list(dones),
        )

    def __len__(self) -> int:
        return len(self.values)


class StateBuilder:
    def __init__(
        self,
        policy: OpticalDrivingPolicy,
        processor: Any,
        settings: Any,
        device: torch.device,
    ) -> None:
        self.policy = policy
        self.processor = processor
        self.settings = settings
        self.device = device

    def one(self, observation: dict[str, Any], *, grad: bool) -> torch.Tensor:
        return self.batch([observation], grad=grad)

    def batch(
        self, observations: list[dict[str, Any]], *, grad: bool
    ) -> torch.Tensor:
        if all("state" in observation for observation in observations):
            return torch.as_tensor(
                np.stack([observation["state"] for observation in observations]),
                dtype=torch.float32,
                device=self.device,
            )
        if all("visual_feature" in observation for observation in observations):
            visual = torch.as_tensor(
                np.stack(
                    [
                        np.asarray(observation["visual_feature"], dtype=np.float32)
                        for observation in observations
                    ]
                ),
                device=self.device,
            )
        else:
            images = [_observation_image(observation) for observation in observations]
            inputs = preprocess_vision(self.processor, images, self.device)
            with torch.set_grad_enabled(grad):
                visual = self.policy.encode(
                    inputs["pixel_values"], inputs["image_grid_thw"]
                )
        speed = torch.tensor(
            [float(observation["speed"]) for observation in observations],
            device=self.device,
        )
        command = torch.tensor(
            [int(observation["command"]) for observation in observations],
            dtype=torch.long,
            device=self.device,
        )
        target = torch.as_tensor(
            np.asarray(
                [observation["target_point"] for observation in observations],
                dtype=np.float32,
            ),
            device=self.device,
        )
        conditioning = normalized_driving_state(
            speed,
            command,
            target,
            speed_scale=self.settings.speed_normalization_mps,
            target_clip=self.settings.target_point_clip_m,
            num_commands=self.settings.num_commands,
        )
        return torch.cat([visual.float(), conditioning], dim=-1)

    @torch.no_grad()
    def compact(self, observation: dict[str, Any]) -> dict[str, Any]:
        if self.settings.sac_store_images:
            return _compact_copy(observation)
        state = self.one(observation, grad=False).squeeze(0).cpu().numpy()
        return {"state": state.astype(np.float32)}


def train_sac(
    policy: OpticalDrivingPolicy,
    processor: Any,
    settings: Any,
    device: torch.device,
    *,
    env: Any | None = None,
    bc_checkpoint: Path | None = None,
) -> dict[str, Any]:
    if bc_checkpoint is None:
        bc_checkpoint = settings.output_dir / "checkpoints" / "bc_policy_best.pt"
    if bc_checkpoint.is_file():
        payload = torch_load(bc_checkpoint)
        policy.backbone.core.load_state_dict(
            payload["backbone"]["core_state_dict"]
        )
        policy.backbone.recombiner.load_state_dict(
            payload["backbone"]["recombiner_state_dict"]
        )
        policy.actor.load_state_dict(payload["actor_state_dict"])
    else:
        raise FileNotFoundError(
            f"SAC must initialize from a stable behavior-cloning policy: {bc_checkpoint}"
        )
    policy.actor.log_std.requires_grad_(True)
    environment = env if env is not None else make_environment(settings)
    state_builder = StateBuilder(policy, processor, settings, device)
    critic = TwinQCritic().to(device)
    target_critic = copy.deepcopy(critic).requires_grad_(False)
    critic_optimizer = torch.optim.Adam(
        [
            {"params": critic.parameters(), "lr": settings.sac_learning_rate},
            {
                "params": policy.backbone.recombiner.parameters(),
                "lr": settings.bc_linear_learning_rate,
            },
            {
                "params": policy.backbone.core.parameters(),
                "lr": settings.bc_optical_learning_rate,
            },
        ]
    )
    actor_optimizer = torch.optim.Adam(
        [
            {"params": policy.actor.parameters(), "lr": settings.sac_learning_rate},
            {
                "params": policy.backbone.recombiner.parameters(),
                "lr": settings.bc_linear_learning_rate,
            },
            {
                "params": policy.backbone.core.parameters(),
                "lr": settings.bc_optical_learning_rate,
            },
        ]
    )
    log_alpha = torch.tensor(
        math.log(settings.sac_initial_alpha),
        device=device,
        requires_grad=settings.sac_autotune_alpha,
    )
    alpha_optimizer = (
        torch.optim.Adam([log_alpha], lr=settings.sac_learning_rate)
        if settings.sac_autotune_alpha
        else None
    )
    target_entropy = -3.0
    replay = ReplayBuffer(settings.sac_replay_capacity)
    _set_backbone_trainability(policy, settings, step=0)
    observation, _ = _reset(environment, settings.random_seed)
    previous_action: np.ndarray | None = None
    episode_return = 0.0
    episode = 0
    history = settings.output_dir / "metrics" / "sac_training.csv"
    last_update: dict[str, float] = {}
    for step in range(1, settings.sac_total_steps + 1):
        _set_backbone_trainability(policy, settings, step)
        if step <= settings.sac_random_steps:
            normalized_action = np.random.uniform(-1.0, 1.0, size=3).astype(np.float32)
        else:
            with torch.no_grad():
                state = state_builder.one(observation, grad=False)
                normalized_action = (
                    policy.actor.sample(state)[0].squeeze(0).cpu().numpy()
                )
        control = (
            decode_normalized_action(torch.from_numpy(normalized_action))
            .cpu()
            .numpy()
        )
        next_observation, _native_reward, terminated, truncated, info = _step(
            environment, control
        )
        reward, reward_parts = shaped_reward(
            info, control, previous_action, settings
        )
        done = bool(terminated or truncated)
        replay.add(
            state_builder.compact(observation),
            normalized_action,
            reward,
            state_builder.compact(next_observation),
            done,
        )
        episode_return += reward
        previous_action = control
        observation = next_observation
        if len(replay) >= settings.sac_batch_size:
            last_update = _sac_update(
                policy,
                critic,
                target_critic,
                actor_optimizer,
                critic_optimizer,
                log_alpha,
                alpha_optimizer,
                target_entropy,
                replay,
                state_builder,
                settings,
                device,
                backbone_trainable=_backbone_trainable(policy),
            )
        if done:
            append_csv(
                history,
                {
                    "step": step,
                    "episode": episode,
                    "episode_return": episode_return,
                    "replay_size": len(replay),
                    **reward_parts,
                    **last_update,
                },
            )
            episode += 1
            observation, _ = _reset(environment, settings.random_seed + episode)
            previous_action = None
            episode_return = 0.0
        if step % max(1, settings.log_interval_batches) == 0:
            critic_text = (
                f"{last_update['critic_loss']:.4f}"
                if "critic_loss" in last_update
                else "n/a"
            )
            print(
                f"[sac] step={step}/{settings.sac_total_steps} "
                f"replay={len(replay)} alpha={float(log_alpha.exp()):.4f} "
                f"critic={critic_text}",
                flush=True,
            )
    payload = {
        "step": settings.sac_total_steps,
        "backbone": policy.backbone.checkpoint_state(),
        "actor_state_dict": policy.actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "target_critic_state_dict": target_critic.state_dict(),
        "log_alpha": log_alpha.detach().cpu(),
        "settings": {
            "freeze_backbone_steps": settings.sac_freeze_backbone_steps,
            "unfreeze_linear_step": settings.sac_unfreeze_linear_step,
            "unfreeze_phase_step": settings.sac_unfreeze_phase_step,
        },
    }
    path = settings.output_dir / "checkpoints" / "sac_policy_last.pt"
    atomic_torch_save(path, payload)
    result = {
        "steps": settings.sac_total_steps,
        "episodes": episode,
        "checkpoint": str(path),
        "initialized_from_bc": str(bc_checkpoint),
        "last_update": last_update,
    }
    write_json(settings.output_dir / "metrics" / "sac_summary.json", result)
    if hasattr(environment, "close"):
        environment.close()
    return result


def make_environment(settings: Any) -> Any:
    if not settings.sac_env_factory:
        raise RuntimeError(
            "sac.env_factory is empty. Install CARLA 0.9.15 and Bench2Drive, "
            "then configure 'package.module:function'. The factory must return a "
            "Gymnasium-style environment with rgb_front/speed/command/target_point "
            "observations and the documented reward signals in info."
        )
    module_name, separator, function_name = settings.sac_env_factory.partition(":")
    if not separator:
        raise ValueError("sac.env_factory must use 'package.module:function' syntax")
    factory: Callable[..., Any] = getattr(
        importlib.import_module(module_name), function_name
    )
    return factory(settings)


def _sac_update(
    policy: OpticalDrivingPolicy,
    critic: TwinQCritic,
    target_critic: TwinQCritic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    log_alpha: torch.Tensor,
    alpha_optimizer: torch.optim.Optimizer | None,
    target_entropy: float,
    replay: ReplayBuffer,
    state_builder: StateBuilder,
    settings: Any,
    device: torch.device,
    *,
    backbone_trainable: bool,
) -> dict[str, float]:
    observations, actions, rewards, next_observations, dones = replay.sample(
        settings.sac_batch_size
    )
    states = state_builder.batch(observations, grad=backbone_trainable)
    with torch.no_grad():
        next_states = state_builder.batch(next_observations, grad=False)
        next_action, next_log_prob, _ = policy.actor.sample(next_states)
        target_q1, target_q2 = target_critic(next_states, next_action)
        alpha = log_alpha.exp()
        target = torch.tensor(rewards, device=device).unsqueeze(-1) + (
            1.0 - torch.tensor(dones, device=device, dtype=torch.float32).unsqueeze(-1)
        ) * settings.sac_gamma * (
            torch.minimum(target_q1, target_q2) - alpha * next_log_prob
        )
    action_tensor = torch.tensor(np.stack(actions), device=device)
    q1, q2 = critic(states, action_tensor)
    critic_loss = torch.nn.functional.mse_loss(q1, target) + torch.nn.functional.mse_loss(q2, target)
    critic_optimizer.zero_grad(set_to_none=True)
    critic_loss.backward()
    critic_optimizer.step()

    actor_states = state_builder.batch(observations, grad=backbone_trainable)
    sampled_action, log_prob, _ = policy.actor.sample(actor_states)
    actor_q1, actor_q2 = critic(actor_states, sampled_action)
    actor_loss = (log_alpha.exp().detach() * log_prob - torch.minimum(actor_q1, actor_q2)).mean()
    actor_optimizer.zero_grad(set_to_none=True)
    actor_loss.backward()
    actor_optimizer.step()
    alpha_loss_value = 0.0
    if alpha_optimizer is not None:
        alpha_loss = -(log_alpha * (log_prob.detach() + target_entropy)).mean()
        alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        alpha_optimizer.step()
        alpha_loss_value = float(alpha_loss.detach())
    with torch.no_grad():
        for target_parameter, parameter in zip(
            target_critic.parameters(), critic.parameters()
        ):
            target_parameter.lerp_(parameter, settings.sac_tau)
    return {
        "critic_loss": float(critic_loss.detach()),
        "actor_loss": float(actor_loss.detach()),
        "alpha_loss": alpha_loss_value,
        "alpha": float(log_alpha.exp().detach()),
    }


def _set_backbone_trainability(
    policy: OpticalDrivingPolicy, settings: Any, step: int
) -> None:
    policy.backbone.requires_grad_(False)
    if (
        settings.sac_unfreeze_linear_step is not None
        and step >= settings.sac_unfreeze_linear_step
    ):
        policy.backbone.recombiner.requires_grad_(True)
    if (
        settings.sac_unfreeze_phase_step is not None
        and step >= settings.sac_unfreeze_phase_step
    ):
        # Closed-loop fine-tuning is deliberately conservative: only the
        # physical phase masks are released. The input adapter, router and OEO
        # electronics remain frozen so SAC cannot silently redesign the whole
        # pretrained encoder.
        policy.backbone.core.expert_layers.requires_grad_(True)
        policy.backbone.core.global_phase.requires_grad_(True)


def _backbone_trainable(policy: OpticalDrivingPolicy) -> bool:
    return any(parameter.requires_grad for parameter in policy.backbone.parameters())


def _observation_image(observation: dict[str, Any]) -> Image.Image:
    if "rgb_front" not in observation:
        raise RuntimeError(
            "Environment observation needs rgb_front, or a precomputed visual_feature"
        )
    value = observation["rgb_front"]
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[-1] not in {3, 4}:
        raise RuntimeError(f"rgb_front must be HWC RGB/BGRA, got {array.shape}")
    if array.shape[-1] == 4:
        array = array[..., :3][:, :, ::-1]
    return Image.fromarray(array.astype(np.uint8), mode="RGB")


def _compact_copy(observation: Any) -> Any:
    if isinstance(observation, dict):
        return {
            key: (
                np.asarray(value, dtype=np.uint8).copy()
                if key == "rgb_front" and not isinstance(value, Image.Image)
                else np.asarray(value).copy()
                if isinstance(value, np.ndarray)
                else value.copy()
                if isinstance(value, Image.Image)
                else copy.deepcopy(value)
            )
            for key, value in observation.items()
        }
    return copy.deepcopy(observation)


def _reset(env: Any, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    value = env.reset(seed=seed)
    return value if isinstance(value, tuple) else (value, {})


def _step(env: Any, action: np.ndarray) -> tuple[Any, float, bool, bool, dict[str, Any]]:
    value = env.step(action)
    if len(value) == 5:
        return value
    if len(value) == 4:
        observation, reward, done, info = value
        return observation, reward, bool(done), False, info
    raise RuntimeError("Environment step must return Gymnasium 5-tuple or Gym 4-tuple")
