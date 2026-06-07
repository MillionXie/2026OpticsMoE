from typing import Dict, List, Sequence

import torch.nn as nn


class ProgressiveUnfreezingSchedule:
    """Configure trainable modules for six-stage optical layer unfreezing."""

    def __init__(
        self,
        num_layers: int,
        enabled: bool = True,
        order: str = "backward",
        stage_epochs: Sequence[int] = (3, 3, 3, 3, 3, 10),
        train_prompt_always: bool = True,
        train_global_fc_always: bool = True,
    ) -> None:
        if order not in {"backward", "forward"}:
            raise ValueError("Progressive order must be 'backward' or 'forward'.")
        self.num_layers = int(num_layers)
        self.enabled = bool(enabled)
        self.order = order
        self.stage_epochs = [int(value) for value in stage_epochs]
        self.train_prompt_always = bool(train_prompt_always)
        self.train_global_fc_always = bool(train_global_fc_always)
        expected_stages = self.num_layers + 1
        if self.enabled and len(self.stage_epochs) != expected_stages:
            raise ValueError(
                f"stage_epochs must contain {expected_stages} values for "
                f"{self.num_layers} expert layers."
            )
        if any(value <= 0 for value in self.stage_epochs):
            raise ValueError("Every progressive stage must last at least one epoch.")

    @property
    def total_epochs(self) -> int:
        return sum(self.stage_epochs)

    def stage_for_epoch(self, epoch: int) -> int:
        if not self.enabled:
            return 0
        if epoch <= 0:
            raise ValueError("epoch is one-based and must be positive.")
        cumulative = 0
        for stage_idx, stage_length in enumerate(self.stage_epochs):
            cumulative += stage_length
            if epoch <= cumulative:
                return stage_idx
        return len(self.stage_epochs) - 1

    def active_layer_indices(self, stage_idx: int) -> List[int]:
        if not self.enabled:
            return list(range(self.num_layers))
        count = max(0, min(int(stage_idx), self.num_layers))
        if count == 0:
            return []
        if self.order == "backward":
            return list(range(self.num_layers - count, self.num_layers))
        return list(range(count))

    @staticmethod
    def _set_module_trainable(module: nn.Module, trainable: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad = bool(trainable)

    def apply(self, model: nn.Module, stage_idx: int) -> Dict:
        if not self.enabled:
            for parameter in model.parameters():
                parameter.requires_grad = True
            active_layers = list(range(self.num_layers))
        else:
            for layer in model.expert_layers:
                self._set_module_trainable(layer, False)
            active_layers = self.active_layer_indices(stage_idx)
            for index in active_layers:
                self._set_module_trainable(model.expert_layers[index], True)

            global_fc_trainable = bool(
                stage_idx == 0 or self.train_global_fc_always
            )
            self._set_module_trainable(model.prompt, False)
            model.prompt.amplitude_logits.requires_grad = bool(
                stage_idx == 0 or self.train_prompt_always
            )
            # Stage 0 is deliberately amplitude-only. Optional phase biases
            # join from Stage 1 onward when prompt training remains enabled.
            if isinstance(model.prompt.phase_biases, nn.Parameter):
                model.prompt.phase_biases.requires_grad = bool(
                    stage_idx > 0 and self.train_prompt_always
                )
            self._set_module_trainable(model.global_fc, global_fc_trainable)
            # An electronic readout is a classifier head and remains trainable.
            self._set_module_trainable(model.readout, True)

        trainable_names = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        return {
            "stage_idx": int(stage_idx),
            "order": self.order,
            "active_layer_indices_zero_based": active_layers,
            "active_layers": [index + 1 for index in active_layers],
            "trainable_parameter_names": trainable_names,
            "trainable_parameter_count": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
        }
