from typing import Dict, List, Sequence

import torch.nn as nn


class MultitaskProgressiveUnfreezingSchedule:
    """Progressive schedule for the separate multitask four-expert model."""

    def __init__(
        self,
        num_layers: int,
        enabled: bool = True,
        order: str = "backward",
        stage_epochs: Sequence[int] = (3, 3, 3, 3, 3, 10),
        train_task_prompts_always: bool = True,
        train_global_fc_always: bool = True,
    ) -> None:
        if order not in {"backward", "forward"}:
            raise ValueError("order must be backward or forward.")
        self.num_layers = int(num_layers)
        self.enabled = bool(enabled)
        self.order = order
        self.stage_epochs = [int(value) for value in stage_epochs]
        self.train_task_prompts_always = bool(train_task_prompts_always)
        self.train_global_fc_always = bool(train_global_fc_always)
        if self.enabled and len(self.stage_epochs) != self.num_layers + 1:
            raise ValueError(
                f"stage_epochs must contain {self.num_layers + 1} values."
            )

    @property
    def total_epochs(self) -> int:
        return sum(self.stage_epochs)

    def stage_for_epoch(self, epoch: int) -> int:
        if not self.enabled:
            return 0
        cumulative = 0
        for stage_idx, length in enumerate(self.stage_epochs):
            cumulative += length
            if epoch <= cumulative:
                return stage_idx
        return len(self.stage_epochs) - 1

    def active_layer_indices(self, stage_idx: int) -> List[int]:
        if not self.enabled:
            return list(range(self.num_layers))
        count = max(0, min(int(stage_idx), self.num_layers))
        if self.order == "backward":
            return list(range(self.num_layers - count, self.num_layers))
        return list(range(count))

    @staticmethod
    def _set_module(module: nn.Module, value: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad = bool(value)

    def apply(self, model: nn.Module, stage_idx: int) -> Dict:
        if not self.enabled:
            for parameter in model.parameters():
                parameter.requires_grad = True
            active_layers = list(range(self.num_layers))
        else:
            for layer in model.expert_layers:
                self._set_module(layer, False)
            active_layers = self.active_layer_indices(stage_idx)
            for index in active_layers:
                self._set_module(model.expert_layers[index], True)

            self._set_module(model.task_prompt_bank, False)
            prompt_enabled = bool(stage_idx == 0 or self.train_task_prompts_always)
            for parameter in model.task_prompt_bank.amplitude_logits.values():
                parameter.requires_grad = prompt_enabled
            for parameter in model.task_prompt_bank.phase_biases.values():
                parameter.requires_grad = bool(
                    stage_idx > 0 and self.train_task_prompts_always
                )
            self._set_module(
                model.global_fc,
                bool(stage_idx == 0 or self.train_global_fc_always),
            )
            if hasattr(model, "task_readouts"):
                self._set_module(model.task_readouts, True)
            else:
                self._set_module(model.readout, True)

        names = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        return {
            "stage_idx": int(stage_idx),
            "order": self.order,
            "active_layer_indices_zero_based": active_layers,
            "active_layers": [index + 1 for index in active_layers],
            "trainable_parameter_names": names,
            "trainable_parameter_count": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
        }
