"""KADID-10k DMOS distillation with a 4x4 electronic-router optical MoE."""

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
SUPPORTED_TASKS = ("DMOS",)
QUALITY_SCORE_MIN = 1.0
QUALITY_SCORE_MAX = 5.0

TASK_PROMPTS = {
    "DMOS": (
        "Predict the human-rated perceptual quality of this artificially "
        "distorted image on a 1-5 scale. Score:"
    ),
}
