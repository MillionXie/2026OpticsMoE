"""SPAQ single-attribute multimodal distillation with electronic amplitude routing."""

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
SUPPORTED_TASKS = ("MOS", "Brightness", "Colorfulness", "Contrast")

TASK_PROMPTS = {
    "MOS": "Predict the human-rated overall perceptual quality of this image on a 0-100 scale. Score:",
    "Brightness": "Predict the human-rated brightness and exposure quality of this image on a 0-100 scale. Score:",
    "Colorfulness": "Predict the human-rated colorfulness of this image on a 0-100 scale. Score:",
    "Contrast": "Predict the human-rated contrast of this image on a 0-100 scale. Score:",
}
