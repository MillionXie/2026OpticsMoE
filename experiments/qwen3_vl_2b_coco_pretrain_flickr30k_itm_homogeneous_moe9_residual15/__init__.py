"""COCO generic distillation followed by Flickr30k optical-MoE fine-tuning."""

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
DATASET_REPO_ID = "nlphuji/flickr30k"
GENERIC_DATASET_REPO_ID = "HuggingFaceM4/COCO"
PROMPT_TEMPLATE = (
    "Determine whether the caption accurately describes the image.\n"
    "Caption: {caption}\n"
    "Match score:"
)
GENERIC_PROMPT_TEMPLATE = (
    "Represent the image and caption for general multimodal understanding.\n"
    "Caption: {caption}\n"
    "Representation:"
)
CLASS_NAMES = ("not_match", "match")
