"""Flickr30k image-text matching with Qwen3-VL-2B and homogeneous optical MoE stacks."""

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
DATASET_REPO_ID = "nlphuji/flickr30k"
PROMPT_TEMPLATE = (
    "Determine whether the caption accurately describes the image.\n"
    "Caption: {caption}\n"
    "Match score:"
)
CLASS_NAMES = ("not_match", "match")
