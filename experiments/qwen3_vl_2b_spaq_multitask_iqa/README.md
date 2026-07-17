# Qwen3-VL-2B SPAQ Multitask IQA

This experiment uses a frozen `Qwen/Qwen3-VL-2B-Instruct` backbone for text-conditioned multitask image-quality regression on SPAQ. It runs the original Qwen chat template, processor, vision encoder, multimodal injection, and language decoder. It does not generate text. The 2048-dimensional final-language-layer hidden state at the last valid prompt token is cached and passed to one shared regression head.

The four tasks are `MOS`, `Brightness`, `Colorfulness`, and `Contrast`. Every original image is paired virtually with four fixed English prompts; image files are not copied. The shared head is:

```text
Linear(2048, 64) -> GELU -> Dropout(0.1) -> Linear(64, 1) -> Sigmoid
```

All targets are required to be finite scores in `[0,100]` and are divided by 100 during training. The loss is `SmoothL1Loss(beta=0.1)`. All Qwen parameters are frozen and Qwen remains in eval mode. Each image-prompt feature is cached, so later head epochs do not rerun Qwen.

## Split and evaluation protocol

The split unit is the original image, not the expanded image-task pair. With seed 42, 90% of original images enter train and 10% enter test. The exact image lists are persisted in `data_split.json`; all four tasks of an image always remain together. There is no validation set.

Training always runs the configured fixed number of epochs. It does not inspect the test set, perform early stopping, or select a checkpoint using test metrics. Only the final epoch checkpoint is saved as `checkpoints/final_regression_head.pt`. Test evaluation happens after training and reports per-task plus macro-average MAE (on the original 0–100 scale), SRCC, and PLCC.

## SPAQ data layout and automatic discovery

`download=true` is the default. If `data_root` does not already contain both annotations and images, `prepare_data`/`all` downloads the public `spaq.tgz` mirror from the configured Hugging Face dataset repository, resumes interrupted downloads, safely extracts it, and then validates the contents. The archive is about 34.8 GB, so the first run needs substantial disk space and time. `download_repo_id`, `download_filename`, and `download_endpoint` are configurable. The supplied config uses `https://hf-mirror.com` because the target server cannot resolve `huggingface.co`; set the endpoint to `null` to use the official Hub endpoint. A Google Drive file/folder can alternatively be selected with `download_source="google_drive"` and `download_url`.

The authors' official repository currently documents Baidu and Google Drive distribution links. The default Hugging Face mirror is used because it provides a resumable non-interactive Linux download. Set `download=false` when you intentionally manage the dataset yourself.

A common layout is:

```text
data/SPAQ/
  MOS and Image attribute scores.xlsx
  TestImage/
    00001.jpg
    ...
```

After download/extraction, the loader recursively inspects `.csv`, `.xlsx`, and `.xls` files and only accepts a table with one unambiguous image-name column plus all four exact score concepts. This recursive step only handles archive wrapper directories and common layouts such as `SPAQ/Annotations` plus `SPAQ/TestImage`; it is not a replacement for downloading the data. You can set `annotations_file` and `image_dir` explicitly when discovery is ambiguous. Relative values for these two fields are resolved under `data_root`.

The accepted image column aliases are `image_name`, `image`, `filename`, `file_name`, `img`, `image_path`, and `dist_img`. Score columns must unambiguously match MOS, Brightness, Colorfulness, and Contrast. If discovery fails, the error lists the annotation files and columns it found; it never silently substitutes another label.

## Full multimodal feature path

```text
SPAQ image + task-specific prompt
 -> Qwen chat template with image placeholder
 -> original AutoProcessor (image + text)
 -> complete frozen Qwen3-VL vision-language forward
 -> final language hidden states
 -> last non-padding prompt position [2048]
 -> frozen feature cache
 -> shared regression head
```

Changing the data split, labels, prompts, model ID, processor pixel budget, dtype, attention implementation, or feature dimension invalidates cache metadata and produces an explicit error rather than silently reusing stale features.

## Outputs

The run directory contains:

- `resolved_config.json`, `environment.json`, `dataset.json`, and `data_split.json`
- `features/train.pt` and `features/test.pt`
- `metrics/feature_extraction_train.json` and `metrics/feature_extraction_test.json`
- `training_history.csv`
- `checkpoints/final_regression_head.pt`
- `test_predictions.csv` and `test_metrics.json`
- `figures/training_loss.png`
- `figures/scatter_mos.png`, `scatter_brightness.png`, `scatter_colorfulness.png`, and `scatter_contrast.png`

SPAQ is described by its authors as containing 11,125 smartphone photographs with MOS and image-attribute annotations. This experiment uses only the four configured targets and validates the actual local annotation columns before running.
