# Codes for temporal video quality assessment with LightGen2

## Introduction

This is the code to assess temporal video quality with LightGen2 in Python using PyTorch. We provide pretrained weights and video frames from the LGVQ dataset to evaluate 558 videos, with 16 videos processed in parallel. The code predicts a mean opinion score (MOS) for each video and provides examples of the optical wave propagation for reference.

## Environment

We have tested the code with Python 3.11.15 and PyTorch 2.6.0 on an NVIDIA A100 GPU. For the required Python packages, please refer to requirements.txt.

## Installation guide

Please follow the PyTorch installation guide to install a build compatible with your system. Then install the dependent packages from this directory. [PyTorch installation guide](https://pytorch.org/get-started/locally/)

```bash
python -m pip install -r requirements.txt
```

## Files

| File | Description |
| --- | --- |
| `data/input_for_review.npz` | Video frames from 558 LGVQ videos, video IDs and human temporal MOS labels. |
| `weights/LightGen2.pt` | Model weights, including the frozen Qwen frontend and prompt features. |
| `model.py` | The network definition and optical propagation based on the angular spectrum method. |
| `infer.py` | The script to run the complete video quality assessment. |
| `visualize.py` | The functions to save predictions and visualize the input frames and optical fields. |
| `requirements.txt` | The required Python packages. |
| `README.md and README.docx` | Instructions for installing and running the code. The Word version is editable. |
| `LICENSE, NOTICE and licenses/` | The Apache-2.0 license and attribution for upstream components and data. |

The following files are generated in `results/`.

| File | Description |
| --- | --- |
| `01_predictions.csv` | The video ID, human MOS and predicted MOS for each of the 558 videos. |
| `02_metrics.json` | SRCC, number of videos, complete program runtime and computing device. |
| `03_prediction_vs_mos.png` | The predicted MOS compared with the human MOS for all videos. |
| `04_input_frames.png` | The input frames for the 16 videos used in the optical example. |
| `05_vision_input_field.npz and .png` | The input optical field of the vision branch. |
| `06_vision_mask.npz and .png` | The complete learned phase mask of the vision branch. |
| `07_vision_output_field.npz and .png` | The output optical field of the vision branch. |
| `08_language_input_field.npz and .png` | The input optical field of the language branch. |
| `09_language_mask.npz and .png` | The complete learned phase mask of the language branch. |
| `10_language_output_field.npz and .png` | The output optical field of the language branch. |

## How to execute

Run the following command in the LightGen2 directory.

```bash
python infer.py
```

The code first reads the video frames from data/input_for_review.npz and loads the model weights from weights/LightGen2.pt. It extracts visual features with the Qwen-derived frontend and processes them with routed experts and optical propagation in the vision branch. The language branch combines these features with the prompt features and applies its routed experts and optical propagation. The two-layer readout predicts one MOS for each video. Higher MOS indicates better temporal video quality.

Predictions and figures are saved in results/. For the supplied 558 videos, we obtained a Spearman rank correlation coefficient (SRCC) of 0.797065 between the predicted and human MOS. The median complete program runtime was 56.216 seconds over 5 runs on a shared NVIDIA A100-PCIE-40GB. Each execution records its own runtime in 02_metrics.json.

![Predicted and human temporal MOS](results/03_prediction_vs_mos.png)

The same execution saves the input video frames and captures the input field, learned mask and output field for one group of 16 videos. The examples below show the vision branch in the top row and the language branch in the bottom row, ordered from input to mask to output. Each optical result is saved as numerical data in an NPZ file and as a PNG image.

| Input | Mask | Output |
| --- | --- | --- |
| ![vision input field](results/05_vision_input_field.png) | ![vision mask](results/06_vision_mask.png) | ![vision output field](results/07_vision_output_field.png) |
| ![language input field](results/08_language_input_field.png) | ![language mask](results/09_language_mask.png) | ![language output field](results/10_language_output_field.png) |

The Qwen components and LGVQ data retain their upstream attribution. See `NOTICE`
and `licenses/QWEN_LICENSE.txt`. [Qwen](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct), [LGVQ](https://github.com/zczhang-sjtu/UGVQ)
