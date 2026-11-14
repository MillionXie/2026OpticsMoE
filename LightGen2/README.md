# LightGen2 temporal video quality inference

LightGen2 predicts a mean opinion score (MOS) for each video. The inference
graph combines a frozen Qwen-derived visual frontend, electronic routers and
linear experts, vision and language optical propagation, and a two-layer quality
readout. Each coherent optical field contains 16 videos.

The editable detailed guide is supplied as `README.docx`.

## Installation

Use Python 3.11 and install the dependencies from this directory.

```bash
python -m pip install -r requirements.txt
```

PyTorch 2.6 or later supports the checkpoint loader. For GPU inference, install
a CUDA-enabled PyTorch build compatible with your driver. Figure export uses
Arial, with 7 pt labels and 8 pt titles.

## Run the complete evaluation

The input archive `data/test_rgb.npz` contains preprocessed RGB frames for all
558 original TEST videos, their ordered sample IDs and human temporal MOS labels.
Start the complete evaluation from this directory.

```bash
python infer.py
```

The command selects CUDA when available and otherwise uses CPU. Select a device
explicitly with `python infer.py --device cuda` or `python infer.py --device cpu`.
Predictions, metrics and figures are written to `results`.

The 558 distinct videos are processed in 35 ordered optical fields. Fields 1–34
contain 544 distinct videos. Field 35 contains the remaining 14 videos and two
padding entries copied from videos 1 and 2. SRCC uses exactly the 558 actual
video predictions and labels. The normalized-RGB reference SRCC is 0.798371.
The evaluation split was used in checkpoint selection.

Five complete runs from an extracted package had a median runtime of 71.533
seconds, ranging from 64.997 to 72.778 seconds, on a shared NVIDIA A100-PCIE-40GB.
One complete warm-up preceded five fresh Python processes. Timing covers imports,
model and NPZ loading, all 558 predictions, metrics and result/figure export,
with CUDA synchronized before completion. Outer ZIP extraction precedes timing.
The supplied results come from the median run. Each invocation saves its own
`total_runtime_seconds` in `02_metrics.json`. Runtime varies with shared load,
hardware and storage performance.

The same run captures field 1 while computing its predictions and exports its
input frames, complete masks, and optical input/output fields. Choose another
field with the following command. Field numbers range from 1 to 35.

```bash
python infer.py --visualize-field 12
```

Every invocation evaluates all samples in the input NPZ. The selected field's slots
can be looked up by field and slot in `01_predictions.csv`. Optical NPZ files
also record the selected field's video IDs and padding flags.
Expert-share statistics cover all 558 actual videos. Whole-canvas coherent
propagation couples the slots, so preserve sample order and field composition.

Before packaging, each video contributes four RGB frames sampled at 10% to 90%
of its frame range. Frames are center-cropped to 65% of their shorter side,
resized to 448 × 448, and normalized by `(RGB / 255 - 0.5) / 0.5`.
Frames are normalized on CPU and rounded to the frozen patch kernel's bfloat16
precision. Float16 storage preserves these prepared values exactly.
Patch projection, position addition, fixed pooling and the remaining network
are executed by the inference code. Higher MOS indicates better predicted quality.

Select another compatible normalized RGB archive or result directory with
the following arguments.

```bash
python infer.py --input path/to/test_rgb.npz --output path/to/results
```

## Vision and language branches

| Component | Processing |
| --- | --- |
| `vision_frontend` | Saved Qwen patch projection and position tensors, followed by fixed mean pooling |
| `vision_adapter` | 49 × 1024 visual embeddings to 49 × 192 per frame |
| `vision_router`, `vision_experts` | Top2 routing and four tokenwise linear experts |
| `vision_global` | 27 × 27 input patches, phase modulation and wave propagation |
| `vision_merger` | One 192-dimensional summary per frame |
| `language_adapter` | 38 × 2048 prompt embeddings to 38 × 192 |
| `language_router`, `language_experts` | Process four visual summaries plus 38 language tokens |
| `language_global` | 56 × 56 input patch per video, phase modulation and wave propagation |
| `readout` | 42 × 96 features to five grade probabilities and their MOS expectation |

Both optical branches use a 518 × 518 canvas with a 478 × 478 active area.
Sixteen videos occupy a 4 × 4 arrangement. Within each vision slot, four frames
occupy a 2 × 2 arrangement. Language uses one patch per video.

`LightGen2.forward(normalized_rgb)` accepts floating-point normalized RGB with shape
`[batch, 16, 4, 3, 448, 448]`. `predict_from_embeddings(visual_embeddings)`
accepts saved frontend outputs with shape `[batch, 16, 4, 49, 1024]`.

## Read the results in numbered order

| Result files | Contents |
| --- | --- |
| `01_predictions.csv` | All 558 video IDs, field/slot numbers, human MOS and predicted MOS |
| `02_metrics.json` | SRCC, video/field counts, complete program runtime and device |
| `03_prediction_vs_mos.png` | Predicted versus human MOS for all 558 videos |
| `04_input_frames.png` | Four sampled frames from each captured slot |
| `05_optical_expert_share.npz` and `.png` | Expert selections, probabilities and shares over 558 actual videos |
| `06_vision_mask.npz` and `.png` | Complete phase map, sixteen phase tiles and complex modulation |
| `07_vision_input_field.npz` and `.png` | Incident field, amplitude, intensity and modulated field |
| `08_vision_output_field.npz` and `.png` | Propagated field, intensity and propagation transfer |
| `09_language_mask.npz` and `.png` | Complete language phase map and modulation |
| `10_language_input_field.npz` and `.png` | Language incident field and modulated field |
| `11_language_output_field.npz` and `.png` | Language propagated field, intensity and transfer |

CSV field and slot numbers start at 1. Each optical NPZ includes its
`field_number`, sixteen `video_names` and a `valid` mask. Expert-share arrays
contain 558 sample IDs and exclude padding. Evaluation follows the original
NPZ sample order.

Complex fields use `complex64`. Phase and intensity use `float32`. Load NPZ
files with `numpy.load(path, allow_pickle=False)`. Phase plots use a linear
−π to π scale. Input amplitude is peak-normalized for display. Output detector
intensity is scaled linearly by its 99.8th percentile. The brightest 0.2% of
output pixels saturate in the displayed color scale. Cyan outlines mark the
478 × 478 active area.

Input NPZ files store `display_amplitude` and `display_scale`. Output NPZ files
store `display_intensity`, `display_scale` and `display_percentile`. Display
arrays use a linear [0, 1] scale. The `field`, `amplitude` and `intensity` keys
retain the original numerical arrays at 518 × 518 resolution.

## Files

| Path | Purpose |
| --- | --- |
| `model.py` | Network layers, optical propagation and checkpoint loader |
| `infer.py` | Complete NPZ evaluation entry point, grouping and metrics |
| `visualize.py` | Plotting and actual-forward field capture called by `infer.py` |
| `weights/LightGen2.pt` | Weights, configuration and frozen frontend tensors |
| `data/test_rgb.npz` | Normalized RGB for all 558 videos, sample IDs and temporal MOS labels |
| `results` | Full predictions, evaluation plots and selected-field NPZ/PNG files |
| `README.docx` | Editable installation, architecture and file-format guide |
| `requirements.txt` | Combined inference and visualization dependencies |
| `LICENSE`, `NOTICE`, `licenses` | Apache-2.0 terms and Qwen attribution |

## License

The source and LightGen2 weights are distributed under Apache-2.0. Qwen
components retain their upstream attribution. LGVQ videos and evaluation labels
retain their authors' attribution and applicable dataset terms. See `NOTICE`
and `licenses/QWEN_LICENSE.txt`. [LGVQ dataset](https://github.com/zczhang-sjtu/UGVQ)

Upstream components originate from Qwen3-VL-2B-Instruct.
[Qwen model](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct),
[Qwen source](https://github.com/QwenLM/Qwen3-VL).
