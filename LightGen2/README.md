# LightGen2 temporal video quality inference

LightGen2 predicts a mean opinion score (MOS) for each video. The numerical
inference graph combines a frozen Qwen-derived visual frontend, electronic
routers and linear experts, vision and language optical propagation, and a
two-layer quality readout. Each optical field contains 16 videos.

The editable user guide is supplied as `README.docx` in the distribution.

## Installation

Use Python 3.11 and install the dependencies from this directory.

```bash
python -m pip install -r requirements.txt
```

PyTorch 2.6 or later supports the checkpoint loader. For GPU inference, install
a CUDA-enabled PyTorch build compatible with your driver. Figure export uses
Arial, with 7 pt labels and 8 pt titles.

## Sixteen video demo

The distribution includes 16 distinct synthetic clips in `demo_videos`.
One forward processes their 64 sampled frames in a shared optical field.

```bash
python demo.py --device cpu
```

For a CUDA device, use the following command.

```bash
python demo.py --device cuda
```

The demo writes predictions, complete phase masks, and input/output optical
fields to `outputs/demo`. Numerical arrays use NPZ and figures use PNG.
The bundled `results` directory contains reference outputs from the same demo.
The synthetic clips provide a reproducible execution example. Their MOS values
are model predictions without human-rated reference labels.

## Your own videos

```bash
python infer.py --video-dir path/to/videos --device cuda --output outputs/predictions.json
```

Videos in a directory are sorted by filename. Supply an explicit list to control
their order.

```bash
python infer.py --videos video_a.mp4 video_b.mp4 --device cpu --output outputs/predictions.json
```

Inputs are grouped into 16-video fields in order. The last field is padded
cyclically from the beginning of the supplied list. Predictions are returned
only for the supplied videos. Optical propagation couples the field, so keep
video order, grouping and padding fixed when comparing results.

Each video is sampled at four positions from 10% to 90% of its frame range.
Each frame is center-cropped to 65% of its shorter side and resized to 448 × 448.
The model returns one MOS per video. Higher values indicate better predicted
temporal quality. The bundled checkpoint includes the prompt and grade-score
tensors needed for this task.

## Full evaluation

The supplied `test_videos.csv` records the order, relative video paths and
temporal MOS labels of the 558-video evaluation. Obtain the original videos
from the dataset authors and point `--video-root` to the directory containing
their generator subdirectories. [LGVQ dataset](https://github.com/zczhang-sjtu/UGVQ)

```bash
python infer.py --evaluation-csv test_videos.csv --video-root path/to/LGVQ/videos --device cpu --output outputs/test.json
```

The same forward processes 35 fields of 16 videos. The final field contains 14
actual videos and two cyclic padding entries. The JSON contains 558 predictions
and the overall SRCC calculated against their temporal MOS labels. The reference
pixel-input result is approximately 0.7992. The evaluation split was used in
checkpoint selection. Keep the CSV order to reproduce this field composition.

## Vision and language branches

The names in the code follow the two branches.

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
Sixteen videos occupy a 4 × 4 arrangement. Within each vision slot, the four
frames occupy a 2 × 2 arrangement. The language branch uses one patch per video.

`LightGen2.forward(pixels)` accepts uint8 RGB data of shape
`[batch, 16, 4, 3, 448, 448]`. The separate
`predict_from_embeddings(visual_embeddings)` method accepts saved frontend
outputs of shape `[batch, 16, 4, 49, 1024]`.

## Optical arrays and figures

To export one 16-video field from a directory, run the following command.

```bash
python visualize.py --video-dir path/to/16_videos --device cuda --output outputs/fields
```

| Files in each branch | Contents |
| --- | --- |
| `vision_mask.npz`, `language_mask.npz` | Complete phase map, all 16 phase tiles and actual complex modulation |
| `vision_input_field.npz`, `language_input_field.npz` | Incident complex field, amplitude, intensity and modulated input to propagation |
| `vision_output_field.npz`, `language_output_field.npz` | Propagated complex field, detected intensity and propagation transfer function |
| Same stems with `.png` | Matching full-canvas visualizations |
| `input_frames.png` | The four sampled frames from all 16 videos |
| `optical_expert_share.npz`, `optical_expert_share.png` | Top2 selections and shares for this demonstration batch |
| `predictions.json` | Video filename, field index, zero-based slot and MOS |

Complex fields are stored as `complex64`. Phase and intensity arrays are
`float32`. NPZ files load with `numpy.load(path, allow_pickle=False)`.
Phase plots use a linear scale from −π to π. Input plots show field amplitude
and output plots show detected intensity. Both use a linear color scale from
zero to the maximum value in each field.

## Files

| Path | Purpose |
| --- | --- |
| `model.py` | Network layers, wave propagation and checkpoint loader |
| `infer.py` | Video decoding, input grouping and prediction |
| `demo.py` | Sixteen-video demonstration and synthetic clip generation |
| `visualize.py` | Actual-forward field capture and NPZ/PNG export |
| `weights/LightGen2.pt` | Standalone weights, configuration and frozen frontend tensors |
| `demo_videos` | Sixteen bundled synthetic input videos |
| `test_videos.csv` | Ordered 558-video evaluation list and temporal MOS labels |
| `results` | Reference predictions, optical arrays and figures |
| `README.docx` | Editable installation, architecture and file-format guide |
| `requirements.txt` | Inference and visualization dependencies |
| `LICENSE`, `NOTICE`, `licenses` | Apache-2.0 terms and Qwen attribution |

## License

The source, LightGen2 weights and generated demo clips are distributed under
Apache-2.0. The frozen Qwen components retain their upstream attribution.
See `NOTICE` and `licenses/QWEN_LICENSE.txt`.

The upstream visual components originate from Qwen3-VL-2B-Instruct.
[Qwen model](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct),
[Qwen source](https://github.com/QwenLM/Qwen3-VL).
