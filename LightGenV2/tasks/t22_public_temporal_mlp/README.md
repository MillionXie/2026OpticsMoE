# LightGen-II temporal video-quality reference

作者侧研究入口为本T22目录。Google Drive的独立推理与审阅入口为仓库根目录
[`LightGen2`](../../../LightGen2/README.md)，其中的
[`中文导读`](../../../LightGen2/中文导读.md)逐项解释文件和配置。
T22保留研究缓存、训练入口和核验材料。打包命令从根目录LightGen2读取推理源码。

LightGen-II predicts the temporal perceptual quality of a video. This compact
PyTorch reference combines a frozen Qwen-derived input front, electronic Top2
routing and linear experts, two simulated optical global stages, and a
five-grade electronic readout. Each video receives one predicted mean opinion
score (MOS).

The standalone checkpoint contains the extracted visual tensors, fixed-prompt
embedding, task layers, and learned phase masks. Inference runs offline once
the Python dependencies and release files are installed.

| Item | Selected reference |
|---|---|
| Task | Temporal video-quality prediction |
| Input | One or more video files, supplied in a fixed order |
| Output | One MOS per video, saved as JSON |
| Sampled frames | Four RGB frames per video, each 448 × 448 |
| Field organization | Sixteen videos per coherent simulated field |
| Registered model parameters | 3,777,350, with frozen frontend tensors stored as buffers |
| Final readout | 4032 → 32 → 5, with 129,221 parameters |
| Standalone checkpoint | Approximately 19.11 MiB |
| License | Apache-2.0, with Qwen attribution |

## Contents

- [Quick start](#quick-start)
- [Environment and installation](#environment-and-installation)
- [Input preparation](#input-preparation)
- [Outputs and score interpretation](#outputs-and-score-interpretation)
- [Model structure](#model-structure)
- [Reference performance](#reference-performance)
- [Optical-field visualization](#optical-field-visualization)
- [Release files and verification](#release-files-and-verification)
- [Troubleshooting](#troubleshooting)
- [License and acknowledgements](#license-and-acknowledgements)

## Quick start

Extract the release ZIP and open a terminal in its `lightgen-temporal` directory.
The commands below use paths relative to that directory.

```bash
python -m pip install -r requirements.txt
python infer.py --weights weights/LightGen.pt --videos examples/demo.mp4 --output outputs/demo.json
```

The supplied video is a deterministic synthetic example. The command prints
its prediction and creates `outputs/demo.json`. The verified example MOS is
approximately **47.5284**. The release also supplies its recorded output in
`examples/expected_predictions.json`. Floating-point and video-decoder
differences can produce small numerical changes.

To score your own videos, supply their paths in the intended order.

```bash
python infer.py --weights weights/LightGen.pt --videos video_a.mp4 video_b.mp4 --output outputs/mos.json
```

Append `--device cuda` to use the default CUDA GPU. A specific GPU can be
selected with `--device cuda:0`. CPU is the default device.

| Option | Usage |
|---|---|
| `--weights` | Required checkpoint path |
| `--videos` | Required ordered list of one or more video paths |
| `--device` | PyTorch device, default `cpu` |
| `--output` | Required destination JSON path |

## Environment and installation

Python 3.11 is the tested Python series. Independent inference and regression
checks have passed in the following environments.

| Environment | Python | PyTorch |
|---|---|---|
| Windows | 3.11.9 | 2.11.0+cu128 |
| Linux | 3.11.15 | 2.6.0+cu124 |

The inference requirements are `torch>=2.6`, `numpy`, and `opencv-python`.
Visualization additionally uses `matplotlib`. The Windows checks used NumPy
2.4.4, OpenCV 5.0.0, and Matplotlib 3.11.0.

A separate environment can be created with Conda.

```bash
conda create -n lightgen-temporal python=3.11
conda activate lightgen-temporal
python -m pip install -r requirements.txt
```

For GPU execution, select the PyTorch build matching your platform and CUDA
driver, then install the remaining requirements. The official installation
selector provides the corresponding command. [PyTorch installation](https://pytorch.org/get-started/locally/)

Optional visualization dependencies can be installed separately.

```bash
python -m pip install -r requirements-visualization.txt
```

The plotting tools use Arial. Install Arial in the operating system and verify
that Matplotlib can resolve it before generating figures.

## Input preparation

### Video files

The command accepts at least one video path. MP4 is used by the supplied example.
Video decoding is handled by the installed OpenCV backend. Quote paths that
contain spaces.

The inference entry point performs these steps for every video.

1. Read the total frame count and select four frame indices at approximately
   10%, 36.67%, 63.33%, and 90% of the video.
2. Convert each decoded frame from BGR to RGB.
3. Take a centered square crop with side length equal to 65% of the shorter
   image dimension, rounded to an integer.
4. Resize the crop to 448 × 448 using OpenCV area interpolation.
5. Stack the four frames as a `uint8` tensor of shape `4 × 3 × 448 × 448`.

Frame positions use `round(fraction * (frame_count - 1))`. Short videos can
therefore reuse frame indices. Pixel scaling for the frozen visual front is
performed inside the model.

### Ordered groups of sixteen

The entry point packs consecutive videos into groups of sixteen, producing
`1 × 16 × 4 × 3 × 448 × 448` pixel tensors. A short final group is filled by
cyclically repeating videos from the beginning of the input list. Only the
supplied videos receive records in the prediction JSON.

All sixteen slots participate in a joint coherent optical simulation. Preserve
the video order, group membership, and tail-padding convention when comparing
results. The same video can receive different predictions when its slot or
neighboring inputs change. The reference performance section reports this
slot sensitivity.

### Text conditioning

The checkpoint stores 38 language embeddings for the temporal-quality prompt.
The prompt asks for a rating using Excellent, Good, Fair, Poor, or Bad. The
command-line workflow uses these embedded tokens automatically.

The Python model's `set_prompt()` method accepts a compatible `38 × 2048`
embedding tensor. Preparing another prompt requires its matching Qwen-derived
embeddings and a corresponding evaluation profile.

## Outputs and score interpretation

A single-video result has the following JSON structure.

```json
{
  "model": "public_surrogate",
  "predictions": [
    {
      "video": "examples/demo.mp4",
      "slot": 0,
      "MOS": 47.528350830078125
    }
  ]
}
```

| Field | Meaning |
|---|---|
| `model` | The implementation identifier `public_surrogate` |
| `predictions` | One record for each supplied video, in input order |
| `video` | The supplied video path, with platform-specific path separators |
| `slot` | Position within its sixteen-video field, numbered 0 through 15 |
| `MOS` | Predicted temporal mean opinion score |

Slot numbering restarts for every group. The output directory is created
automatically. JSON is written after all requested groups finish successfully.

The readout produces five softmax probabilities ordered as Bad, Poor, Fair,
Good, and Excellent. These probabilities weight the five TRAIN-derived grade
scores stored in the checkpoint, approximately

```text
28.62, 40.79, 52.96, 65.13, 77.30
```

Their weighted sum gives the MOS. Higher values indicate higher predicted
temporal quality. The current weighted-score range is approximately
28.62 to 77.30. The synthetic example demonstrates the input/output workflow.
Dataset-level accuracy is evaluated using correlations with human MOS labels.

## Model structure

The visual front applies the extracted Qwen patch projection and position
tensors. Two fixed spatial mean-pooling operations produce 49 tokens per frame.
Task adapters, routing, experts, optical propagation, and readout then process
these tokens in sequence.

Shapes below refer to one video within a sixteen-video field.

| Stage | Operation | Output shape |
|---|---|---|
| Visual front | Frozen patch projection, position features, spatial mean pooling | 4 × 49 × 1024 |
| Visual adapter | LayerNorm1024 and Linear1024→192 | 4 × 49 × 192 |
| Visual router | Per-frame LayerNorm, Flatten9408, FC128, ReLU, FC4, softmax and Top2 | Four probabilities per frame |
| Visual experts | Four independent bias-free Linear192→192 experts, Top2 weighted sum | 4 × 49 × 192 |
| Vision field encoder | Channel projection 192→27 and token-axis projection 49→27, with ReLU | Four 27 × 27 amplitude patches |
| Vision global | Whole-field phase modulation, angular-spectrum propagation and intensity detection | Four 56 × 56 detector regions |
| Vision detector processing | Relative intensity, adaptive pooling, LayerNorm96, ReLU and Linear96→192 | 4 × 49 × 192 |
| Frame merger | Spatial mean/max, LayerNorm384, Linear384→192, ReLU and frame positions | 4 × 192 |
| Text adapter | LayerNorm2048 and Linear2048→192 | 38 × 192 |
| Joint sequence | Four visual tokens, 38 text tokens and sequence positions | 42 × 192 |
| Joint router and experts | Router reads all 42 tokens and mixes a separate set of four linear experts | 42 × 192 |
| Joint field encoder | Linear192→56, ReLU and row padding | One 56 × 56 amplitude patch |
| Joint global | Whole-field propagation and intensity detection | One 115 × 115 detector region |
| Joint detector processing | Relative intensity, adaptive pooling, LayerNorm96 and ReLU | 42 × 96 |
| Readout | Flatten4032, Linear4032→32, ReLU, Linear32→5, softmax and grade expectation | One MOS |

Within a stage, expert parameters are shared across its frames, videos, and
slots. The four experts have independent matrices. Each visual frame uses one
Top2 choice for all of its 49 tokens. The joint stage makes one choice per
video using both visual and text tokens. The selected two weights are
normalized by their sum. The software computes all four expert outputs before
applying the sparse weights.

Both global stages use a 518 × 518 propagation canvas and a central
478 × 478 active area. The vision stage places four frames per video in a
2 × 2 arrangement, forming 8 × 8 frame patches across the field. The joint
stage places one patch per video in a 4 × 4 arrangement. Both field encoders
apply per-patch root-mean-square (RMS) normalization.

Detector processing divides each region's intensity by its mean, with a
denominator floor of 0.000001. Adaptive pooling maps the detector grid to
49 × 96 vision features or 42 × 96 joint features. The final joint projection
has been merged algebraically into the first readout layer.

The 3,777,350 count refers to registered model parameters. Frozen visual tensors
and prompt embeddings are stored as buffers. The model card gives the selected
training history and evaluation conditions. [Model card](docs/MODEL_CARD.md)

## Reference performance

The LGVQ temporal reference uses 2250 TRAIN videos and 558 original TEST videos.
The original TEST split was used for development selection. The selected
32-unit readout comes from epoch 290 of 400, using the requested development
SRCC interval of 0.790 to 0.799.

| Evaluation condition | SRCC |
|---|---:|
| Original ordered groups, cached visual input, full-model forward | 0.7988976 |
| Direct video input over the same 558 videos | 0.7992455 |
| Mean across sixteen cyclic slot shifts | 0.7953460 |
| Minimum across sixteen cyclic slot shifts | 0.7885972 |

SRCC is the Spearman rank correlation coefficient. For the original ordered
cached-input evaluation, Pearson linear correlation coefficient (PLCC) is
0.8005053, Kendall rank correlation coefficient (KRCC) is 0.5864579, root mean
square error (RMSE) is 8.4795709, and mean absolute error (MAE) is 6.3281161.

The direct-video and cached-input evaluations have a mean absolute MOS
difference of 0.019374 and a maximum difference of 2.122620. Decoder, frontend
numerics, and grouping conditions should be recorded when reproducing scores.
Full LGVQ evaluation requires its videos, labels, exact sample order, and split
manifest. Dataset access is provided by the original authors. [LGVQ dataset](https://github.com/zczhang-sjtu/UGVQ)

### Electronic and optical controls

The from-scratch electronic control concatenates the complete fixed visual
`4 × 49 × 1024` and text `38 × 2048` embeddings. It trains a randomly initialized
`278528 → 32 → 5` readout with 8,913,093 parameters using TRAIN labels.
The best development SRCC is **0.7409566**, at epoch 30 of 400 with seed 713.

The optical reference inherits task-trained components and uses TRAIN teacher
distillation during readout refinement. Interpretation of the accuracy
difference should account for the input representation, initialization,
objective, and optimization history. The model card also records the earlier
task-initialized electronic capacity check and its SRCC of 0.7937766.

Post-training phase and propagation controls keep the other trained components
fixed. Constant phase in both globals gives SRCC 0.7945205. Identity propagation
in both globals gives 0.6436202. The supplied random-control summaries report
ten seeds and their sample standard deviation. Condition definitions are stored
in `results/global_ablation_protocol.json`.

## Optical-field visualization

After installing the visualization requirements and Arial, trace the supplied
example through both optical stages.

```bash
python tools/visualize_fields.py --profile configs/selected.json --videos examples/demo.mp4 --out outputs/fields
```

Choose a fresh output directory. The tool checks the selected checkpoint SHA256
and verifies that tracing reproduces the model's prediction. Use `--group` to
choose a sixteen-video group, `--detail-slot` to choose a slot, and
`--device cuda` for GPU execution.

| Generated file | Contents |
|---|---|
| `input_frames.png` | Cropped four-frame previews for the sixteen slots |
| `frame_global.png` | Vision-stage input, phase, modulated field, propagation and detection |
| `video_global.png` | Corresponding joint-stage optical panels |
| `slot_detail.png` | Selected slot's frames, detector regions, routing and grade probabilities |
| `frame_global_arrays.npz` | Vision-stage complex fields, intensities, phase and detector arrays |
| `video_global_arrays.npz` | Corresponding joint-stage arrays |
| `group_predictions.csv` | Slot predictions, video paths and valid-input flags |
| `routing.json` | Selected Top2 weights for both stages |
| `visualization_manifest.json` | Checkpoint, input and source hashes, display settings and numerical audits |

The CSV includes all sixteen physical slots. Padding slots have `valid=false`.
The NPZ files retain numerical arrays for additional plotting and inspection.
Phase plots use a linear scale from −π to π. Intensity display transformations
are recorded separately from the model's relative-intensity processing.

The release includes learned phase masks and expert-selection statistics.

![Learned vision and joint phase masks](results/phase_masks.png)

The expert-share figure reports each expert's Top2 selection count divided by
the total number of selections. [Optical expert share](results/optical_expert_share.png)

The provided random-control summary can be replotted directly.

```bash
python tools/plot_global_ablation.py --summary results/global_ablation_summary.csv --output outputs/ablation.png
```

Figures use Arial at 7 pt with 8 pt titles. New dataset-level expert shares and
full-dataset control evaluations require the corresponding dataset inputs and
author-side analysis workflow.

## Release files and verification

| Path | Purpose |
|---|---|
| `infer.py` | Video preprocessing, ordered grouping and JSON inference |
| `model.py` | Model components, optical simulation and checkpoint loader |
| `weights/LightGen.pt` | Selected standalone checkpoint |
| `configs/selected.json` | Checkpoint identity and evaluation profile |
| `examples/demo.mp4` | Deterministic synthetic demonstration video |
| `examples/make_example.py` | Script that regenerates the demonstration video |
| `examples/expected_predictions.json` | Recorded demonstration output |
| `tools/visualize_fields.py` | True-forward optical-field tracing and plotting |
| `tools/plot_global_ablation.py` | Plotting from the supplied control summary |
| `results/` | Phase images, expert shares, control metrics and protocols |
| `MODEL_CARD.md` | Training history, evaluation conditions and control interpretation |
| `requirements*.txt` | Inference and visualization dependencies |
| `LICENSE`, `NOTICE`, `licenses/` | Project license and upstream attribution |
| `MANIFEST.json` | Source commit, checkpoint export and per-file SHA256 hashes |

The checksum beside the ZIP verifies the complete archive. `MANIFEST.json`
records the distributed files and model-state fingerprint. The public-export
verification checks each model tensor against the selected research checkpoint.

In the directory containing `current.zip`, check its checksum with either of
these platform-specific commands.

```bash
sha256sum current.zip
```

```powershell
Get-FileHash -Algorithm SHA256 current.zip
```

Compare the result with `current.zip.sha256`. For a reproducibility report,
record the source commit, checkpoint SHA256, Python and PyTorch versions,
device, command, input video hashes, and exact video order.

## Troubleshooting

| Situation | Action |
|---|---|
| `Cannot decode video` or `Frame decoding failed` | Check the path and test decoding with the same OpenCV environment. Confirm that the video has a readable frame count. |
| CUDA initialization or device errors | Check the installed PyTorch build and GPU driver. Use `--device cpu` for the verified CPU workflow. |
| Arial font lookup error | Install Arial and refresh Matplotlib's font discovery before plotting. |
| `FileExistsError` during field visualization | Select a fresh `--out` directory. |
| Checkpoint SHA mismatch | Restore the matching checkpoint and `configs/selected.json` from the same verified release. |
| Scores change across runs with different input lists | Restore the same ordered groups, neighboring videos and cyclic padding. Then compare decoder, dependency and device versions. |

## License and acknowledgements

The code is distributed under Apache-2.0. Qwen-derived patch-projection,
position tensors, and prompt embeddings are acknowledged in `NOTICE`, with the
upstream license in `licenses/QWEN_LICENSE.txt`. [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)

LGVQ data should be obtained and cited through its authors' distribution.
The bundled demonstration is generated by `examples/make_example.py`.

## Author workspace

The research runtime files are `infer.py` and `model.py`. Google Drive inference
sources are maintained in the repository-root `LightGen2`. Research commands are
in `tools/`, detailed author notes are in `docs/DEVELOPMENT.md`, and the selected
research checkpoint and records are in `runs/simulation/current/`.

```bash
python -m pip install -r tools/requirements.txt
python tools/build_release.py
```

The generated release is `releases/current/`, with one distributable ZIP and
checksum alongside it. The release builder validates the exported tensors,
file hashes, privacy checks, and standalone example inference.

A PDF reading edition of the current root LightGen2 README can be included
through `--readme-pdf`. The canonical package is `releases/LightGen2.zip`.
