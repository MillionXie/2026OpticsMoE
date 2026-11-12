# LightGen-II temporal quality reference

A compact reference model for temporal video-quality assessment, with Top2
routing, four linear experts per stage, two simulated optical global stages,
and a five-level electronic readout. The model has 3,777,350 parameters.

## Quick start

From the unpacked release directory, install the inference dependencies and run
the supplied synthetic example.

```bash
pip install -r requirements.txt
python infer.py --weights weights/LightGen.pt --videos examples/demo.mp4 --device cpu --output outputs/predictions.json
```

Use `--device cuda` for a CUDA-enabled PyTorch installation. Supply video paths
after `--videos` in their intended evaluation order. Each prediction contains
the video path, its physical field slot, and the estimated mean opinion score
(MOS).

The checkpoint embeds the frozen visual components and fixed-prompt embedding
used by this reference model. Inference is available offline.

## Input and model

Each video contributes four RGB frames sampled at 10%, 36.67%, 63.33%, and 90%
of its duration. A centered square crop spanning 65% of the shorter image side
is resized to 448×448. Sixteen videos share one coherent simulated field.
The final group is cyclically padded from the beginning of the input list.

| Stage | Per-video representation |
|---|---|
| Frozen visual front and adapter | 4 × 49 × 192 |
| Visual Top2 router and linear experts | 4 × 49 × 192 |
| Vision global and frame merger | 4 × 192 |
| Fixed prompt concatenation | 42 × 192 |
| Joint Top2 router and linear experts | 42 × 192 |
| Joint global and detector processing | 42 × 96 |
| Two-layer readout, 4032 → 32 → 5, and grade expectation | one MOS |

The global simulations use a 518×518 canvas with a central 478×478 active area.
The vision field contains an 8×8 arrangement of frame patches. The joint field
contains a 4×4 arrangement of video patches.

Each detector region is divided by its mean intensity with a protected
denominator, pooled, and passed through LayerNorm and ReLU. The vision stage
then projects 96 channels to 192. The joint stage supplies its 96-channel tokens
directly to the readout through a merged linear map. The text adapter and the
custom mean/max frame merger are retained.

## Reference results

The original 558-video TEST split was used for development selection. The
fixed-group cached-feature SRCC, evaluated through the full model, is 0.79890.
The 32-unit readout is selected at epoch 290 of 400 TRAIN-only fine-tuning
epochs using the requested 0.790–0.799 development-SRCC range.
Full evaluation conditions and selected-weight checks are described in
`MODEL_CARD.md`.

An electronic-capacity check with 942,341 parameters and an 8064 → 32 → 5 head
achieved development SRCC 0.78425 after training its retained electronic layers.
The optical reference fine-tunes its readout. The differing head input sizes and
optimization scopes are described in `MODEL_CARD.md`.

Post-training phase and propagation controls retain each stage's input/output
interface. Their condition definitions and selected-weight results are supplied
in `results/`.

## Visualization

```bash
pip install -r requirements-visualization.txt
python tools/plot_global_ablation.py --summary results/global_ablation_summary.csv --output outputs/global_ablation.png
python tools/visualize_fields.py --profile configs/selected.json --videos examples/demo.mp4 --out outputs/fields --device cpu
```

Figures use Arial at 7 pt with 8 pt titles. Install Arial for these rendering
commands. Phase plots use a linear −π to π color scale.

## Release contents

```text
infer.py
model.py
weights/LightGen.pt
configs/selected.json
examples/demo.mp4
examples/make_example.py
tools/
results/
MODEL_CARD.md
LICENSE
NOTICE
licenses/QWEN_LICENSE.txt
MANIFEST.json
```

`MANIFEST.json` records the SHA256 of every distributed file and the model-state
fingerprint. The exported checkpoint retains bitwise-identical model tensors
with public release metadata.

## License

Apache-2.0. The Qwen-derived visual components and prompt embeddings are
acknowledged in `NOTICE`, with the upstream license in
`licenses/QWEN_LICENSE.txt`. [Upstream model](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)

## Author workspace

The canonical runtime files are `infer.py` and `model.py`. Research commands are
in `tools/`, detailed author notes are in `docs/DEVELOPMENT.md`, and the selected
research checkpoint and records are in `runs/simulation/current/`.

```bash
pip install -r tools/requirements.txt
python tools/build_release.py
```

The generated release is `releases/current/`, with one distributable ZIP and
checksum alongside it. The release builder validates the exported tensors,
file hashes, privacy checks, and standalone example inference.
