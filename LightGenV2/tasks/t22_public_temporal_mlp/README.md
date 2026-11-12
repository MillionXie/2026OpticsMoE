# LightGen-II temporal quality reference

A compact reference model for temporal video-quality assessment, with Top2
routing, four linear experts per stage, two simulated optical global stages,
and a five-level electronic readout. The model has 5,732,678 parameters.

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
| Joint Top2 router, linear experts, and global | 42 × 192 |
| Two-layer readout and grade expectation | one MOS |

The global simulations use a 518×518 canvas with a central 478×478 active area.
The vision field contains an 8×8 arrangement of frame patches. The joint field
contains a 4×4 arrangement of video patches.

## Reference results

The original 558-video TEST split was used for development selection. The
fixed-group cached-feature SRCC is 0.79910, and direct video-input SRCC is
0.79924. The visual expert selection shares are 26.79%, 25.09%, 23.34%, and
24.78%. Full evaluation conditions are described in `MODEL_CARD.md`.

Post-training global interventions and their random-seed results are supplied
in `results/`. These evaluations fix the remaining model components and apply
the stated intervention directly.

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
