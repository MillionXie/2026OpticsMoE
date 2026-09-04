# Caltech-101 Multiplane Optical Retrieval

This independent experiment compares four primary Vision+Language optical stacks under
one Caltech-101 retrieval recipe.  The Qwen3-VL tokenizer, token embeddings,
Vision patch/position embedding and main visual merger are retained and frozen.
Native Qwen Transformer blocks, electronic Mixer branches, fusion gates and
identity residuals are not part of the Student.

Every modality begins with one electronic dimensional interface and ends with
one final CCD/readout interface.  There is no electronic residual bypass around
the optical stack.

## Controlled variants

| Variant | Optical planes per modality | Intermediate OEO | Router policy |
|---|---:|---:|---|
| `d2nn_continuous` | 5 D2NN 224x224 | none | no router |
| `d2nn_oeo_sigmoid` | 5 D2NN 224x224 | after planes 1--4 | no router |
| `moe_continuous_fixed_router` | 4 MoE4 expert + 1 global | none | one input Top-2 router |
| `moe_oeo_dynamic_router` | 4 MoE4 expert + 1 global | after every expert plane | four independent Top-2 routers |

`moe_oeo_fixed_router` is retained as a **supplemental ablation**, not as a
replacement for the D2NN OEO baseline.  It reuses the input Top-2 allocation.

The D2NN OEO transfer is deliberately fixed and simple:

```text
complete complex field
-> square-law CCD intensity over the full 224x224 aperture
-> non-affine normalization
-> sigmoid
-> zero-phase amplitude reload
```

The MoE OEO transfer is:

```text
complex field
-> square-law CCD intensity per expert
-> per-expert non-affine LayerNorm
-> sigmoid
-> apply routing weights
-> zero-phase amplitude reload
```

The final global plane always ends in the same final CCD/readout.  There are
four reload boundaries in each OEO variant and no reload after the final CCD.

The dynamic-router variant collapses the selected normalized expert responses
to one canonical 224x224 field, invokes the next independent router, and fans
the field out with the newly predicted Top-2 weights.  This is intentionally
different from merely evaluating one shared router several times.

See [ARCHITECTURE.md](ARCHITECTURE.md) for data flow and physical assumptions,
and [RUN_COMMANDS.md](RUN_COMMANDS.md) for commands.

## Saved evidence

Each run saves:

- resolved `config.yaml` and `architecture_report.json`;
- `train_log.csv` and per-epoch train/test retrieval metrics;
- last, minimum-train-loss, best-observed-test and EMA checkpoints where the
  shared trainer enables them;
- per-plane phase tensors and a 2x5 Vision/Language phase preview;
- router selection/importance diagnostics already present in the shared log;
- final Teacher/Student retrieval CSV/JSON and confusion matrices from the
  shared evaluator.

`compare_results.py` consolidates the four primary runs plus the supplemental
fixed-router OEO run into plot-ready
`comparison/comparison.csv` and `comparison/comparison.json`.  It also emits
`router_diagnostics.csv` (stage/expert selection rate, probability importance
and sparse weight) and `power_diagnostics.csv` (per-stage optical power and
final CCD scale).  These tables retain parameter count, runtime, phase motion,
Top-1, Top-3 and MRR for the selected checkpoint and final epoch, plus the
explicitly labelled selection-biased best-observed/EMA test values.  Later
paper plots therefore do not need to scrape terminal logs or conflate these
different checkpoint definitions.

## Runtime-limited formal schedule

Full epochs measured 7.5--8.6 minutes and would make the primary 80-epoch
comparison take more than 43 hours.  The release recipe therefore runs 28
partial epochs of 1,300 samples (`65 steps x batch 20`).  Every batch contains
all ten classes with two images per class.  A deterministic per-class cyclic
queue guarantees that each image is used before that class begins repeating;
the largest class is fully covered within six epochs.  This schedule is shared
by all primary variants.  The supplemental run adds runtime if it is also run.
