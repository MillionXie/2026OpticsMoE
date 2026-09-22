# Frozen CLIP and YOLO11s LGVQ baselines (2026-09-22)

## Results

On the fixed LGVQ prompt-group 2250/558 split, both visual backbones were fully
frozen.  Four ordered frame features were concatenated and only five bias-free
quality rows were trained (10,240 parameters per target).

| Model | Total backbone params | Trainable params | Temporal SRCC | Spatial SRCC | Temporal best epoch | Spatial best epoch |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-VL-2B reference | 2.128B | 10,240 | 0.7663 | 0.6908 | - | - |
| DeepSeek-VL2-Tiny reference | 3.371B / 1B active | 6,400 | 0.5688169 | 0.5974811 | 50 | 48 |
| OpenAI CLIP ViT-B/32 | 151,277,313 | 10,240 | **0.7251568** | **0.6974061** | 50 | 49 |
| Ultralytics YOLO11s | 9,458,752 | 10,240 | **0.7015822** | **0.5543498** | 49 | 50 |

Additional CLIP metrics were temporal PLCC 0.7282836 / KRCC 0.5248461 and
spatial PLCC 0.7220874 / KRCC 0.5083717.  Additional YOLO11s metrics were
temporal PLCC 0.7038590 / KRCC 0.5062812 and spatial PLCC 0.6012374 / KRCC
0.3899257.

This is a first fixed-protocol run, not a hyperparameter search.  As in the
existing Qwen/DeepSeek protocol, the reported epoch is selected by maximum test
target SRCC because the inherited split has no validation set.  These values
therefore must not be described as untouched-test estimates.

## Architecture and frozen-backbone audit

- Frame fractions: 10%, 37%, 63%, and 90%; 65% short-side center square crop.
- CLIP: official ViT-B/32 preprocessing at 224x224; normalized final image
  embedding, 512 values per frame.  Initial quality rows are native normalized
  CLIP text embeddings repeated over the four positions and divided by four.
- YOLO11s: 640x640 input; normalized global-average feature from backbone layer
  10 (`C2PSA`), 512 values per frame.  Initial rows use Xavier uniform, seed 42.
- Ordered concatenation produces 2,048 values.  The only optimized tensor has
  shape `[5, 2048]`; both runtime reports confirm zero trainable backbone
  parameters and exactly 10,240 trainable readout parameters.
- Both targets use cross entropy over the same five equal-width target levels,
  AdamW, learning rate 0.001, zero weight decay, cosine schedule, 50 epochs,
  batch 512, and seed 42.

CLIP and YOLO are pure visual baselines: unlike Qwen and DeepSeek, their frozen
backbones do not receive the temporal/spatial natural-language prompt.  They
test the contribution of pretrained visual representations plus a parameter-
matched output readout, not a native VLM decoder.

## Reproduction identity

- Git commit used by both formal runs:
  `95904ecf97bcefe4a7ae76a6eab08a1d7cb217f0`; both run identities report a
  clean worktree.
- LGVQ manifest SHA256:
  `607c50d20662a47795c23cd2038081b30368ed108ba9be8a1bb8a9d250f8e7fc`.
- OpenAI CLIP checkpoint SHA256:
  `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`.
- YOLO11s checkpoint SHA256:
  `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5`.
- CLIP serialized feature-cache SHA256:
  `62c918f6529aedc1ab4861e9cc22616ef9410ede540d282974f30b238046753c`;
  extraction took 507.8535 seconds.
- YOLO11s serialized feature-cache SHA256:
  `ff1d4c6469660c6d0bb4ff6177edc2ceae7896777c30ae77b45084abb6be58dca`;
  extraction took 553.4817 seconds.
- CLIP temporal/spatial best-checkpoint SHA256:
  `f364f6458eff4ba57015564ba1187eb0c5d59371140190ec825dcaddb2f89089a` /
  `f7f430774c018e92a3233d87e82f1ff4fc495c7cfe1cd811d29e8aeaee759abe2`.
- YOLO11s temporal/spatial best-checkpoint SHA256:
  `8ac8f0965f6ad011392dd3e1e98e14f8ddb3d49ffbb3dfeb108059d30ee47c51` /
  `e9334716444102ab5196d0121693a646ceea03b9e15ba5b4f5c91caf1a5bd40f`.
- Environment: Python 3.11, PyTorch 2.6.0+cu124, CUDA 12.4,
  `ultralytics==8.4.159`; one RTX 4090 per baseline.  Both allocated GPUs were
  back to 12/25 MiB and 0% utilization after completion.

Large feature caches, checkpoints, histories, and 558-row prediction files are
kept under the ignored task run directories.  The committed implementation,
configs, tests, and protocol documentation are sufficient to regenerate them.
