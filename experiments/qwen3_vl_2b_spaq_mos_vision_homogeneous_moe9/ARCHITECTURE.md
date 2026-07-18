# Architecture

## Student input cache

The original RGB SPAQ image is decoded and resized by the frozen Qwen image processor once. Its `pixel_values` and `image_grid_thw` tensors are stored as float16/float32 processor-cache shards. Student epochs restore these arrays as float32 and feed them directly to the frozen Qwen visual stem. No JPEG decode or processor resize is repeated during an epoch.

Teacher-hidden and processor-input cache shards use the same sample boundaries. Epoch sampling is random at the shard level and within each shard, rather than globally random at individual-sample granularity. This retains stochastic ordering while avoiding LRU shard thrashing.

## Preserved from the CIFAR-10 vision homogeneous MoE

- 480x480 optical canvas and centered 450x450 active area.
- Nine 120x120 homogeneous D2NN experts on a 3x3 grid with 150-pixel pitch.
- Input-conditioned electronic top-3 router.
- Five expert phase planes, optoelectronic interlayers, global phase, and full-plane detector readout.
- Strict per-image token boundaries and RGB Qwen preprocessing.
- Frozen Qwen patch embedding; all original Qwen vision blocks are replaced in the student.
- Separate router learning rate, phase dropout, parameter reports, routing logs, checkpoints, and field/phase visualizations.

## Regression-specific changes

- Dataset is SPAQ and only `MOS` is used.
- Output dimension changes from ten logits to one normalized score.
- Teacher and student both use `LayerNorm(1024) -> Linear(1024,1)` with a linear output.
- Their structures match, but the student head is freshly initialized; neither LayerNorm nor Linear parameters are copied from the teacher.
- The old frozen Qwen feature cache remains valid, but a former Sigmoid teacher head and its cached predictions must be regenerated.
- CE and categorical KD become ground-truth SmoothL1 and teacher-score SmoothL1.
- Accuracy/F1/confusion matrix become MAE, RMSE, SRCC, PLCC, threshold accuracy, and MOS scatter plots.
- Teacher cache stores floating-point normalized targets rather than class labels.

No language decoder or multimodal text path is executed.
