# LGVQ spatial/temporal optical-router VQA

This experiment adapts the current Caltech four-layer **optical core, execution
order, balanced fusion, and router contract** to LGVQ. It is not a bit-for-bit
full-Qwen replacement: its Qwen boundary is the documented lightweight frozen
cache interface needed for the sister experiment's 196-token convention. It
predicts exactly two values per video:
`spatial_quality` and `temporal_quality`. Image-text alignment is deliberately
excluded from the manifest, cache, model head, loss, metrics, and checkpoints.

The primary design is ours; `lgvq_optical_parallel16` is used only for three
useful LGVQ conventions: four sampled frames, prompt-group splitting, and a
four-frame parallel optical plane. It does **not** import the sister model's
alignment target, hashed prompt/generator shortcuts, SPAQ dependency, or large
Transformer temporal head.

## Delivered comparison

| Config | Router | Active experts per MoE4 | Same downstream model |
|---|---|---:|---|
| `e1.yaml` | electronic | 1 | yes |
| `e2.yaml` | electronic | 2 | yes |
| `e4.yaml` | electronic | 4 | yes |
| `o2.yaml` | optical detector energy | 2 | yes |

All four variants use corrected straight-through selection and `power_l2`
weights. O2 replaces only the router: its Vision router produces all four
frame-wise MoE4 decisions in one 1024-plane propagation, while its Language
router produces one per-video MoE4 decision on the unchanged 518/478 plane.
The present router table deliberately uses the **low-optical fusion regime**
only: every feature layer learns `alpha` in `[0.01,0.49]`, initialized at
`0.055`. It does not constitute an `alpha>0.5` or optical-off ablation.

O2 keeps the same downstream feature blocks and prediction head, but it is not
computationally identical to E2: it replaces the electronic score head with a
trainable router phase, an extra propagation and detector integration, a router
phase learning rate of `1e-2`, and a capture regularizer. Those differences are
the mechanism under comparison, not hidden implementation drift.

> **Hardware boundary:** the 1024/986 parallel16 plane is a simulation contract
> inherited for a fair comparison with the uploaded sister architecture. A
> 986-pixel field at 17 µm spans 16.762 mm, equivalent to about 2,095 pixels on
> the 8 µm phase SLM, which exceeds its 1,920-pixel width and the currently
> estimated 518/478 usable ROI. Therefore this project does not claim direct
> deployment on the current Caltech bench and intentionally contains no fake
> hardware bridge. A future hardware version must either reuse the verified
> 518/478 plane sequentially (more exposures) or define smaller lanes and
> retrain them.

## Data protocol

- LGVQ metadata are joined by normalized video path, never row number.
- `prompt_cls.json` has the uploaded `{data: [...], length: 2808}` schema;
  `MOS.txt` is independently parsed as `path;spatial;temporal;alignment` and
  spatial/temporal values must agree for all 2,808 paths.
- The 468 prompts are split as 375 train groups (2,250 videos) and 93 test
  groups (558 videos). All six generators for a prompt remain together.
- There is no validation split. Test is evaluated at epoch 1, every 5 epochs,
  and the final epoch. `best_observed_test_checkpoint.pt` maximizes the mean of
  spatial SRCC and temporal SRCC. This intentionally uses test for selection,
  as requested, and the checkpoint records that policy.

The frozen Qwen cache is a separate reproducible stage. Each MP4 contributes
frames at 10%, 37%, 63%, and 90%; each frame is center-cropped to 65% of the
short side and resized to 448x448. The processor is fixed to 200,704 pixels,
giving a 28x28 pre-merger grid (784 tokens). Qwen's real `patch_embed` and
`fast_pos_embed_interpolate` create `[784,1024]`, then each contiguous 2x2
block-major group is mean-pooled to `[196,1024]`. This deterministic reduction
does not call the learned Qwen merger. The exact five-level prompt is passed
through the frozen Qwen language backbone once and stored as singleton
`[1,L,2048]`, then broadcast without duplicating gigabytes of cache data.
Although only the final five labels were required to remain fixed, this run
locks the complete sentence as an additional experimental control:
`Please evaluate the quality of this video and rate it using one of the
following five levels: Excellent, Good, Fair, Poor, or Bad.`

See [ARCHITECTURE.md](ARCHITECTURE.md) for every tensor shape and
[RUN_COMMANDS.md](RUN_COMMANDS.md) for the complete runnable sequence.

## Scope and initialization

The formal training input is a frozen-Qwen cache, not raw pixels. This keeps the
2B backbone out of every optical ablation step and makes E1/E2/E4/O2 comparable.
If an old compatible checkpoint is supplied, only exact name-and-shape matches
are loaded. SPAQ is never required; otherwise the 1024→192 and 2048→192 Qwen
boundary projections use an explicit orthogonal initialization report.

`--phase smoke` swaps in a tiny but geometrically valid plane and runs E2 and
O2 forward/backward on CPU. It is a wiring test, not a performance result.

## Uploaded sister version versus this experiment

| Item | Uploaded `lgvq_optical_parallel16` | This project |
|---|---|---|
| Visual input boundary | 384 preprocessing / old 192-wide cache | 448 preprocessing; real Qwen patch+position `[784,1024]`, deterministic 2x2 mean to `[196,1024]`, then 1024→192 |
| Language/multimodal input | no complete frozen-Qwen boundary; hashed prompt, generator one-hot, and handcrafted auxiliaries | exact fixed prompt through frozen Qwen `[1,L,2048]`; four final-Vision tokens enter before Language Block 1 |
| Four-layer order | architecture-specific parallel shortcuts | Vision expert → Vision global → merger → Language expert → Language global |
| Temporal model | Transformer temporal head | attention-free depthwise Conv1D lightweight head |
| E/O residual | unbalanced `E + alpha*O` style | equal-RMS `(1-alpha)E + alpha O` with common post-rescale `F` |
| Routing | electronic router baseline | fair E1/E2/E4 and pure detector-energy O2 interfaces |
| Objective | five losses / three quality targets | Smooth-L1 + pairwise rank; only spatial and temporal |
| Phase learning rate | approximately `1e-6` in the uploaded setup | feature phase `6e-3`; optical-router phase `1e-2` |
| Initialization dependency | SPAQ/checkpoint/cache assets referenced but absent on the uploaded server | SPAQ optional; explicit Qwen-boundary orthogonal fallback |
| Alignment | three-head output includes alignment | alignment hard-disabled end to end |
| Propagation organization | serial/parallel shortcuts including an eight-propagation path | two parallel Vision feature propagations plus two serial Language feature propagations; O2 adds router propagations |

### What is reused, and what remains independent

| Component | Shared/reused | Independent |
|---|---|---|
| Frozen Qwen boundary | one 1024→192 Vision adapter and one 2048→192 Language adapter | input frame/token values differ per video |
| Four video frames | the same field encoder, CCD readout and electronic frame router are applied to every frame | the 16 Vision-expert phase masks are independent (4 frames × 4 experts) |
| Vision global optics | one operation processes the four frame lanes in parallel | one trainable 986×986 phase plane covers the complete parallel canvas; it is not four copied masks |
| Language optics | the verified 518/478 geometry and the same encoder/readout contract are reused | Language expert/global phases are separate from Vision phases |
| Output electronics | one lightweight temporal head and one regressor are shared by all samples | no generator-specific head or one-hot shortcut exists |

Thus “reuse + shrink + parallel” means parameter sharing in the adapters,
encoders/readouts and frame router, deterministic 1024→192/784→196 feature
reduction, and simultaneous four-frame execution. It does not mean that all
optical phase masks are tied together.

The uploaded sister results cannot currently be independently reproduced from
the server copy because referenced checkpoint/cache dependencies are missing.
Accordingly, its code informs conventions but is not used as a numerical
baseline until those assets are supplied.
