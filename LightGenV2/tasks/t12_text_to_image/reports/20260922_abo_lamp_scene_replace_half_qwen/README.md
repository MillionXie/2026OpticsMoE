# ABO lamp compositional background replacement (half-depth Qwen)

## Task

The input is an ABO CleanRender lamp already composited into a scene.  A text
instruction specifies a new room, colour tone, brightness, and side-window
direction; the model generates the replacement background and preserves the
source object with its provided alpha matte.

The control space contains 48 combinations:

- room: modern study, minimalist bedroom, boutique hotel lounge, concrete loft;
- tone: cool, warm, muted neutral;
- brightness: dim or bright;
- window direction: left or right.

Eight exact combinations are excluded from training.  Validation and test use
only these held-out combinations, while every individual attribute value is
present in the 40 training combinations.  Product identities are also split.
The data contain 4,608 training pairs, 512 validation pairs, and 512 test pairs.
All underlying ABO examples are CC BY 4.0; generated backgrounds contain no
external image assets.

Example instruction: `Keep the object unchanged and replace its current
background with a cool-toned, dim modern study with soft window light from the
left.`

## Architecture

The frozen Qwen3-VL language Transformer is physically truncated from 28 to 14
blocks.  The visual tower and LM head are not used.  Retained language-block
parameters are 704,704,000 (from 1,409,408,000), and the retained text encoder,
including embeddings and final normalization, contains 1,015,870,976 frozen
parameters.  Its masked-mean feature feeds a 4,273,280-parameter condition
adapter.

The image path is frozen VAE encoder -> one BK-SDM-v2-Tiny UNet call -> frozen
VAE decoder.  At the first/deepest decoder up block, electronic and optical
branches consume the same activation in parallel and are RMS-fused.  The
optical branch has 2,786,277 parameters and the selected checkpoint has
`alpha=0.4996` (constrained to at least 0.4).  The generation tail contains
383,413,495 parameters; approximately 246.22M were trainable.  There is no GAN
and no iterative denoising loop.

## Result

Epoch 2 is selected by held-out validation MSE.  On 512 identity-disjoint test
pairs whose exact target combinations were never trained:

| Metric | Value |
|---|---:|
| Latent MSE | 0.167573 |
| Background L1 | 0.267239 |
| Improvement over copying the input | 72.93% |
| Room / tone / brightness / direction accuracy | 100% each |
| Exact four-attribute combination accuracy | 100% |
| UNet calls / VAE decoder calls | 1 / 1 |

The generated backgrounds remain softer than the procedural targets, but the
held-out control attributes and scene replacement are visibly present.  In the
grids, columns are input scene, target scene, and generated scene.

- `pair_definition_preview.jpg`: data-pair definition before training.
- `test_grid.jpg`: best-checkpoint held-out test examples.
- `test_results.json`: test metrics and parameter accounting.
- `training_summary.json`: all four epochs and model selection.

Server checkpoint:
`/DATA/DATA1/guest3/t12_assets/runs/abo_lamp_scene_replace_half_qwen_v1/best_model.pt`
