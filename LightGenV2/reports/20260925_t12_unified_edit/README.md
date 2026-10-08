# Unified ABO product editor: 2026-09-25 pilot

For the clean, by-task preview and checkpoint index, start with
[deliverables/START_HERE.md](deliverables/START_HERE.md).

This pilot uses one checkpoint per size for three instructions: replace only
the room/background; replace only the product while retaining the room; or
change both. It uses lamps and tables only, not chairs. Inference receives a
full RGB input and a text instruction. It never receives the target image or
mask, and does not paste input pixels into the output. The training masks and
procedural room renderer are used only to create supervised pairs.

## Scope and data

- Source: ABO CleanRender, CC BY 4.0; 1,152 train, 128 validation, 128 test
  source views across lamps and tables. Splits are by product sequence.
- Eight paired instructions per view: four background, two object, two joint.
  This yields 9,216 train and 1,024 validation/test pairs each.
- Targets use four fixed product designs (two per category) and a constrained
  procedural room grammar. Thus this is **not** open-set object generation or
  unconstrained arbitrary-prompt text-to-image. Test views are unseen, but the
  target design vocabulary is not unseen.
- The first run used 128-pixel ABO files resized to 256 for the large editor;
  a follow-up experiment uses genuine 256-pixel ABO render files. The small
  editor operates natively at 128 pixels.

## Models

| Model | Counted parameters | Text frontend | Image generator | Output |
|---|---:|---|---|---:|
| Narrow large editor | 142,528,245 | 2-layer Qwen-mini 640 + bridge (11,904,288) | optical/electronic one-step UNet (42,682,473), VAE encoder (34,163,664), VAE decoder (49,490,199), adapter/router | 256 |
| Small editor | 13,711,128 | 2-layer Qwen-mini | compact optical/electronic full-frame decoder | 128 |

The shared frozen Qwen token embedding is excluded from the parameter budget
as agreed; it remains required for arbitrary prompt tokenization/lookup. The
large checkpoint bundles the two-layer text head and bridge. Optical and
electronic branches run in parallel in the optical blocks; the learned optical
fusion alpha is about 0.5. The large model makes one UNet call; there is no
multi-step diffusion loop. VAE and the token embedding are frozen; the narrow
UNet was initialized by overlapping-weight transfer, not trained from scratch.

## Current evidence and caveats

- Old 128-source large test latent MSE: 0.04312 overall; 0.05947 background,
  0.01370 object, 0.03986 joint. This is a latent reconstruction metric, not
  perceptual image quality.
- Small test image-space MSE: 0.00111 background, 0.00091 object, 0.00189
  joint. Visual artifacts remain, especially in object/joint mode.
- Small FID-like score 47.23, KID 0.01086 ± 0.00172 using *torchvision*
  ImageNet Inception-v3 features. This is **not canonical TensorFlow FID** and
  must not be compared directly with literature FID. The repeated paired
  targets/four-design vocabulary further limit distribution-metric meaning.
- The exact text bridge matches the 244 training prompts closely but has not
  established robust generalization to free-form unseen instructions.
- Blur remains visible. True-256 VAE reconstruction is substantially more
  detailed than the previous source-upsample path in some samples, but merely
  raising source resolution does not guarantee a better trained editor. The
  true-256 run exposed a mask-preprocessing bug: erosion removed thin lamp
  stems. The corrected alpha-mask pilot (`large_143m_true256_alpha_epoch1.jpg`)
  restores that anatomy but is only one epoch, and its test latent MSE is
  0.05480 on a *different* target distribution. It should not replace the
  older 0.04312 model based solely on this metric or one preview grid. No
  perceptual or adversarial loss was added in this pilot.

See `large_143m_epoch2.jpg`, `small_final_grid.jpg`, and the JSON metrics in
this folder. A copy of the model-generated output for a white-background lamp
input is `infer_smoke_white_input.png` (an out-of-distribution input; not a
quality benchmark). The user-facing controls should be tested with room-scene
inputs similar to training examples.

Model paths on the training server:

- 142.53M original paired-data pilot: `t12_assets/runs/abo_unified_qwenmini_143m_v1/unified_editor.pt`
- 142.53M corrected true-256-alpha pilot: `t12_assets/runs/abo_unified_qwenmini_143m_true256_alpha_v2/unified_editor.pt`
- 13.71M small pilot: `t12_assets/runs/abo_unified_qwenmini2_optical_13m_v1/best_model.pt`

All training and evaluation processes exit after their work and release their
task GPU allocation. The packed large checkpoint still expects the frozen VAE
and shared Qwen token embedding from the model caches; it is not a standalone
single-file inference artifact.
