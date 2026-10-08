# Unified editor, matched 256 × 256 check (2026-09-26)

Both current models now take and produce 256 × 256 RGB. The large checkpoint
was already native-256; the small Qwen-mini/optical checkpoint was warmed from
the 128 × 128 version (99.927% parameter coverage, optical phase masks
interpolated from 16² to 32²), then trained for two more epochs. The new small
checkpoint has 9,502,616 counted parameters. The large has 142,544,637.

The former “small MSE 0.00116 versus large MSE 0.03963” comparison was invalid:
the first was 128² RGB MSE, the second was 32² latent MSE. Mean MSE already
divides by the number of elements, so pixel count alone does not account for
the numerical gap. The common pixel space, image content and target pairing
matter. Also, MSE alone rewards smooth averaging and does not measure edge
fidelity or perceptual quality.

## Same RGB metric and test pairs

`audit_256.json` compares 288 held-out pairs: one view from eight distinct
product identities in each of lamp/table/pillow, all twelve edit variants per
view. Both use a zero noise tensor and the same 256² raw target images. Pixels
are normalized to [-1, 1]. This is a diagnostic subset, not an open-set image
generation score or a cross-seed quality comparison.

| Mode | Large RGB MSE | Small RGB MSE | Lower |
| --- | ---: | ---: | --- |
| Background | 0.005463 | 0.002084 | Small |
| Object | 0.001764 | 0.003891 | Large |
| Both | 0.003021 | 0.005756 | Large |

The small model's complete 2,304-pair test MSE is 0.003694. The 288-pair
diagnostic above is a different sample, so these should not be conflated.
The new small gallery shows that 256² does **not** automatically fix appearance:
lamp replacements and some tabletop/pillow boundaries remain ghosted or
blurred. The large model remains the safer quality reference.

## One-image timing demo

`timing_256.json` records CUDA events on one RTX 4090, batch 1, 5 warmups and
30 repeats. Start = input to the first retained language Transformer block;
end = final 256² RGB tensor. Tokenization, token embedding and the projection
immediately before the first mini block are excluded. The optical FFT simulator
is removed from the timed path; 1.0447 × 6 = 6.2682 ms is then added as a
**hypothetical optical-device delay**, not a measurement of this physical
system. This additive estimate is conservative when parallel optical and
electronic residual paths can overlap. Baseline is the older 28-block Qwen +
pure electronic decoder, trained for background editing only; timing is
comparable at this boundary, task quality is not.

| Model | Counted parameters | Electronic-path mean | + optical estimate | Speed-up versus baseline |
| --- | ---: | ---: | ---: | ---: |
| Qwen-28 + electronic decoder baseline | 1,824.20 M | 63.53 ms | 63.53 ms | 1.00× |
| Large unified, Qwen-mini-2 + optical decoder | 142.54 M | 28.88 ms | 35.14 ms | 1.81× |
| Small unified, Qwen-mini-2 + optical decoder | 9.50 M | 7.06 ms | 13.33 ms | 4.77× |

The shared frozen Qwen token embedding (311,164,928 parameters) is excluded
from all counts by project convention. These are software + assumed-device
estimates, not demonstrated wall-clock optical-hardware speedups.

## Optical alpha and routing

The learned `alpha` is 0.51036 (large) and 0.49808 (small). It mixes RMS-scaled
parallel optical and electronic residual features. It is **not** a percentage
of photons, multiply-adds or latency. Expert/global sigmoid gates are
0.2313/0.2306 (large) and 0.1996/0.1933 (small).

The learned FFT router has four unnamed experts and selects two per example.
Values below are overall Top-2 selection frequencies on the same 288 pairs;
each row sums to 200%.

| Model | E0 | E1 | E2 | E3 |
| --- | ---: | ---: | ---: | ---: |
| Large | 62.50% | 37.85% | 2.43% | 97.22% |
| Small | 76.04% | 72.22% | 10.76% | 40.97% |

Detailed per-mode softmax, selections and normalized mixture weights are in
`audit_256.json`. The nearly unused E2 of the large model is **routing
collapse**, even though its mean softmax probability is nonzero. Avoid
presenting the four branches as equally active optical experts. The routing
is measured in the learned FFT simulation, not on fabricated optical hardware.

## Artifacts

- Large checkpoint: `expanded_large_v2_unified_editor.pt` (unchanged).
- Native-256 small checkpoint: `expanded_small9m_256_best_model.pt`.
- Native-256 small preview: `expanded_small9m_256_gallery.jpg`.
- Full small training/test summary: `expanded_small9m_256_training_summary.json`.
- Paired RGB error, alpha and routing: `audit_256.json`.
- Timing samples and assumptions: `timing_256.json`.

Training and inference use at most one model call for the image generator; no
GAN or iterative diffusion loop is involved. Both task GPUs were released
after the runs.
