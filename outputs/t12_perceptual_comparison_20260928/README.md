# T12 perceptual re-evaluation, fixed TEST 2304 (2026-09-28)

This audit reads the **saved native 256×256 RGB PNGs**, not model tensors. It uses all 2304 fixed TEST examples, 768 each of background/object/joint editing. `per_image_lpips.csv` and `per_image_dists.csv` hold every image score. `summary_*.csv` hold mode-wise arithmetic means. Six annotated figures use the exact indices in `handoffs/t12_four_group_summary_20260927/figure_indices.json` (first and halfway example per category/mode), with **no quality-based selection**.

## Results

All-test figures below use quantized PNGs. The previous official model-forward mean PSNR for small SIM is 34.277751 dB; PNG re-evaluation gives 34.249675 dB. Do not silently exchange these measurement protocols.

| Model | Condition | Full PSNR ↑ | Full LPIPS-Alex ↓ | Full DISTS ↓ | Object-mode ROI LPIPS ↓ | Object-mode ROI DISTS ↓ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Small ours | clean SIM, `small.pt` SHA `5b4f` | 34.250 | 0.151 | 0.186 | 0.174 | 0.184 |
| Large ours | clean SIM | 31.418 | 0.118 | 0.127 | 0.058 | 0.089 |
| Qwen28 + decoder | clean SIM | 27.254 | 0.189 | 0.142 | 0.053 | 0.080 |
| pix2pix-Turbo | clean SIM | 20.741 | 0.323 | 0.211 | 0.430 | 0.263 |
| Small ours EXP | **physical output; different electronically tuned checkpoint, SHA `eeec`** | 31.486 | 0.184 | 0.197 | 0.173 | 0.187 |

The clean SIM four rows are the comparable image-generation set; the EXP row is a separate diagnostic, **not** a same-weight clean/physical gap. The physical row's PSNR comes from saved PNGs; its official forward/reconstruction report is 31.552886 dB.

For **object replacement** (768 images), small clean SIM has whole-image PSNR 35.372 dB versus large 34.645 dB, but edited-region PSNR **31.476 versus 32.062 dB** and product-ROI LPIPS **0.174 versus 0.058**. Qwen's product-ROI LPIPS is 0.053. Thus the small version wins whole-image pixels while visibly losing object fidelity and material detail. Its object-mode preserved-region MAE is 0.00161 versus large 0.00914, confirming that it maintains unchanged pixels much more tightly. This does not imply cheating or literal target paste by the small model; it reflects its identity-biased residual path and the full-frame metric's weighting.

For **background replacement**, small clean SIM has the lowest full LPIPS (0.129; large 0.153, Qwen 0.258, pix2pix 0.264). For **joint replacement**, large clean SIM is best on full LPIPS (0.161; small 0.215). There is no single quality winner across all editing modes.

## Three-task, region-aware comparison

Each task has **768 fixed TEST pairs**. These are means over all pairs, not selected examples. Lower is better for LPIPS, DISTS, and MAE. The four clean-SIM rows are comparable within a task; the small-EXP row is a separate physical result using a **different electronically tuned weight**. All values here were recomputed from saved 8-bit 256×256 PNGs. The full, unrounded audit is `three_task_scores.csv`, joined from `summary_lpips.csv`, `summary_dists.csv`, and `summary_regions.csv`; `per_image_regions.csv` contains all 11,520 model-image rows.

**Background replacement** — edit = background; preservation = original product. The perceptual metric is a *custom background-masked* LPIPS/DISTS, computed after neutral-filling the 12-pixel-dilated product union in both output and GT; it is not the standard full-frame LPIPS/DISTS.

| Model | Edit PSNR ↑ | Background-masked LPIPS ↓ | Background-masked DISTS ↓ | Preserved product MAE ↓ |
| --- | ---: | ---: | ---: | ---: |
| Small SIM | 34.921 | 0.115 | 0.182 | **0.0061** |
| Large SIM | 30.765 | **0.098** | **0.110** | 0.0376 |
| Qwen28 SIM | 23.739 | 0.213 | 0.141 | 0.0423 |
| pix2pix-Turbo SIM | 22.909 | 0.218 | 0.134 | 0.0822 |
| Small EXP, tuned weight | 30.448 | 0.168 | 0.200 | 0.0067 |

**Object replacement** — edit = dilated source/target product union; preservation = outside that union. Product-ROI LPIPS/DISTS uses the union bounding box resized to 256×256. “Old footprint” is the source product area at least four pixels outside the target product, compared with GT; it is a ghosting *proxy*, not proof of a visible old-object remnant. The mask is nonempty in 754/768 pairs.

| Model | Edit PSNR ↑ | Product-ROI LPIPS ↓ | Product-ROI DISTS ↓ | Preserved background MAE ↓ | Old-footprint MAE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Small SIM | 31.476 | 0.174 | 0.184 | **0.0016** | 0.0203 |
| Large SIM | **32.062** | 0.058 | 0.089 | 0.0091 | **0.0155** |
| Qwen28 SIM | 31.112 | **0.053** | **0.080** | 0.0096 | 0.0165 |
| pix2pix-Turbo SIM | 17.151 | 0.430 | 0.263 | 0.0246 | 0.1368 |
| Small EXP, tuned weight | 30.964 | 0.173 | 0.187 | 0.0030 | 0.0243 |

**Joint replacement** — the entire frame may change, so there is **no preserved region** to score. Product and background perceptual scores are diagnostics alongside full-frame scores; do not present either as a preservation metric.

| Model | Full PSNR ↑ | Full LPIPS ↓ | Full DISTS ↓ | Product-ROI LPIPS ↓ | Background-masked LPIPS ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Small SIM | **32.264** | 0.215 | 0.244 | 0.225 | **0.088** |
| Large SIM | 30.081 | **0.161** | **0.154** | **0.138** | 0.089 |
| Qwen28 SIM | 24.104 | 0.271 | 0.193 | 0.191 | 0.193 |
| pix2pix-Turbo SIM | 19.172 | 0.405 | 0.245 | 0.441 | 0.178 |
| Small EXP, tuned weight | 28.684 | 0.265 | 0.257 | 0.249 | 0.137 |

The comparison should be **within each task and metric column**, not between background-masked, ROI, and full-frame perceptual scores. For background replacement, small SIM preserves the product best but large SIM has the best edited-background perceptual scores. For object replacement, small SIM preserves the background best, whereas large/Qwen have much better new-product perceptual scores. For joint replacement, large SIM leads on full-frame perceptual quality despite small SIM's higher PSNR. This is why a single pooled PSNR over the three tasks is misleading.

## Metric definitions

- LPIPS-Alex: `torchmetrics` 1.9.0, `LearnedPerceptualImagePatchSimilarity(net_type='alex', reduction='none', normalize=True)`; model output and GT are RGB `[0,1]`; per-image scores are averaged. Official pre-trained AlexNet weights `alexnet-owt-7be5be79.pth`.
- DISTS: `DISTS-pytorch` 0.1 with packaged learned alpha/beta and torchvision ImageNet VGG16 weights. Inputs are RGB `[0,1]`; per-image scores are averaged. Lower is better.
- Product ROI: bounding box of the union of source and target product alpha masks after 12-pixel dilation, cropped from the unaltered PNG and bilinearly resized to 256×256 for feature metrics. This includes ghost edges outside the new object. Evaluation masks are used **only for scoring**, never given to any model at inference.
- Edited-region PSNR: per-image MSE only in the evaluation edit mask, converted to dB, then averaged. For object mode, the mask is the dilated source/target product union; for background mode, the complement of the source product mask; for joint mode, the whole image. `preserve_region_mae` measures the complement (or the source object for background mode), in `[0,1]` RGB units.
- Region audit: `eval_regions.py` reconstructs source/target alpha masks from TEST metadata, then applies one common mask to all model outputs and GT. Background preservation MAE excludes the 12-pixel-dilated product union; background-edit perceptual scores fill that union with neutral gray before LPIPS/DISTS. The outer-boundary MAE and old-footprint MAE are additional diagnostics in the CSV. Evaluation alpha masks are never inference inputs.
- Full PSNR: per-image full RGB MSE on `[0,1]` converted to dB, then averaged. Quantized PNGs yield small differences from the earlier pre-export tensor reports.

## Figures

- `annotated_{background,object,joint}_clean_sim.png`: matched clean-SIM comparison, recommended for paper draft.
- `annotated_{background,object,joint}_with_exp.png`: same plate with a clearly marked EXP diagnostic column; do not present it as an equal-condition simulation comparison.

The panel headers show the 768-sample mode mean perceptual scores. Each generated image is annotated with its own PSNR and LPIPS/DISTS; the object plate uses edited-region PSNR and product-ROI LPIPS/DISTS. Figures and raw CSVs should be kept together so scores can be checked against individual images.

## Limits

No blind human-preference score is claimed. That requires independent raters and a predeclared sampling/ranking protocol. Standard FID/KID was attempted but the required torch-fidelity pre-trained Inception weights were absent and the official weights host disconnected; **no substitute Inception weights or fabricated FID/KID numbers are reported**. Even if later available, FID/KID would be distribution-level supplementary metrics, not paired edit-success scores.

Reproduction scripts used on the server are copied here as `eval_lpips.py`, `eval_dists.py`, and `eval_regions.py`; `assemble_three_task_scores.py` joins their summaries. The fixed figure builder is `build_annotated_figures.py`. GPU analysis ended after writing CSVs; no optical hardware was accessed.
