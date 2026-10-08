# T12 unified image editing — result index

This is the curated entry point. Each preview is a selection of **complete,
unaltered test rows** from the original grids. Columns are always
`input | supervised target | model generation`; the left-hand text is truncated
in the original grids, so inspect the dataset code for full prompts.

## Recommended current pilots

| Variant | Status | Counted parameters | Resolution | Checkpoint on training server |
|---|---|---:|---:|---|
| Large, original 128-source data | Main quality reference; trained 2 epochs after structured width transfer | 142,528,245 | 256² output, but source files were 128² | `/DATA/DATA1/guest3/t12_assets/runs/abo_unified_qwenmini_143m_v1/unified_editor.pt` |
| Large, true-256 alpha-mask correction | Experimental anatomy fix; trained only 1 epoch; do not promote as universally better | 142,528,245 | 256² source/output | `/DATA/DATA1/guest3/t12_assets/runs/abo_unified_qwenmini_143m_true256_alpha_v2/unified_editor.pt` |
| Small Qwen-mini | Main compact reference | 13,711,128 | 128² output | `/DATA/DATA1/guest3/t12_assets/runs/abo_unified_qwenmini2_optical_13m_v1/best_model.pt` |

The large packed weights are not fully standalone: inference separately loads
the frozen SD-Turbo VAE, BK-SDM topology config, and shared Qwen token
embedding. The small weight also needs the shared Qwen token embedding.

## View by requested edit

| Edit | Large original | Large true-256 experiment | Small |
|---|---|---|---|
| Change background | [preview](large_128source_background.jpg) | [preview](large_true256_alpha_background.jpg) | [preview](small_128_background.jpg) |
| Change object | [preview](large_128source_object.jpg) | [preview](large_true256_alpha_object.jpg) | [preview](small_128_object.jpg) |
| Change both | [preview](large_128source_joint.jpg) | [preview](large_true256_alpha_joint.jpg) | [preview](small_128_joint.jpg) |

The large previews contain lamp and table examples for each edit. The small
joint and object previews show visible shape/edge artifacts; those failures
must remain part of the presentation. The true-256 experiment preserves thin
lamp geometry better but can still be visibly soft.

## Evaluation records

- Large original test latent MSE: 0.04312; background 0.05947, object 0.01370,
  both 0.03986. Details: [training summary](../large_143m_training_summary.json).
- Large true-256 alpha experiment test latent MSE: 0.05480; background
  0.07987, object 0.01524, both 0.04424. This uses a different target
  distribution, so it is **not** a clean numeric comparison to the original.
  Details: [training summary](../large_143m_true256_alpha_training_summary.json).
- Small test pixel MSE (pixels normalized to [-1, 1]): background 0.001107,
  object 0.000914, both 0.001885. FID-like 47.23 / KID 0.01086 ± 0.00172
  use ImageNet-trained torchvision Inception-v3, **not canonical FID**. The
  four fixed target designs make FID especially limited. Details:
  [small metrics](../small_distribution_metrics.json).
- No matched Qwen+decoder speed benchmark was completed for these unified
  weights. Do not put a speedup number on slides yet.

This is a narrow supervised product-editing pilot, not open-domain generation:
train/validation/test contain 1,152/128/128 ABO lamp/table source views; each
view generates eight paired tasks; target objects come from four fixed designs
and rooms from a procedural grammar. Masks are used to build supervised pairs,
not as inference inputs or pixel pasting.
