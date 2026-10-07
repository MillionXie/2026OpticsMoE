# Training assets (not committed)

Place full-training inputs under `assets/` and update only the path fields in
`configs/temporal_16x4_s163.yaml`:

| expected path | purpose |
|---|---|
| `assets/manifest/lgvq_train2250_test558.csv` | fixed train/test sample order and MOS |
| `assets/cache/qwen3vl_front_4f_49x1024_quality14.pt` | four-frame visual and quality tokens |
| `assets/cache/qwen3vl_front_temporal_prompt_2048.pt` | fixed Temporal prompt tokens |
| `assets/cache/training_only_teacher_predictions.pt` | train-only soft targets |
| `assets/initialization/multivideo9x4_best_checkpoint.pt` | shape-compatible 9×4 warm start |

These assets are not needed by the fixed-weight teacher ZIP, which contains all
35 packaged test fields.  Do not use the packaged test fields as training data.

Known retained training identities:

| asset | SHA-256 | bytes |
|---|---|---:|
| manifest | `607c50d20662a47795c23cd2038081b300368ed108ba9be8a1bb8a9d250f8e7fc` | 546767 |
| train-only teacher predictions | `b3224632202d4c5677c7e859e88080d35fa01a29547ea9fbc018b2313490c558` | 189165 |
| 9×4 initialization checkpoint | `cfda5cd8cbad94d0060f54f4c07cf4314b84677c45944fdff196d945efa32ba0` | 81800352 |

The two original monolithic Qwen-front cache files were not found at their
recorded server paths during this cleanup.  Their exact 558-video test tensors
are retained field-by-field in the teacher ZIP, so fixed-weight evaluation is
complete; fresh full training remains blocked until the train caches are
restored or regenerated.  This limitation must not be hidden in a public
reproduction claim.
