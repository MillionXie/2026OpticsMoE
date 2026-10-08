# Row 04 — Ours identity and timing scope

The performance identity is the 10 cm `optical_text_to_image_64_10cm_compact_e0p5_target088_seed43_20260923` checkpoint, with full-test R@1 = 0.8800 and SHA-256 recorded in `checkpoint_sha256.txt`.

The formal latency follows the table-wide six-pass multimodal convention. The 6 × 1.0447 ms physical interval is combined with the `abo_text_to_image` narrow CUDA Event mean from `paper_narrow_summary.csv`. The preceding tasks in this same process preserve the table-wide no-explicit-warm-up, first-call-retained sequence; the isolated cold-start diagnostic was rejected and is not included here.
