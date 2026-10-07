# T03/T08 A100 completion audit (candidate only)

This directory preserves the first 200-call completion audit for the current
ABO image-to-image and text-to-image versions.  It is **not a formal paper
result yet** because physical A100 GPU 6 also contained the foreign process
`guest0/PID 621376` (680 MiB).  The `guest3` account could neither signal that
process nor use sudo.  Every JSON report retains the process audit.

## Bound model identities

- T03 image-to-image: `physical_bounded30_spatial_consistent_20260927/best.pt`,
  SHA-256 `0b1ff5b682b833f8c4fd25e6330147e993e974272d7d2812fa679082ef0e87bc`,
  R@1 = 0.8225.  The measured boundary includes the current V5/L5 electronic
  context, `spatial2x2_64` readout, 1600-gallery cosine scoring, and stable full
  ranking.
- T08 text-to-image: `20260927_language_balance_input/last_checkpoint.pt`,
  SHA-256 `5b4f9a37f19ce95cf23e4b874badc8e88955559553d47929fb80cd3cbf527cac`,
  full-test PSNR = 34.2778 dB.  The 17,026,642-parameter version including its
  spatial-prior decoder is used.

## Correct six-pass timing contract

Both models use exactly six optical passes: vision router/expert/global and
language router/expert/global.  One pass is
`0.714 + 0.300 + 0.0307 = 1.0447 ms`; therefore the fixed physical time is
`6 × 1.0447 = 6.2682 ms`.

No explicit warm-up was used.  The first call and all 200 calls are retained.
Ours uses CUDA Event timing for serialized GPU operations; parallel electronic
residual branches are excluded.  Baseline uses synchronized wall timing from
the first native transformer block through the final task output.

## Candidate results (do not cite until exclusive rerun)

| Task | Metric | Ours electronic mean | Physical | Ours total | Baseline mean | Ours energy | Baseline energy | Speedup | Energy-efficiency gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ABO image-to-image | R@1 0.8225 | 6.4442 ms | 6.2682 ms | 12.7124 ms | 44.9299 ms | 1.4348 J | 19.045 J | 3.53× | 13.27× |
| ABO text-to-image | PSNR 34.2778 dB | 17.2228 ms | 6.2682 ms | 23.4910 ms | 87.0956 ms | 2.5583 J | 40.900 J | 3.71× | 15.99× |

Energy uses the agreed equations:

`E_ours = 6.2682 ms × (80.358 + 41.388) W + t_electronic × (41.388 + 62.842) W`

`E_baseline = t_baseline × (338.2 W + measured A100 board power)`

T03 reuses the unchanged, previously exclusive Qwen baseline audit (active
board power 85.7593 W).  T08 candidate baseline board power is 131.444 W and
must also be repeated after the foreign process exits.

Raw per-call timing, power telemetry, identities, commands, and hashes are in
`raw_candidate/`.  The candidate directory must remain separate from the final
exclusive-GPU rerun.
