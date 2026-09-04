# Qwen3-VL Temporal 4/9/16/25/36/49-frame timing report

Every latency is for one complete video and one Temporal prompt, not one frame.
N is the number of unique fixed-split LGVQ test videos.

| Frames | N | Mean total ms/video | Median | P95 | Decode | Processor | H2D | Qwen | Linear | video/s | Peak GiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 558 | 134.058 | 95.009 | 378.928 | 89.364 | 4.917 | 1.657 | 37.895 | 0.214 | 7.459 | 4.052 |
| 9 | 558 | 278.559 | 204.352 | 737.544 | 188.412 | 21.985 | 10.451 | 57.458 | 0.242 | 3.590 | 4.169 |
| 16 | 558 | 476.591 | 344.483 | 1277.810 | 348.603 | 36.535 | 8.137 | 83.054 | 0.252 | 2.098 | 4.283 |
| 25 | 558 | 773.266 | 565.622 | 1929.744 | 557.863 | 62.112 | 10.604 | 142.432 | 0.243 | 1.293 | 4.479 |
| 36 | 558 | 1133.494 | 832.388 | 2776.558 | 853.029 | 81.342 | 14.105 | 184.746 | 0.259 | 0.882 | 4.678 |
| 49 | 558 | 1532.186 | 1144.709 | 3732.220 | 1154.196 | 106.086 | 18.664 | 252.985 | 0.243 | 0.653 | 4.945 |

The earlier conditional rule skipped 49 frames because the 36-frame mean exceeded
910 ms/video. This follow-up measures 49 frames because the user explicitly requested it.

The benchmark retains batch size 1 and the original per-position random seek/read.
No sequential decode, parallel decode, frame/feature cache, quantization,
torch.compile, or manual FlashAttention override is used.
