# Qwen3-VL Temporal 4/9/16/36/49-frame timing report

## Main result

Every reported latency is for one complete video and one Temporal prompt,
not for one frame. N is the number of unique LGVQ test videos.

| Frames | N | Mean total ms/video | Median | P95 | Decode | Processor | H2D | Qwen | Linear | video/s | Peak GiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 558 | 134.058 | 95.009 | 378.928 | 89.364 | 4.917 | 1.657 | 37.895 | 0.214 | 7.459 | 4.052 |
| 9 | 558 | 278.559 | 204.352 | 737.544 | 188.412 | 21.985 | 10.451 | 57.458 | 0.242 | 3.590 | 4.169 |
| 16 | 558 | 476.591 | 344.483 | 1277.810 | 348.603 | 36.535 | 8.137 | 83.054 | 0.252 | 2.098 | 4.283 |
| 36 | 558 | 1133.494 | 832.388 | 2776.558 | 853.029 | 81.342 | 14.105 | 184.746 | 0.259 | 0.882 | 4.678 |

## Conditional 49-frame rule

The predeclared threshold was mean total latency <= 910.0 ms/video.
The measured 36-frame mean was 1133.494 ms/video.
49-frame status: **skipped because the 36-frame mean exceeded 910 ms**.

## Statistical definitions

- `N`: unique test videos measured once at that frame count.
- `Mean`: arithmetic mean across all N videos; it is the threshold metric.
- `Median`: the middle latency; half the videos are faster and half slower.
- `P95`: the 95th percentile; 95% of videos finish no slower than this value, while the slowest 5% exceed it.
- `Decode`: MP4 open plus one random seek/read for every requested frame, including crop and resize.
- `Total`: Decode + processor/tokenizer + H2D + full frozen Qwen + pool/linear + small Python glue overhead.

The benchmark uses batch size 1, no sequential decode, no parallel decode,
no frame/feature cache, no quantization, and no torch.compile.
