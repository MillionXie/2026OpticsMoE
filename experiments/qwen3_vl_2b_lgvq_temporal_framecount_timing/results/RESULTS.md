# Qwen3-VL temporal frame-count timing result

The reported total is for **one video and one Temporal prompt**. It starts
before opening the MP4 and stops after the scalar prediction reaches CPU.
Model loading and warmup are recorded separately and excluded.

| Frames | Total mean (ms/video) | Total median | Total p95 | Decode | Processor | H2D | Qwen backbone | Pool+Linear | video/s | Peak GiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 134.058 | 95.009 | 378.928 | 89.364 | 4.917 | 1.657 | 37.895 | 0.214 | 7.459 | 4.052 |
| 9 | 278.559 | 204.352 | 737.544 | 188.412 | 21.985 | 10.451 | 57.458 | 0.242 | 3.590 | 4.169 |
| 16 | 476.591 | 344.483 | 1277.810 | 348.603 | 36.535 | 8.137 | 83.054 | 0.252 | 2.098 | 4.283 |

- GPU: `NVIDIA GeForce RTX 5090 D`
- Model load time: `1.330 s`
- Batch size: `1 video`.
- Decode: original repeated random-seek implementation; no sequential-decode optimization.
- Qwen feature caching, frame caching, quantization and `torch.compile` are disabled.
- The linear layer is present only to preserve the baseline output boundary; its numerical score is ignored in this timing audit.
