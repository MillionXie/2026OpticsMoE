# LGVQ temporal-quality A100 multi-baseline demo

This directory is the paper-facing replacement for the earlier mixed-task comparison table. It compares LightGenV2 with Qwen3-VL-2B, DeepSeek-VL2-Tiny, CLIP ViT-B/32 and YOLO11s on LGVQ temporal quality.

## Formal comparison contract

- Device: one NVIDIA A100-PCIE-40GB, physical GPU index 6.
- Timing sample: the first 200 test videos in the fixed LGVQ manifest order.
- Batch sizes: 1, 2, 4 and 8 videos. Processing 16 videos therefore requires 16, 8, 4 and 2 calls.
- Warm-up: none. The first measured call is retained in every mean and P95.
- Boundary: first native backbone block input to the scalar temporal-quality score ready on GPU.
- Reported latency: synchronized wall time. CUDA Event timings remain in the raw files for audit.
- Reported performance: fixed full-test SRCC from the corresponding trained checkpoint/protocol. Batch size does not change the displayed SRCC.
- Baseline energy for 16 videos: `(338.2 W host + measured active A100 board mean) × synchronized-wall latency for 16 videos`.
- Ours uses the previously audited 16-video optical-MoE point: SRCC 0.8044, 10.523785 ms and 1.206688 J.

All four baseline batch settings use 200 distinct videos. Every batch setting starts in a new Python process so CUDA/model state is not inherited from another batch setting.

## Inputs

- Qwen3-VL-2B: four 448×448 frames and the trained five-quality-token readout.
- DeepSeek-VL2-Tiny: four 384×384 frames and an equivalent five-row scalar readout.
- CLIP ViT-B/32: four 224×224 frames; normalized 512-D features are concatenated to 2048-D before the five-row readout.
- YOLO11s: four 640×640 frames; layer-10 features are globally averaged, concatenated to 2048-D and read by five rows.
- For CLIP, DeepSeek and YOLO, frames are sampled at 10%, 37%, 63% and 90% of the video and a 65% centered square crop is resized to the model input.

The timing profilers instantiate the same readout shape used by the trained baseline. Formal SRCC is read from the fixed full-test result rather than recomputed from randomly initialized timing-only readout weights.

## Files

- `LGVQ_temporal_multi_baseline_A100.xlsx`: reader-facing table plus formula-driven audit columns and sources.
- `calculated_summary.csv` and `.json`: machine-readable derived results.
- `srcc_vs_latency_16videos.*`: SRCC versus 16-video latency.
- `srcc_vs_energy_16videos.*`: SRCC versus 16-video energy.
- `raw_remote/`: all per-call timing, power samples, predictions where available, and original JSON reports.
- `build_analysis.py`: reproducible derivation and plotting script.

## Audit event

During the first Qwen batch-2 attempt, PID 4183466 owned by `guest0` entered GPU 6. That attempt was stopped and excluded. The process later exited normally. Qwen batch 2/4/8 and the final synchronized-wall rerun were measured again with the A100 exclusive. No result from the contaminated attempt appears in `calculated_summary.*`.

DeepSeek has a large first-call lazy-initialization cost. It is intentionally retained because the protocol specifies no explicit warm-up and includes the first test item.
