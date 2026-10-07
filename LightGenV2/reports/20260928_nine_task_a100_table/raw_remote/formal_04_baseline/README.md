# Row 04 — frozen Qwen text-to-image retrieval baseline

- Protocol: 100 official ABO titles query a precomputed 2,400-image held-out gallery; the 100-query set is repeated once to obtain 200 timed calls.
- Performance: R@1 = 0.8200, R@5 = 0.9000, R@10 = 0.9600.
- Timing boundary: first Qwen language transformer block input through normalized 2048-D query embedding, cosine scoring, and stable ranking.
- Timing statistic: synchronized-wall mean, no explicit warm-up, first call retained.
- Power: physical A100 GPU 6 `nvidia-smi power.draw`, sampled every 10 ms during active calls.
- The process listed in `process_audit_after` is the profiler itself, observed before Python exited; `process_audit_before` is empty and no foreign A100 process participated.

See `baseline_timing_per_sample.csv`, `baseline_power_samples.csv`, and `baseline_report.json` for raw values.
