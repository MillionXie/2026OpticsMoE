# LightGenV2 execution rules

- Hardware acquisition defaults to **one complete dataset per optical layer**, in dependency order. Keep the model, SLMs and camera initialized across samples/layers where supported. Do not default to repeatedly executing all six layers for tiny sample batches. GPU batching is allowed without changing the optical layer order.
- Preserve and reuse verified same-checkpoint captures. Never mix masks, amplitude encodings, exposure/gain, geometry or camera orientation in one result. Switch execution strategies without deleting valid data or overlapping device processes.
- Estimate acquisition time from measured end-to-end throughput, not camera FPS alone; state the number of samples and total CCD captures separately.
- Simulation and amplitude BMP must use the same physical bounded amplitude; BMP quantization must not introduce an extra peak normalization absent from simulation.
- Save best/last only unless the user explicitly asks for intermediate checkpoints. Keep final metrics, dataset identities, masks, weights and useful analysis; do not delete valid results casually.
- Commit scoped code changes and synchronize GitHub plus the relevant server. If network synchronization fails, report it as pending rather than claiming completion.
