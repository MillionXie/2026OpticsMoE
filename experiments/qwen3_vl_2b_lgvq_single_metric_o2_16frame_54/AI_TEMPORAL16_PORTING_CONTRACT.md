# Temporal-16 hardware bridge porting contract

This file is written for an AI/code maintainer. The current checkpoint class is
`LGVQSingleMetricOEO16`; its exact input tuple is
`(vision_tokens[B,16,49,1024], quality_tokens[B,16,49,14],
language_tokens[1,T,2048], language_mask[1,T])`. Never load the checkpoint into
the older `LGVQFourStageOEO` class.

Port only the session/acquisition mechanics from the sibling project's
`hardware_bridge.py` and `hardware_contract.py`. The exact Temporal-16 forward
must be split at these boundaries:

1. Build `prompt_tokens`, prompt-conditioned `vision`, `fields1`, and
   `electronic1` exactly as lines in `LGVQSingleMetricOEO16.forward`.
2. `vision_router`: play the 16 `fields1` in centered 4x4 lanes. Convert the
   measured canonical CCD into `[B,16,4]` energies using
   `parallel_router_intervals`, then the same centered/RMS logits, temperature,
   softmax and sparse Top-2 as `OpticalRouterParallel16`.
3. `vision_expert`: lay out 64 fields with the measured router weights. Decode
   each measured 114x114 lane with `parallel_optics._normalize` and
   `expert_readout`, reshape `[B,16,49,192]`, and RMS-fuse with `electronic1`.
4. Build `fields2` and `electronic2`; `vision_global` uses centered 54x54 input
   in each lane, then `global_readout` and the second RMS fusion.
5. Build `image_tokens`, concatenate prompt tokens, create the exact mask and
   sequence position values, then build `fields3` and `electronic3`.
6. `language_router`: measured four ROI energies produce one `[B,4]` Top-2
   decision. `language_expert`: use 2x2 109x109 layout, decode only the actual
   sequence length, mask, and RMS-fuse with `electronic3`.
7. Build `fields4` and `electronic4`; `language_global` uses the centered
   109x109 field, then decode actual sequence length, mask, RMS-fuse, and call
   the exact `TemporalReadout`.

All measured CCD readers must reproduce model normalization, not invent a new
preprocessing path. All measured passes must be a contiguous prefix of the six
physical passes. Seal one session manifest containing 2250 train and 558 test
sample IDs/cache indices; reject missing, duplicate, renamed, or hash-mismatched
frames. Select the hardware-finetuned checkpoint by the highest test SRCC at a
5-epoch interval, as requested by the experiment owner.

The best-mask export report is the authoritative geometry/orientation source.
The sibling bridge is authoritative only for hardware safety checks, canonical
CCD naming, hash manifests, prefix enforcement, and checkpoint-chain auditing.
