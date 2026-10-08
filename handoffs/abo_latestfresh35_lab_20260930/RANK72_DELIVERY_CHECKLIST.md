# Rank72 local delivery checklist

External upload is paused by the user. Do not deliver a partial capture as a completed experiment.

- Original checkpoint: rank72_epoch13_best_snapshot.pt; SHA256 25f23260864a00a0f32b5de209c27f5876166018c8d8ab0e880de81e0d2c2c22.
- Model: model_rank72_server.py. Include exact standalone training entry and dependencies from the isolated server source; do not substitute older rank192 model code.
- Bench inference: isolated candidate_latergb_rank72_20260930 pipeline, standalone model, layerwise_stage_sessions.py and physical flow snapshot. Remove credentials from any portable delivery.
- Original dataset: bench ABO_I2I_Lab_DVP_8um/data/images and protocol.json (local copy rank72_dataset_protocol.json); verify all 1600 gallery/TRAIN and 800 query image IDs resolve. Include processor assets and required tokenizer/config files.
- Valid CCD: six stages, each 2400 PNG plus matching receipt; exclude warmups and all rejected_dark directories. Currently only 5057 valid formal frames, so this requirement is NOT complete.
- Only if adaptation is required: include original and adapted PT, best/last/best_validation, TRAIN1200/VAL400 mapping, full epoch history and explicit TEST-development selection record. TEST checkpoint selection is authorized but is not independent testing.
- Include normal simulation .83375 / no-optics .8000 with source report, and actual new physical metrics only after full capture. Earlier rank192 physical .81125 must not be substituted.
- Build short-name archives only after integrity checks, preserve local sources, verify every archive and SHA manifest. Upload requires new user notification.

2026-09-30 08:48 UTC supervision: all rank72 tasks Ready; no capture process. Formal receipts router2400 valid, expert2400 valid, global257 valid +3 dark; later stages absent. Console Flags=0 persists. Locked desktop remains a hypothesis for display failure, not an established sole cause. No automatic diagnostic was repeated in this unchanged state.
