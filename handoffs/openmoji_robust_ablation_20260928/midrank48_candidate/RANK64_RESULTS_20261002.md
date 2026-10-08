# OpenMoji rank-64 — verified results as of 04:55 Beijing

Same original rank-64 architecture throughout; only the existing final decoder is adapted. All upper modules, optical phases and alpha remain protected. No added branch or post-hoc layer.

| Version | Original normal simulation, bench CPU | Real CCD Changed-cell accuracy | Preserved cells | Entire scene exact |
| --- | ---: | ---: | ---: | ---: |
| G1 ideal / G2 no trick, same PT | 93.90% | G2 61.85% | See G2 report | See G2 report |
| G5 robust, no physical adaptation | 92.70% | 69.10% | 98.7396% | 47.90% |
| G5 original decoder, TRAIN1000, selected epoch85 | Not yet re-audited for adapted PT | 87.70% | 98.6102% | 59.00% |
| Same decoder, existing edit bias calibrated | Not yet re-audited for adapted PT | 89.00% | 98.0465% | 54.40% |

Normal simulation above uses the same bench CPU path. Server G2/G5 clean simulation .9385/.9290 are a separate path, not substituted as identical results. Original G1 .939 gives relative 5% floor .89205, relative 1% floor .92961. Current .8900 falls short of both. Original G1 reference is not replaced by any adapted PT's changed simulation score.

Each original G2/G5 TEST has six stages ×1000 true CCD frames with matching receipts. G5 independent original TRAIN1000 also has6000 true CCD frames. TEST selects checkpoints every5 epochs and selects bias; TEST never supplies gradients. These are development-selected scores, not independent generalization estimates.

Calibration increases Changed-cell but lowers scene-exact .590→.544 and preserved cells .986102→.980465. Both PTs are retained, not silently replaced.

## Sealed local checkpoints

- `rank64_decoder_testselected_best_20261002.pt`: SHA256 `8011f7e4ac03d2c5894a72b975c26e4bea8c36803c3719c088d76a3ca2b3204f`.
- `rank64_bias_calibrated_best_20261002.pt`: SHA256 `df3fc8211e57667384398f97fc5afb87c3641ceaba47ee21121517cbe3097a49`.
- Local hashes checked against sealed bench reports; calibrated serialized PT strict reload uses default .5 gate and reproduces .8900.

Corresponding reports: `rank64_g2_full_physical_report_20261002.json`, `rank64_g5_full_physical_report_20261002.json`, `rank64_decoder_testselected_report_20261002.json`, `rank64_bias_calibration_report_20261002.json`.

## In progress, not results

Additional source-disjoint TRAIN1000 ×six stages is being captured using unchanged G5 optical weight and settings. After stage-count/receipt/phase/signal audit and SDK release, the one-shot queue will start the same original decoder on combined TRAIN2000, TEST every5 epoch selection. No performance improvement is yet claimed for that run.

ABO is sealed and unchanged. External upload remains paused.
