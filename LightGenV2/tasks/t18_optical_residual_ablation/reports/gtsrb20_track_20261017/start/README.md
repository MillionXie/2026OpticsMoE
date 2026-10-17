# GTSRB20 grouped-sign depth experiment: actual start

Human requested resolving the Caltech generalization bottleneck today (2026-10-17), leaving at least a day for optical experiments before Sunday evening. Kather is explicitly excluded because used in the paper's depth study; other paper-planned datasets are excluded. The current registry/scorecard/protocol search had no GTSRB. This is the sole new traffic-sign development task, not a full43-class benchmark or a promise of ten-point residual gains. Caltech accepted weights remain protected fallbacks. No scheduled task is used.

## Actual data and graph

TRAIN4000, VAL2000, TEST10140; twenty original classes selected by author competition-training counts only, numeric ID tie-break. TRAIN200/VAL100 per class, fixed sampling across assigned sign tracks, capped for today's compute budget. Full selected-class official final TEST retained. Whole physical sign tracks, merged by exact cropped-pixel components, assigned80/20 TRAIN/VAL before sampling; no cross-role exact cached pixels. No TEST score was used for selection. Every role uses the provided ROI inclusive crop andbicubicRGB150. This competition Training_fixed package is not claimed to be the larger Final_Training package. Source images remain private.

Author benchmark: https://github.com/houbensebastian/GermanTrafficSignBenchmarks#GTSRB . Author archive: https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/ . Archive MD5s were checked against the primary torchvision loader and ZIP CRC checks passed. Author research download is available; no unverified CC license assertion or image redistribution.

Cache SHA256 `9f381f7e781aab7f488ed2d1b159ceec7f9a31a418c1dcd5bc728870273f6560`; manifest SHA256 `937035c1f10eead99d88dd5e2e5879cd28f36582918e0d5ef711f8e7eae53ee0`. See class_support.csv for all twenty supports. Initial input QA inspected forty TRAIN crops, TEST0/model0; a first preview-only aspect stretch was retained privately and corrected to square display. The cache/model inputs remained unchanged.

Same current20-class inference graph: router once, E-G repeated1/2/3 for depths2/4/6; four224 experts/global478/RGB100/joint8bit/OEO. Phase479364/908552/1337740; identical learnedalpha(.7001,.95), per-stage9-weight electronic residual and5140terminal linear; total electronics5160/5180/5200. Alpha is a normalized feature coefficient, not global optical power. Both arms train identical electronics; only coherent opticalrho.3/0 differs. No new inference head or network. Horizontal flips are disabled for directional signs.

## Actual admission/start

Published training source `6fe6d5924bbf2af70307e2ff1b3c8edca19c8530`; all283 old sources unchanged and new289-source closure verified. CPU admission first passed, SHA256 `dddfcf837d34d0c651d1737159cee48440c89bc45311af07c1ea59b26b4f0b5e`, twelve real/synthetic forwards across all six depth/rho models, exact ABO fusion/gradient parity, black20uniform, counts/trace/NumPy readout/data ownership verified, optimizer0/TEST0. Each of four actual producers separately passed its own all20/macro16/tail16 ordinary/dropout micro1vs4 gate. Initial cards areGPU0=L6residual, GPU4=L4residual, GPU2=L6control, GPU5=L4control; L2 pair is queued. No A100 T18 process, no foreign process interruption.

Run `gtsrb20_track_current246_s17_equal100_20261017`; supervisor PID3304730 is historical identity, not a claim it remains alive indefinitely. Root actual state at snapshot `fit_running`. First actual epochs show learning and are not final models/results; TEST model evaluations are still0 at this snapshot.

## Frozen protocol and acceptance

Each of six fresh arms completes100/25000Adam, effective16/micro4/tail16, identical seed17/init/order/affineMix/masks within depth. EMA.99/MixUp.3/LS.05/drop.1/TV.02/capture.05/AdamW.002 cosine to.0002/electronicdecay.01. Clean fullTRAIN/VAL every epoch; first minimum validation EMA balancedNLL selects. Maximum14400fitsec is a fault guard; incomplete100 cannot be accepted or tested. Actual throughput on shared cards controls completion time; no timing guarantee.

Each original producer must exitrc0; own exclusive independent source/config/rawEMA/init/25000Adam/plans/cosine/firstminimum/saved IDs-labels-CM-NLL/pooling-linear audit and matching completed SHA precede once-TEST per selected model. Own independent saved TEST math audit does no second forward. No old audit/output substitutes. Any failure preserves evidence and stops downstream, no automatic retry or weakened thresholds. No-residual controls freeze after100; residual followup requires actual current curves and a separate fixed profile/source/admission, not automatic speculative continuation.

Goals: L4residual TEST>.8, L6residual TEST>L4, preferably each gap10points. These are not guaranteed or manufactured by weakening controls or TEST-based class/epoch choice. Report TRAIN/VAL/TEST/macro/allclassrecall, actual/selected budgets and negatives. Subsequent TEST-informed development/adoption and unequal residual budgets must be disclosed. All original Caltech/data/PT/claims/failures protected. Source onlyGit; rawJSON/PT/IDs/logs private. No lab/native/formal/A100 action.
