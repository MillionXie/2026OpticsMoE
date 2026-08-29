# Caltech101 four-layer CCD-noise robustness study

This is a weights-only continuation of the 81% warmstart5 model. It changes no
tensor topology and does not retrain Qwen from scratch. Four controlled arms
study training-time, pixelwise, truncated biased Gaussian CCD noise. All arms
use a 1% learned optical-fusion floor and deliberately stronger phase training.

See `RUN_COMMANDS.md` for the exact commands and comparison contract.

