# TRAIN-only microbatch throughput admission

The user requests efficient foreground use of up to five shared non-A100 GPUs.
Four source495 methods already run on0/1/3/4; do not edit, interrupt or move them.
Use otherwise idle authorized3090GPU2 for one bounded benchmark and release it.

One retained six-layer model, one fixed TRAIN macro16, no optimizer/EMA update,
no VALID/TEST images, no new checkpoint. Warm the existing FFT plans; compare
micro2/4/8 with two timed repetitions each of the actual source495 prototype-both
loss and gradients, keeping order, seed, MixUp, masks, teacher and all loss weights
identical. CPU independently checks analytic accumulated gradients before dispatch.
Measure peak allocation under unchanged2GiB+800MiB policy. Any allocation failure
rejects that micro size, stops larger sizes, and is retained without retry. Numerical
mismatch is a real failed diagnostic; preserve its claim/failure, never relax it.

Compare every trainable gradient against micro2 (relative<=2e-4), probabilities and
capture<=3e-6, loss and exact state SHA; all weights remain unchanged. Save raw
timings and device identity. A local3090 macro speedup does not prove full4090
epoch speedup. This is performance admission, not an accuracy result or audit of
old runs. Any future larger-micro training still needs its own all20/tail5 production
gate, new fixed profile/source and saved science. L2 and all controls untouched.
