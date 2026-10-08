# OpenMoji overnight electronic adaptation — deadline 2026-10-02 09:00 Beijing

User finalized ABO rank72 checkpoint SHA `25f23260864a00a0f32b5de209c27f5876166018c8d8ab0e880de81e0d2c2c22`; leave its source, weights and CCD untouched. External uploads remain paused.

OpenMoji goal remains physical Changed-cell accuracy around **0.855**, not a manufactured no-trick baseline. G5 rank48 is the robust-trick candidate; G2 rank48 is the no-trick control. Current prior bests: G5 TEST-selected .7975; G2 focused decoder TEST-selected .8085. Do not conflate groups.

## Actual offline experiment

- Bench task: `OpenMoji_Electronic_Overnight_1002`.
- Bench runner: `E:/code/guest/2026OpticsMoE/OpenMoji_Robust_Midrank48_G2_SHS_20260930/overnight_electronic_suite_20261002.py`.
- Log: same project `overnight_electronic_suite_20261002.log`.
- Output: `E:/code/guest/2026OpticsMoE/OpenMoji_Electronic_Overnight_20261002`.
- Sequential four trials: `g5_continue`, `g5_residual64`, `g2_continue`, `g2_residual64`, each 120 epochs. One non-A100 RTX4060; no SDK or hardware calls.
- All inputs are previously audited **real CCD decoder-input caches** from each respective checkpoint, not simulation and not mixed masks. Original optical/front-end/router/alpha/pre-decoder state remains frozen and SHA-protected.
- TRAIN800 gradients; VAL200 recording only; user-authorized TEST1000 every5 epochs selects the highest checkpoint. TEST never enters gradient computation. Scores are TEST-selected development metrics, not independent held-out performance.
- Continuation: original decoder, AdamW lr1e-5. Extended decoder: same input, GroupNorm→1×1 rank64→depthwise3×3→1×1 zero-initialized residual before original decoder; base lr1e-5, adapter lr1e-4. Not a new image branch. Feature jitter .015 relative RMS on TRAIN only, changed-cell-focused loss, preserved-cell .2CE, edit BCE and .05 anchor.
- Zero-init adapter equivalence assertion, protected upstream hash, strict saved checkpoint reload and score alignment are built into the runner. `best_full.pt` exists only if there is an improvement over the starting checkpoint; otherwise the starting PT remains best. `last_decoder.pt`, history and report retained.

First launch safely stopped before CUDA initialization because WDDM listed desktop apps as compute processes. Confirmed the PIDs were DWM/Explorer/Edge/ToDesk etc., no Python train job, GPU memory324MiB. Guard now requires <512MiB and no other Python/train process. Empty trial directory was preserved and retried; no trial results overwritten.

Windows venv launcher and its child interpreter were then identified and allowed as the same job, not two competing tasks. Latest run confirmed actual G5 gradients through epoch4, CUDA memory772MiB, PIDs25124/16048 (launcher/interpreter). No SLM/CCD task. This supersedes initial safe exits; inspect current log for subsequent progress.

## Extension deployment requirement

An extended PT is **not** loadable by merely replacing the original PT. Build the original rank48 model, call `install_extension(model, payload['electronic_extension'])` from this delivered source, then `model.load_state_dict(payload['model'], strict=True)`. The optical path and cached CCD are unchanged. For continuation PT the metadata is null and original loading works.

## Supervision and delivery

Check actual task state, PID, progress, log and GPU before acting; do not restart completed trials or overwrite outputs. Diagnose errors in scope, preserve useful outputs; do not rely only on task Running status. Scheduled thread monitoring every30min until09:15 and one09:00 delivery wake are active. Quiet for ordinary progress, notify meaningful result/failure/user action only.

At 09:00 deliver actual best G5 and G2 separately, code/weights/reports/curves/SHA and whether target was reached. Preserve original lower-capacity experiment to make capacity changes explicit. No external upload without new authorization. If all trials finish early, inspect fit versus TEST and operation errors before a justified follow-up; do not blindly repeat identical long runs.
