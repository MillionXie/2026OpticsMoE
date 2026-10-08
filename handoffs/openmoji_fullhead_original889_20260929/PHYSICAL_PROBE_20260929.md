# Full-head G2/G5 SHS probe (2026-09-29)

The four user-selected full-head weights are preserved under `final_four/`.
G2 and G5 SHA-256 values were checked before transfer to the isolated bench
project `E:/code/guest/2026OpticsMoE/OpenMoji_FullHead_Robust_SHS_20260929`.
Both checkpoints passed the six-stage optical-boundary and replay self-tests
(maximum logit differences 0 and 1.91e-6, respectively).

At the pinned 2000 us / Gain_X4 / 240 ms setting, the G2 and G5 four-sample
pilots each produced 24 CCD frames but the corrected p99 was about 23 across
all six stages. A CPU G2 pilot gave the same weak signal. A **previously
measured rank-16 reference checkpoint**, with the exact same `test_00000`
input and phase BMP SHA as its earlier bright capture, now gave vision-router
p99=23 instead of the earlier p99=235. Its CPU repeat remained p99=24.
The G2 full-frame raw camera diagnostic was also weak (vision-router p99=26,
maximum=61), so this is not merely a moved/cropped bright spot. The camera
frame ID advanced and both device SDKs connected, but software receipts alone
do not prove optical illumination/display. The input and phase reference BMPs
are retained at:

* `E:/code/guest/2026OpticsMoE/OpenMoji_FullHead_Robust_SHS_20260929/runs/reference_bmp1/amplitude/vision_router/test_00000.bmp`
* `E:/code/guest/2026OpticsMoE/OpenMoji_FullHead_Robust_SHS_20260929/runs/reference_bmp1/phase/vision_router.bmp`

The failed pilot data is preserved separately. **No full 1000-test capture or
G5 TRAIN adaptation has run for these full-head weights**; fitting the
electronic decoder to these unexpectedly dark frames would not verify the
requested robustness claim. The G2/G5 full-capture and G5 TRAIN-only decoder
adaptation entry points have been prepared in the isolated bench project. Once
the reference BMP pair produces the expected bright CCD signal again, first
repeat a bounded pilot before whole-dataset layerwise capture.

## Update after manual SLM GUI closure

The preceding no-full-capture statement is superseded. The user manually
displayed the exact retained reference BMP pair and observed a clear spot,
then closed the manual SLM/camera GUIs. Automatic display of the same BMP
pair immediately afterward gave corrected p99=255 rather than 23. The
automatic G5 four-sample, six-stage pilot then completed with bright CCDs.
The cause of the earlier display-state discrepancy is not conclusively
isolated, so the dark pilots remain quarantined.

The full G5 TEST run completed 1000 samples x six stages = 6000 CCD frames.
The unchanged full-head checkpoint had same-weight simulation Changed-cell
accuracy 0.9435 and physical Changed-cell accuracy **0.8740**. Its source
report is `runs/g5_full1000/report.json` in the isolated bench project, with
a local copy `g5_physical_full1000_report.json` beside this document. Relative
to G1 simulation 0.8880, physical G5 is lower by 0.0140 (1.58% relative),
already inside the requested 5% band *without decoder adaptation*.
Saturation per frame is logged; mild saturation was explicitly accepted by
the user without changing 2000 us / Gain_X4. A separate G2 full run was
started after a successful four-sample pilot. It has not yet been reported
complete here.
