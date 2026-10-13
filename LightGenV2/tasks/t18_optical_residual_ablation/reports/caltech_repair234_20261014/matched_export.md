# Matched repair-graph offline deployment export, 2026-10-14

Independent `export_caltech_repair234.py` and `check_caltech_repair234_export.py`,
implementation commit `ecb422f67eb9b27784cf1a12ca2bdf22db979acb`. Frozen126 fit
sources are unchanged, including all original121 dependencies. Current fits were
not stopped, restarted, retuned or copied. No laboratory connection.

| Stage | Input to actual physical encoder | Raw output / next step |
|---|---|---|
| Extra router |fixed RGB mosaic224 RMS.25 |raw CCD478 ->continuous four L2 amplitude weights |
| Expert |224 plus retained dense4 weights |raw CCD478 ->global478 or expert224 OEO |
| Global |preserved478 field |raw CCD478 ->expert224 OEO or final20 energy windows |

Every stage exports joint amplitude*abs(M) and angle(M); amplitude round255,
phase floor256 inside the unchanged physical_fields. Canonical478; native
amplitude1920x1080/phase1920x1200, centered1016 crop. Fixed detector/router masks,
class order, stage sizes, phase BMPs, amplitude multipliers and source/PT/config/
parameter SHA are included. No additional peak normalization. Active and illuminated
mean/p99/max/zero/saturation are recorded separately; canvas black padding does
not establish illumination or camera SNR.

The new export interface rejects224global,478expert, old hardTop2/non-L2/nonpositive
expert weights and invalid/nonfinite raw CCD. `routing_from_ccd` and
`amplitude_from_ccd` use the same continuous4 and next-stage nonlinear transforms
as the simulation, checked through the complete optical dependency sequence.

Actual CPU native-disk checks passed for all six depth/rho combinations and three
synthetic probes each (randomRGB, uniform4/255, black): six static packages and
18 dynamic sequences, captures3/4/5, every native reverse mapping exact, maximum
probability error0 (threshold1e-6), black all-stage amplitudes0. Separate routing
helper weights exactly match model weights. Static phase and amplitude multiplier
files match the underlying planes. Invalid contracts refused.36 CPU reference/
replay forwards, no GPU/device/dataset/test arrays or test inference. This is a
fresh synthetic-state tool check, not a trained-candidate or hardware acceptance.

[Full check receipt](matched_export_cpu_checks.json). Assets and synthetic PTs
remain in `runs/smoke/caltech_repair234_matched_export_20261014`, excluded from Git.
Never rerun or overwrite that completed output. Selected trained EMA still needs
its own native-disk replay after saved audit and profile choice. The CLI requires
passed all-three study CPU audit, eligible chosen case and actual parent/children
exit before static export; it does no classification forward or test.

For future hardware, acquire the whole dataset layer by layer in dependency order:
router first, retain each sample's continuous weights, then each expert/global
stage with the corresponding original/frozen OEO amplitude. Preserve same-checkpoint
captures and hardware-session identity. Only calibrated raw-linear CCD data feeds
the frozen transforms; LUT/ROI/phase/amplitude response, exposure/gain/orientation
and linear CCD scale require a new hardware session. Nothing here certifies bench
accuracy or the75% simulation target.
