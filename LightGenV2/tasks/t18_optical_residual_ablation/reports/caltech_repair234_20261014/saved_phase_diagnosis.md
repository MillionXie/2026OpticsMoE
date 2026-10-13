# Saved phase diagnostics, CPU only

This independently published tool reads already audited best/last tensor states.
No model construction/forward, dataset/test arrays or GPU allocation. Actual completed
fit identities/current and original126 source blobs/checkpoint/audit SHA are checked.
The seed17 initial tensors were regenerated independently in CPU parameter creation
order, and their SHA exactly matched all recorded initial_parameter_sha256 values.
The existing complete scientific audit was not rerun. Exclusive result at run
`cpu_saved_phase_diagnosis_20261014/checks.json` exists; do not rerun/overwrite.

All arms have908552 trainable values:50176 router +858376 main phase values.
There are11 independently parameterized phase maps, physically four main propagation
stages plus router (five captures); the four parallel experts do not add depth.
Train1246 means729.175 parameters per image. This ratio is descriptive, not a
criterion or proof that capacity causes poor generalization.

Actual smoothness is the mean of22 directional mean squared adjacent differences
of phi=2pi*sigmoid(raw), including router, not a sum over all pixels. This is the
implemented ordinary phase smoothness; do not mislabel it as gradient magnitude or
as the smoothness of the quantized effective residual phase angle(M).

| Case | Parameters | Selected EMA smoothness | lambda .005 times smoothness | Saved epoch mean mixed objective |
|---|---:|---:|---:|---:|
| L4_rho03_mixup02 | 908552 | 0.533118 | 0.002666 | 1.587691 |
| L4_rho03_mixup04 | 908552 | 0.512317 | 0.002562 | 1.641630 |
| L4_rho03_mixup06 | 908552 | 0.530864 | 0.002654 | 1.817482 |

The initial weighted penalty was0.000082266. Selected penalties0.002562-0.002666
are small numerically relative to saved epoch mixed objectives1.588-1.817. These
values are not a loss component decomposition: selected EMA differs from changing
raw parameters within the training epoch, and saved unaugmented balancedNLL is a
different objective. A small penalty value does not establish weak penalty gradients
or predict that increasing its coefficient helps validation.

For selected first-stage expert maps, about87.9-91.4% of the zero-mean complex
modulation variation spectrum lies above radial1/8 cycle per logical pixel. Later
maps are lower, about54.5-72.7%, except the first global around89.5-90.7%; router
around58.0-61.2%. This statistic describes M-mean(M), not the illumination spectrum,
physical output power or image information. It does not prove high frequencies are
unnecessary; the threshold is a descriptive diagnostic, not tuned on accuracy.

Selected phase floor quantization errors max<=0.024544rad, consistent with one
8bit bin2pi/256. Selected maps use172-255 distinct phase levels across arms, no
single-gray collapse. Residual main modulation amplitude means approximately.444-.494
remain physically attenuating; amplitude and angle(M) must still be exported jointly.
These tensor properties do not replace native BMP or real exposure/LUT calibration.

The evidence supports a bounded capacity-control question rather than another MixUp
sweep. Neither capacity nor spectral statistics identify a unique cause, and no new
candidate accuracy has been computed. [Exact CPU receipt](saved_phase_diagnosis_cpu.json).
[Concrete proposed comparison, not authorized or started](PHASE_CAPACITY_PROPOSAL_20261014.md).
