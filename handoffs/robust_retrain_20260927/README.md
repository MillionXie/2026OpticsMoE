# Physical-power and robustness recovery, 2026-09-27

## T12 diagnosis

Current deployment performs shared-batch maximum amplitude normalization. It does not percentile-clip peaks. Prior clipping experiments in other pipelines are not evidence that this exact T12 pipeline uses that method.

Six-item energy audit: remaining propagated energy inside the active CCD ROI is about 91–99.96%. Input-to-output full-canvas energy ratios are about 0.73–0.94; therefore the audit does NOT demonstrate ideal unitary propagation of every mode. Do not describe this as photons created or as proof that all losses occur outside the CCD. Transfer-function support, cropping and real transmission must be distinguished.

The simulation encodes inputs with RMS scaling. Hardware encodes peak-bounded amplitude. Peak-normalized language-router mean intensity is about 0.0014 of a full-white active aperture. This is not fixed by camera-image contrast enhancement. A higher simulation mean in arbitrary normalized units does not guarantee adequate detector photoelectrons.

Candidate training preserves geometry, architecture and parameter count. It adds occupied-input peak-power utilization and ROI-efficiency deficits to the loss, phase learning-rate reheating, 30% coherent phase leakage, one logical-pixel input/phase/CCD shifts, and absolute-plus-signal-dependent detector noise. The absolute noise floor is a proxy, NOT calibrated camera electrons. No gamma, input percentile clipping, k-space filter or 8-bit straight-through quantization was newly added by this recovery.

Stage1: 600 steps; noisy training probability 0.5. No checkpoint passed the original clean-quality guard. Stage2 starts from stage1 last, 600 steps, noise probability0.25, reduced LR. Same original clean VAL MSE limit 0.002732053245320761; 96 fixed validation members select the weight, TEST does not select it. Best/last only.

Stage2 selected step600; full 2304 TEST clean MSE in [-1,1] units=0.0026938080924689225; noisy MSE=0.013984152653898086. These are simulation results, NOT physical test results. Original matched full-test evaluation is separate. On fixed96 VAL, original clean/noisy MSE=.002483684768473419/.030410621121215325; recovered clean/noisy=.002540789083771718/.014186207476692895. Occupied peak-power utilization .05466848→.08015752, not whole-aperture utilization. This increase alone does not prove adequate physical SNR at400us.

Server source: `/DATA/DATA1/guest3/2026OpticsMoE/.worktrees/t12_physical_robust_20260926`, branch `codex/t12-physical-robust-20260926`. Candidate: main `LightGenV2/tasks/t12_text_to_image/runs/simulation/physical_robust_dc30_recovery_20260927/best.pt`. Keep original small checkpoint, never silently overwrite. Before resuming full physical TEST, verify candidate amplitude statistics and actual unsaturated CCD signal; use a documented exposure scan if needed. Then restart in a distinct run; never mix old and new checkpoint batches.

## ABO

Both20-epoch robust runs completed. Clean-score best remains initialization at R@1=.8300, so no clean improvement is claimed. Matched stochastic simulation also evaluates trained last masks rather than mistaking an unchanged best checkpoint for robust training output. `abo_stress.json` retains all conditions and metrics; these are not measured physical scores. Existing physical R@1=.68875 remains separate.

Original stressed R@1: no shift=.6875, one logical pixel=.6025. Trained shifted last: no shift=.70125, one pixel=.65875. Improvement under this simulated stress does not satisfy simultaneous unchanged clean performance; do not promote silently. Exact remaining conditions are in JSON.

## Delivery scope

No six-item Excel is a full-test delivery. The original physical acquisition was stopped after18/2304 items because of near-background CCD signal. Preserve evidence, do not use those batches with a newly trained mask. Final physical delivery must contain every test index, prompt, lossless256RGB originals, per-image metrics, hashes and consolidated CSV/JSON/Excel. An upscaled image is not a higher-resolution native output.
