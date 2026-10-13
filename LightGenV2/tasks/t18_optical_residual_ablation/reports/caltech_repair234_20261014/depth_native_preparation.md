# Selected depth2/3 native verification preparation

Independent `caltech_repair_depth_native.py` adds the final selected-weight BMP
verification after the conditional depth2/3 validation. It does not alter frozen
126/130 training sources or start any fit. Actual selected-weight execution remains
pending the L4 study and both depth checks passing their predeclared criteria.

Only after both actual depth fits and parent/children exit, saved CPU audit passes,
both depths are eligible and their PT/config/history/source identities match:

```
CUDA_VISIBLE_DEVICES='' /home/guest3/miniconda3/envs/xml/bin/python \
  LightGenV2/tasks/t18_optical_residual_ablation/caltech_repair_depth_native.py \
  --study <L4study> --base <depthrun> --out <depthrun>/selected_native_bmp
```

The independent exclusive output preserves failures and rejects repeated/partial
outputs. Each selected validation EMA gets its own static joint phase/amplitude
package and three synthetic randomRGB/dim4/255/black native-disk complete replays.
Native amplitude1920x1080 and phase1920x1200 reverse to478 exactly; entering expert
uses224, entering global478. Capture counts3/4 for main2/3, probability error<=1e-6,
identical continuous routing/CCD helper processing and all black-stage amplitudes
zero are required. Total12CPU classifier forwards for synthetic probes, no dataset,
train/val accuracy recomputation, test, device or GPU. PT/audit/config/current and
original Git identities are recorded. This is offline parity, not physical-device
acceptance or formal training admission on its own.

Linux CPU checks actually passed after source publication eb033a879: premature
selected-native execution refused before output creation; six fresh synthetic
random/dim/black probes for main2/3 produced native disk inverse exact and probability
error0, captures3/4, identical routing/CCD helpers and black amplitude zero.12CPU
forwards, no dataset/test/device/GPU, actual running GPU process identities unchanged.
[Actual CPU receipt](depth_native_cpu_checks.json). Smoke
`caltech_repair_depth_native_preparation_20261014/checks.json` exists and must not be
rerun/overwritten. These fresh states do not constitute actual selected trained EMA
checks, successful depth admission or hardware acceptance.
