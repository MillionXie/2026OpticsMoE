# MNIST-4 10 cm hardware workflow

The amplitude convention is fixed throughout this project:

```text
255 = white / transmissive
0   = black / blocking
```

The phase SLM remains operator-controlled. The acquisition program validates
the exact 1920×1200 phase BMP and records its SHA-256, then waits for the
operator to confirm that this mask is visible before it starts the automatic
Meadowlark amplitude-SLM and TUCam sequence.

## Exported profiles

`--phase export` creates two independent stage directories.

- `demo_topk`: the easiest, high-margin examples from each class. It is useful
  for alignment and a quick visual demonstration, but its accuracy is biased
  and must not be reported as test performance.
- `formal_fixed_random_100_per_class`: 100 examples from every true class,
  drawn with the configured fixed seed without looking at model predictions.
  This is the profile for hardware accuracy.

Each stage contains its own `phase_to_play/`, `amplitude_to_play/`,
`samples.csv`, and `stage_contract.json`. The trained phase filename is
`mnist4_single_layer_17um_10cm.bmp`.

## Server-side export and ZIP

After training, export the hardware payload with the release config and best
checkpoint. Then package the payload, lightweight Python runtime, Meadowlark
SDK, and TUCam SDK into one ZIP:

```powershell
python -m experiments.d2nn_mnist4_single_layer_17um_10cm --config experiments/d2nn_mnist4_single_layer_17um_10cm/configs/release/mnist4_single_layer_17um_10cm_notebook_mse.yaml --phase export --checkpoint experiments/d2nn_mnist4_single_layer_17um_10cm/runs/mnist4_single_layer_17um_10cm_notebook_mse/checkpoints/best.pt

python -m experiments.d2nn_mnist4_single_layer_17um_10cm.lab_package --export-dir experiments/d2nn_mnist4_single_layer_17um_10cm/runs/mnist4_single_layer_17um_10cm_notebook_mse/hardware_export_10cm_normal_polarity --output experiments/d2nn_mnist4_single_layer_17um_10cm/runs/mnist4_single_layer_17um_10cm_notebook_mse/mnist4_single_layer_17um_10cm_lab_bundle.zip
```

The packager also writes a `.zip.json` sidecar containing size and SHA-256.
`--omit-vendor-sdk` is only for a small developer archive; do not use it for
formal delivery to a computer that lacks the SDK directories.

## Laboratory computer

Extract the ZIP, enter its root, and install the light environment. Torch and
Qwen are not needed on the laboratory controller:

```powershell
python -m pip install -r experiments\d2nn_mnist4_single_layer_17um_10cm\requirements-lab.txt
```

Before opening either device, edit:

```text
experiments\d2nn_mnist4_single_layer_17um_10cm\lab_hardware_config.yaml
```

Fill `camera.device_roi_xywh` from the four-focus calibration and select the
LUT matching the actual amplitude-SLM temperature. The ROI values are
`[left, top, width, height]` and must satisfy the camera SDK's divisibility
requirements.

Validate the full formal payload, DLLs, LUT, phase BMP, camera, and ROI without
opening the devices:

```powershell
python -m experiments.d2nn_mnist4_single_layer_17um_10cm.lab_pipeline --phase validate --stage-dir payload\formal_fixed_random_100_per_class
```

Run acquisition after loading the displayed 10 cm phase BMP on the phase SLM:

```powershell
python -m experiments.d2nn_mnist4_single_layer_17um_10cm.lab_pipeline --phase acquire --stage-dir payload\formal_fixed_random_100_per_class
```

Evaluate existing captures:

```powershell
python -m experiments.d2nn_mnist4_single_layer_17um_10cm.lab_pipeline --phase evaluate --stage-dir payload\formal_fixed_random_100_per_class
```

Or execute validation, acquisition, and evaluation as one sequence:

```powershell
python -m experiments.d2nn_mnist4_single_layer_17um_10cm.lab_pipeline --phase all --stage-dir payload\formal_fixed_random_100_per_class
```

For the biased demonstration stage, evaluation is deliberately disabled unless
the diagnostic opt-in is explicit:

```powershell
python -m experiments.d2nn_mnist4_single_layer_17um_10cm.lab_pipeline --phase all --stage-dir payload\demo_topk --allow-biased-demo-metric
```

That command reports `demo_success_rate`, never `accuracy`. It is not a test
metric and must not be used in a paper table.

Use `--clear-output` only when intentionally replacing captures already stored
under that stage. Add `--flip-vertical` or `--flip-horizontal` only if the
Fresnel correspondence measurement requires it. The configured camera already
saves a 478×478 model-ready frame, so no evaluation `--roi` is normally needed.

Results are written to:

```text
<stage>\hardware_evaluation\hardware_metrics.json
<stage>\hardware_evaluation\hardware_predictions.csv
<stage>\hardware_evaluation\processed_478\
```

Formal evaluation first verifies `acquisition_logs/capture_manifest.csv`, the
exact CCD file set and play count, and the phase-mask SHA-256. Missing or
mismatched acquisition records are rejected. Near-black, substantially
saturated, or four-detector-near-equal frames receive prediction `-1`; formal
evaluation fails by default if any such invalid frame exists. The JSON and CSV
retain `valid_count`, `invalid_count`, and per-frame QC reasons for diagnosis.

The metrics include overall accuracy, a 4×4 confusion matrix, and per-class
accuracy. No synthetic or unmeasured background subtraction is performed. The
controller performs an automatic sequential playback/capture loop; the default
configuration does not promise a particular acquisition rate.
