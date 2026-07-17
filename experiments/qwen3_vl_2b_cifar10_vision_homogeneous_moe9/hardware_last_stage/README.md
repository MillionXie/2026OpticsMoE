# Qwen vision homogeneous MoE: final-stage physical validation

This tool exports the last optoelectronic reload plane of the trained
`qwen3_vl_2b_cifar10_vision_homogeneous_moe9` student for a one-shot optical
bench test. It does not export the input to expert layer 5 and it does not
export the detector intensity as an amplitude pattern.

## Verified physical plane

The exact final path in `VisionHomogeneousMoESurrogate.forward()` is:

```text
expert phase layer 5
-> 20 cm propagation to the global plane
-> square-law detection
-> per-selected-expert non-affine LayerNorm
-> activation
-> routing-weight reapplication and unselected-expert hard zero
-> zero-phase amplitude reload
-> global phase modulation (same plane; no propagation gap)
-> 20 cm propagation
-> square-law CCD detector [480,480]
```

Therefore the exported amplitude and global phase must be optically registered
in the same plane (or conjugate planes with unit magnification). The simulation
does not contain a free-space distance between them.

The simulated active area is 450×450 at 16 µm. It is expanded with nearest
neighbour replication to 900×900 at 8 µm and centered as follows:

- amplitude: 1920×1080 BMP, active bounds `x=[510,1410)`, `y=[90,990)`;
- phase: 1920×1200 BMP, active bounds `x=[510,1410)`, `y=[150,1050)`.

The amplitude device and phase device have different raster heights, so their
pixel coordinates differ vertically. Their physical optical centers and their
900×900 active areas coincide.

## Exported groups

- `correct_high_confidence`: per-class examples that the floating student and
  the quantized amplitude/phase replay both classify correctly;
- `random_test`: deterministic random CIFAR-10 test examples, whether correct
  or incorrect, to avoid selecting only easy cases.

The package copies the exact student surrogate/head checkpoints and resolved
source config into `student_package/`. `manifest.csv` links every BMP to its
input, label, token count, routing weights, simulated prediction, and quantized
replay prediction.

## Physical CCD readout and fine-tuning

After acquisition, copy `ccd_capture_manifest_template.csv` to
`ccd_captured_manifest.csv` and fill `ccd_path`. A CCD image is a square-law
intensity image, not an amplitude field. `ccd_readout.py` uses the same final
electronic path as the student:

```text
CCD intensity [480,480]
-> AvgPool2d(4) [120,120]
-> non-affine LayerNorm
-> activation
-> first T token rows
-> trainable OutputAdapter 120->1024
-> token mean
-> trainable normalized-linear head
-> CIFAR-10 logits
```

Rows intended for fine-tuning must use `split=train` or `split=validation`.
Keep the official test captures as `split=test`; the program deliberately
refuses to train when the manifest contains only test rows. By default only the
OutputAdapter and classification head are updated. Qwen and every optical
parameter remain absent/frozen.

Camera registration can be configured with `ccd_crop_xywh`, quarter-turn
rotation and horizontal/vertical flips. The default does not normalize each
sample independently, so measured relative intensity is not silently erased.

