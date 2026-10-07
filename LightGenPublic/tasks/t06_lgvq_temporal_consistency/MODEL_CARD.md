# Model card

## Intended use

Research evaluation of LGVQ Temporal consistency MOS under the pinned
16-video-by-4-frame optical/electronic computation graph.  The package is
intended for numerical review, ablation, and device-porting research.

## Inputs and outputs

- Input to the released student: precomputed frozen Qwen-front visual tokens,
  14-channel quality tokens, the fixed Temporal prompt embedding and mask.
- Physical grouping: 16 unrelated videos, four sampled frames per video.
- Output: 16 scalar Temporal MOS predictions per physical field.
- Formal test: 558 videos in 35 fields; two padding slots in the final field are
  excluded from the metrics.

The release contains frozen test tensors, not original videos.  It therefore
reproduces the pinned test computation but is not a general raw-video CLI.

## Optical/electronic graph

- wavelength: 532 nm
- propagation distance: 0.10 m
- simulated pixel pitch: 17 µm
- canvas / active plane: 518×518 / 478×478
- frame and video routers: four experts, optical Top-2
- optical passes: six
- evaluation unmodulated power fraction: 0.20
- fusion optical coefficients: approximately 0.5611, 0.5600, 0.5679, 0.5695

The coefficients are convex fusion gates after branch-wise RMS normalization;
they are not a percentage attribution of total model performance.

## Metrics

The release manifest records SRCC 0.8043868643.  The original training report
records 0.8043726361; the small difference comes from the pinned release's
independent metric recomputation and floating-point path.  Both round to
0.8044.  The teacher package treats the release manifest as the reproduction
reference.

## Training and evaluation limitations

- The checkpoint was selected by test SRCC every five epochs; there was no
  independent validation split.  The reported test performance is therefore
  selection-biased and should not be described as a blind final test.
- The model uses frozen Qwen-front features and an explicit 14-channel
  electronic quality bank.  It must not be described as pure optical feature
  extraction.
- Same-checkpoint optical bypass produced SRCC -0.0861, but this distribution
  shift is not a separately trained electronic baseline and does not isolate a
  causal optical contribution.
- Fixed cached inputs omit raw-video decoding and Qwen-front latency/energy.
- The 1.796 ms/video number reported elsewhere is a 16-video throughput
  normalization of a composed simulation timing, not single-video latency or
  measured laboratory end-to-end latency.
- Different CUDA/PyTorch FFT kernels may not be bit-identical.  The reproduction
  command checks prediction and metric tolerances explicitly.

## Hardware distinction

The 0.8044 value is the full fixed-weight simulation result.  The separate
six-layer SHS camera experiment obtained SRCC 0.7977 and is not relabeled as
0.8044.  Hardware control code is intentionally outside this numerical release
to keep the submission focused and portable.  The checkpoint uses a 10 cm,
17 µm logical propagation grid; the 8 µm SLM export preserves the 8.126 mm
active aperture by rasterizing it to 1016 device pixels.  The reference
simulation does not model stochastic camera sensor noise or the 8-bit device
quantization.  See [ARCHITECTURE_AUDIT_CN.md](ARCHITECTURE_AUDIT_CN.md).
