# OpticalMoE Experiments

This directory is a clean experiment workspace placed beside the legacy
`opticalmoe/` project. It is intended for reusable experiments built around the
validated Angular-Spectrum global-router OpticalMoE path:

```text
input
-> AngularSpectrumPropagator input_to_prompt
-> prompt-plane complex-amplitude global router
-> AngularSpectrumPropagator prompt_to_expert
-> expert entrance aperture
-> expert phase layers
-> global FC phase
-> detector/readout
```

It intentionally does not use the older spatially partitioned prompt or FFT
convolution shortcut for expert entrance generation.

Implemented first:

- `single_task/`: single-dataset classification baselines and MoE variants.

Placeholders for future work:

- `dataset_switching/`
- `same_input_multitask/`
- `expert_task_ablation/`
- `prompt_ablation/`

