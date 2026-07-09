# BDD100K TimeOfDay-3 Standard Baselines

This folder is for baselines against
`qwen3_vl_2b_bdd100k_timeofday3_optical_fullstack4_token64_residual`.

Dataset handling is intentionally aligned with that experiment:

- the same BDD100K TimeOfDay-3 preparation helper is reused;
- class order is `daytime`, `night`, `dawn_dusk`;
- train/test limits use the same seeds as the Qwen experiment;
- validation split uses `validation_fraction` with seed `42`;
- default epoch sampling uses `train_samples_per_class_per_epoch=1000`;
- configs point `data_root` at the Qwen TimeOfDay-3 experiment data directory.

The network input resize is recorded separately in each config because the Qwen
experiment uses a processor pixel budget rather than a fixed CNN input tensor.

## Included baselines

- `standard_d2nn`: phase-only D2NN. There is no amplitude mask, no intermediate
  O-E-O detection, no convolutional readout, and no MLP readout. The final
  square-law detector plane is integrated over fixed class regions; those
  detector energies are the logits.
- `lenet5`: standard LeNet-5 style `Conv5-AvgPool-Conv5-AvgPool-FC120-FC84`
  on 32x32 grayscale inputs.
- `resnet18`: common residual CNN baseline.
- `vgg11_bn`: conventional VGG-style CNN baseline.
- `mobilenet_v2`: common lightweight CNN baseline.

For adviser-facing comparisons, ResNet-18 plus one of VGG11-BN or MobileNetV2
is usually enough beyond LeNet-5 and D2NN. VGG is a plain deep CNN family;
MobileNetV2 covers the efficient/mobile CNN case.

