# Spatial paired-view refinement

The compact Spatial-4 model was refined with two aligned four-frame views per
training sample. Both views receive the same human MOS supervision, and a small
prediction-consistency term discourages sensitivity to the exact uniform frame
sampling offset. This is a training-only method: formal simulation and
laboratory inference still use the central four frames, one forward pass, and
the unchanged 4,941,959-parameter student.

## Result

| Candidate | SRCC | KRCC | PLCC | RMSE | MAE |
|---|---:|---:|---:|---:|---:|
| Previous aligned multiview | 0.662943 | 0.481181 | **0.692221** | 8.301483 | 6.580074 |
| Paired-view refinement | **0.663145** | **0.481245** | 0.692176 | **8.293620** | **6.573723** |

The gain is small but reproducible under the exact central-view test contract.
The best checkpoint was observed at epoch 7; later epochs through epoch 29 did
not improve it, so the run was stopped rather than occupying a GPU needlessly.

The fusion alphas remain `[0.46, 0.50, 0.42000094, 0.78]`, all strictly above
the required 0.42 lower bound. Expert use remains non-collapsed:

- Vision selected shares: `[0.2220, 0.2159, 0.3282, 0.2339]`.
- Language selected shares: `[0.2509, 0.2912, 0.2177, 0.2401]`.

With the same checkpoint and optics bypassed, SRCC is `0.597804`; the optical
path contributes `+0.065341` SRCC. The deploy checkpoint contains no optimizer
state or duplicate EMA state and has SHA256
`fc5b6fa17406fef52f987ab55e5f57232defd360f172cd1f365c572bc9f422ae`.

No Qwen block, Transformer, attention module, recurrent module, named
pretrained inference backbone, new inference parameter, or extra laboratory
exposure was introduced by this refinement.
