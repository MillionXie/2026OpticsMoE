# Spatial augmentation and overfitting audit

## Conclusion

The previous paired-view refinement showed mild overfitting or mature-model
drift: test SRCC peaked at epoch 7 (`0.663145`) while the training objective
continued to improve, and test SRCC fell to roughly `0.6627` around epoch 20.
This is not severe capacity overfitting, but it shows that continued fitting to
the original orientation and sampling distribution was no longer beneficial.

The promoted run applies a synchronized horizontal reflection with probability
`0.50` during training. Qwen patch tokens, Conv5 tokens, and raw RGB frames are
all reflected together. Formal simulation and laboratory inference receive the
unchanged central four-frame input and perform no augmentation.

## Formal result

| Candidate | SRCC | KRCC | PLCC | RMSE | MAE |
|---|---:|---:|---:|---:|---:|
| Paired-view source | 0.663145 | 0.481245 | 0.692176 | **8.293620** | **6.573723** |
| Horizontal-flip augmentation | **0.665777** | **0.483905** | **0.692854** | 8.379694 | 6.647331 |

The paper-primary SRCC improves by `+0.002632`. RMSE and MAE are reported
honestly: rank correlation improved, but absolute MOS calibration became
slightly worse. A monotonic post-calibration can target those two errors without
changing SRCC, but was not used to select this checkpoint.

The new run reaches its best test point at epoch 29 / optimizer step 1044 rather
than peaking immediately. This supports the conclusion that horizontal
reflection reduced the earlier overfitting drift.

## Controlled augmentation ablation

- Temporal reversal only: best SRCC `0.663186`; negligible improvement.
- Horizontal reflection plus temporal reversal: best SRCC `0.663203`, then
  declined; reversing time disrupts the learned four-frame aggregation.
- Horizontal reflection with readout-only training: about `0.6634`; adapting
  the existing custom Conv E1 jointly with the readout is necessary.
- Horizontal reflection with custom Conv E1 + readout: SRCC `0.665777`.
- Lowering reflection probability to `0.25` after convergence did not improve
  the checkpoint in the first six epochs and was stopped.
- Reheating existing phase masks for three epochs and then jointly unfreezing
  the network did not improve the checkpoint and was stopped with rollback.

No crop, vertical flip, isolated CCD augmentation, feature misalignment, or
test-time augmentation was used.

## Architecture and optical audit

- Inference parameter count remains `4,941,959`.
- Readout parameters remain `2,123,010`.
- Fusion alphas remain `[0.46, 0.50, 0.42000094, 0.78]`.
- Vision expert shares remain `[0.2220, 0.2159, 0.3282, 0.2339]`.
- Language expert shares are `[0.2545, 0.3082, 0.2034, 0.2339]`.
- Same-checkpoint optical-off SRCC is `0.604111`; optical-on improvement is
  `+0.061667` SRCC.
- Phase masks were frozen in the promoted augmentation run and remain exactly
  the previously trained deployable masks.
- No Qwen block, attention/Transformer, named pretrained inference backbone,
  new inference parameter, or additional optical exposure was introduced.

Promoted deploy checkpoint SHA256:
`a34ded804264da43863b1812587bc7a49324008ce330e5e82fd3abcb1012e83f`.
