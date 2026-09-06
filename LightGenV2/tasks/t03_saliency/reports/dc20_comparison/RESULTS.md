# SALICON 正式仿真比较

无 validation；SALICON 官方 val2014 作为 public test，并用于每 5 epoch 选模。

| 方法 | Router | CC↑ | KLD↓ | SIM↑ | NSS↑ | AUC-J↑ | MAE↓ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Optical Router Top-2 | optical | 0.8291 | 0.1330 | 0.8063 | 0.9283 | 0.7631 | 0.0890 |
| Matched D2NN | none | 0.8346 | 0.1296 | 0.8092 | 0.9344 | 0.7643 | 0.0884 |
| Frozen Qwen | none | 待5090D | 待5090D | 待5090D | 待5090D | 待5090D | 待5090D |

大模型速度与功耗也必须留待 RTX 5090 D 按 `BASELINE_5090D_TODO.md` 实测。
