# OpenMoji 正式仿真比较

无 validation；每 5 epoch 测 test 并按 changed-cell accuracy 选模。

| 方法 | Router | Changed↑ | Category↑ | Edit IoU↑ | Object F1↑ | Exact↑ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Optical Router Top-2 | optical | 0.9800 | 0.9895 | 0.9350 | 0.9837 | 0.8950 |
| Matched D2NN | none | 0.9895 | 0.9944 | 0.9813 | 0.9949 | 0.9650 |
| Frozen Qwen | none | 待5090D | 待5090D | 待5090D | 待5090D | 待5090D |

大模型性能、速度与功耗均留待 RTX 5090 D 按 `BASELINE_5090D_TODO.md` 实测。
