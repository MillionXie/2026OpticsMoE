# P13 8→16 ImageNet growth 终态证据快照

本目录保存从服务器正式 run 同步的**小型、可进 Git 的原始 JSON**，不包含 checkpoint。审计时间为 2026-09-01 23:36 CST；服务器终态文件写入时间为 23:24 CST。

## 结果摘要

| 项目 | 数值 |
|---|---:|
| 8-stage 初始重评 Top-1 / Top-5 | `51.346% / 75.560%` |
| 16-stage best（epoch 19）Top-1 / Top-5 | `51.428% / 75.752%` |
| 相对起点 | `+0.082 / +0.192 pp` |
| epoch 20 Top-1 / Top-5 | `51.352% / 75.762%` |
| 光学 phase 参数 | `2,408,448` |
| 电子 backbone 参数 | `965,128` |
| 可训练 backbone 光学参数占比 | `71.39%` |
| best checkpoint 平均绝对 phase motion | `0.756711 rad` |

最终破坏性消融 Top-1 为：`optical_off=0.476%`、`phase_random=0.108%`、`electronic_skip_off=0.096%`。这些消融证明共适应模型同时依赖已训练光学路径和电子 skip，但它们不是独立训练的纯光/纯电子模型性能。

结论边界：该 run 证明 16-stage 全深度训练、固定反馈、phase 更新和 backbone 导出均能完整收口；性能只与同 run 的 8-stage 起点持平。必须完成同预算 8-stage continuation 才能判断深度本身是否有效。

## 文件与校验和

| 文件 | SHA-256 |
|---|---|
| `manifest.json` | `5181051a8ed9ce149b61663930eccbf6405b7b4741325cb5caa6d4de7fa968d1` |
| `history.json` | `5ff9ce4c65f0ae55d393546caf94187b852555f918c6aa7dec0df40b2efa688d` |
| `latest.json` | `ab541f4cf427d83bc80e4aa2bfe3f90015312180a2cc6230896c6d2b9c0766d7` |
| `result.json` | `eeb824de2533c56688e21bef450b1410f48a3231dbc35d70fab0e4e5b2c5a360` |

原始 run：

```text
/DATA/DATA1/guest3/2026OpticsMoE_p13_488f9d48/
  experiments/qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone/
  runs/p13_growth16_fa_source_20e_gb192
```

关键 checkpoint 仍只保存在服务器，路径与 SHA-256 见 [`../../RUN_REGISTRY.md`](../../RUN_REGISTRY.md)。
