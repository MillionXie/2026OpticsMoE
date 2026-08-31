# P13 optimization log

## 2026-09-01 — 渐进扩深原型

### 目标

- 从已完成的 P11 8-stage backbone 严格迁移，而不是随机重训所有深层参数；
- 以 64 stage / 9.63M phase 为主配置，并支持 16/32/100 深度消融；
- 不复制 width-96 mixer，使 reusable 电子 backbone 保持约 0.965M；
- alpha=0 时严格保持 P11 feature function，alpha>0 时新增 phase 可以学习；
- 本轮只完成本地原型和自动验收，不启动服务器训练、不提交 git。

### 实现动作

1. 新建 `QwenStemProgressiveOpticalImageNetBackbone`。
   token/channel stage 交替；四个 P11 pair 均匀放入目标深度。
2. 只在 8 个 P11 anchor 放置现有 width-96 `SlimSpatialTokenMixerSkip`。
   其余 stage 使用无参数 identity electronic skip；每个 stage 保留一个受约束
   光/skip 融合标量。
3. 为新增 stage 增加非训练 outer depth buffer：
   `x + alpha * (Stage(x)-x)`。alpha=0 使用直接 bypass；epoch 1 从 epsilon
   起步，并可按配置线性 ramp 到 1。
4. alpha 同时维护持久化 tensor 与 Python schedule value。forward 不进行逐层
   CUDA host synchronization，state-dict load 后会恢复二者一致。
5. 增加可选 non-reentrant per-stage activation checkpoint，并修复 backward
   recompute closure 的 slot late-binding 风险。
6. 实现严格 P11 migrator：检查 architecture signature、model report、stem
   SHA、state key 集合与 phase hash；迁移 stem/adapter/8 个完整 anchor，排除
   临时 ImageNet head。
7. 增加可审计 checkpoint/manifest 输出和参数报告。

### 锁定预算

- 64 stage phase：`64 * 3 * 224 * 224 = 9,633,792`；
- 8 个唯一 width-96 mixer；
- 56 个新增 identity electronic skip：0 transform 参数；
- 电子 backbone（不含临时 head）：965,176；
- 光学占 reusable trainable backbone：90.8937%。

### 本地验证

本地使用：

```text
C:\Users\Xml12\.conda\envs\qwen3vl-cifar10\python.exe
torch 2.10.0+cpu
pytest 9.1.1
```

已逐项通过：

- 四档 anchor schedule 与 64-stage 参数预算；
- 迁移到 64 stage 后 alpha=0 的最终 feature 与 P11 eval 输出 bitwise equal；
- 16-stage alpha=0.05 全链反传时，全部 8 个新增 phase 梯度 finite 且非零；
- epsilon/ramp 端点以及 state reload 后 alpha 同步。

最终完整命令结果：`8 passed in 5.98s`。测试使用合成冻结 stem 与正式
`backbone.pt` 字段契约，不代表已经对服务器上的 epoch-88 P11 文件执行迁移。

### 未执行与风险

- 未启动任何正式训练，也没有 P13 ImageNet 精度可报告；
- 未在 GPU 测 64/100 层峰值显存和吞吐；逐 stage checkpoint 不能替代正式的
  2--4 stage segment 实测/设计；
- 单轴传递函数仍由 `fft2/ifft2` 实现，真正 1-D FFT 是后续性能 TODO；
- 训练期 alpha<1 的 outer blend 是电子路径；物理部署口径必须以 alpha=1 为准；
- 深层 fixed-feedback 不可循环复用 8 个 P11 connector，需在深层 source
  训练完成后重新冻结完整 connector。

### 正式启动前复盘模板

```text
commit / dirty diff:
P11 checkpoint absolute path:
P11 checkpoint SHA-256:
stem checkpoint SHA-256:
P13 depth / anchor schedule:
migration manifest path:
alpha=0 real-checkpoint feature max_abs / exact_equal:
GPU UUID / PID:
batch / AMP / activation checkpoint / segment size:
peak allocated / peak reserved MiB:
samples/s and step time:
alpha epsilon / ramp epochs:
phase LR / electronic LR / head LR:
first finite-gradient audit for every new phase:
result status and checkpoint path:
```
