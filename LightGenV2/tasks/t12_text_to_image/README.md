# T12 单物体文生图

T12 是 LightGenV2 的单次前向、文本条件图像生成任务。任务只生成少数类别的单个居中
物体，不做开放域、多物体、计数、空间关系、扩散去噪或自回归生成。

当前状态：模型、数据合同、Qwen/VAE 缓存、训练入口、配对 baseline、可视化比较与 CPU
结构测试已经建立；正式数据冻结为 ABO 小子集，首轮 latent 回归训练已完成但明显模糊，
因此保留首轮结果并新增独立的单次前向 GAN 锐化 profile。硬件结果尚未产生。

## 冻结协议

- 数据：Amazon Berkeley Objects（ABO）的 catalog `images-small`，数据目录必须附带原始
  `LICENSE-CC-BY-4.0.txt`，并在产物中保留来源、署名及修改说明。
- 类别：精确使用 `SHOES`、`CHAIR`、`LAMP`、`TABLE` 四种 product type。
- 每类 250 个不同商品，每个商品只取一张 main image；train/val/test 按商品身份切分为
  200/25/25，不允许同一商品跨 split。
- 规模：训练 800，验证 100，测试 100，共 1,000 张。
- 图像：过滤过小、极端长宽比、swatch 和明确多件套；裁掉角落背景后将商品居中到确定性
  中性背景，224×224。正式训练前必须再审计 contact sheet。
- 文本：只使用结构化颜色、材质和类别组成短描述；不要求数量及空间推理，也不输入品牌名。
- 文本前端：完整冻结 `Qwen/Qwen3-VL-2B-Instruct`，缓存最终 hidden state 的 masked mean。
- 图像 codec：完整冻结 `stabilityai/sd-vae-ft-mse`，目标和输出均为 `4×28×28` latent。
- 推理：一次生成主干前向和一次 VAE decode；没有循环。

## LightGen 结构

输入为一个冻结 Qwen 文本向量和一个 256 维 style code。二者投影后注入 14×14、192 维
空间 token。正式 profile 直接复用 T01 已审计的 DC20 `BalancedVisionCore`：

```text
stage-1 input ─┬─ electronic residual-1 ─┐
               └─ optical Router+Top2 experts ─┤ detached-RMS fusion
                                                  ↓
stage-2 input ─┬─ electronic residual-2 ─┐
               └─ optical global block ─────────┤ detached-RMS fusion
                                                  ↓
                          residual latent head: 14×14 → 4×28×28
                                                  ↓
                                  frozen VAE decoder → 224×224 RGB
```

每个 stage 的电子和光学分支严格读取同一个输入；不存在 `electronic → optical` 串行关系。
第一阶段 Router、Top-2 experts 和第二阶段 global block 均沿用现有 224 SLM、478 active CCD、
518 numerical canvas、17 μm、10 cm 和尺度匹配融合合同。

训练时额外使用一个轻量 posterior encoder，从真实 VAE latent 得到 `mean/logvar`；推理时
删除该 encoder，直接采样 `N(0,I)`。第一版损失为 latent L1、latent MSE 和 warm-up KL，
初始 profile 不使用 GAN。首轮正式结果证明 posterior 能恢复类别轮廓，但 L1/MSE 会抹平纹理，
随机 prior 还存在更明显的分布错位；所以 `lightgen_gan` / `baseline_gan` 作为独立第二阶段
profile 使用四层 residual latent decoder，并让冻结 VAE 解码后的 posterior 与随机 prior 同时
接受 RGB PatchGAN 和 feature matching。该变化只作用于 decoder/训练目标，不改变并行主干，
推理仍然只有一次主干前向和一次 VAE decode，也不覆盖首轮 checkpoint。

`compact_fft` 仅供 CPU smoke 和结构调试，不能作为论文性能或硬件结果。

## Qwen + VAE baseline

Baseline 与 LightGen 共用：

- 完全相同的 train/val/test；
- 同一份冻结 Qwen text cache；
- 同一份冻结 VAE latent cache 和 VAE decoder；
- 相同 posterior、style 维度、latent head、损失、训练轮数和采样 seed。

唯一的主干差异是 baseline 用两层条件电子残差替代整个光学支路。正式参数量当前约为：
LightGen 3.65M、baseline 3.09M，不把冻结 Qwen/VAE 参数计为可训练参数。

训练结束后使用 `compare.py` 在相同四条 prompt、相同 seed 下生成配对网格。正式质量报告
还必须在固定 test 上给出 FID/KID、CLIPScore、类别正确率和 LPIPS diversity；当前代码生成
网格不等同于已有质量结果。

## 数据准备

使用 ABO 官方 `abo-images-small` 和 `abo-listings` 归档，运行：

```powershell
python -m LightGenV2.tasks.t12_text_to_image.prepare_abo `
  --abo-root D:\abo `
  --output-dir LightGenV2\tasks\t12_text_to_image\dataset\abo_single_object_v1 `
  --categories "SHOES,CHAIR,LAMP,TABLE"
```

准备脚本会拒绝缺少精确 CC BY 4.0 许可证、类别实例不足及跨 split 身份泄漏。原有
`prepare_uco3d.py` 仍保留为将来网络条件允许时的可选数据入口，但不属于本次冻结协议。

## 运行

安装生成任务的额外依赖：

```powershell
pip install -r LightGenV2/requirements/generation.txt
```

先跑结构测试和 smoke：

```powershell
python -m pytest LightGenV2/tasks/t12_text_to_image/tests -q
python -m LightGenV2.tasks.t12_text_to_image --profile smoke --phase smoke --device cpu
```

数据就绪后只缓存一次冻结特征，两条路线读取相同缓存：

```powershell
python -m LightGenV2.tasks.t12_text_to_image --profile lightgen --phase cache --device cuda
python -m LightGenV2.tasks.t12_text_to_image --profile lightgen --phase train --device cuda
python -m LightGenV2.tasks.t12_text_to_image --profile baseline --phase train --device cuda
python -m LightGenV2.tasks.t12_text_to_image --profile lightgen_gan --phase train --device cuda
python -m LightGenV2.tasks.t12_text_to_image --profile baseline_gan --phase train --device cuda
python -m LightGenV2.tasks.t12_text_to_image --profile lightgen --phase evaluate --device cuda
python -m LightGenV2.tasks.t12_text_to_image --profile baseline --phase evaluate --device cuda
python -m LightGenV2.tasks.t12_text_to_image.compare `
  --lightgen-profile lightgen_parallel_decoder_gan.yaml `
  --lightgen-checkpoint LightGenV2\tasks\t12_text_to_image\runs\simulation\lightgen_parallel_decoder_gan_seed42\best_checkpoint.pt `
  --baseline-checkpoint LightGenV2\tasks\t12_text_to_image\runs\simulation\qwen_vae_baseline_seed42\best_checkpoint.pt `
  --output-dir LightGenV2\tasks\t12_text_to_image\runs\simulation\matched_comparison
```

2026-09-20 的正式服务器训练、失败实验和 checkpoint 选择记录见
[`reports/20260920_server_training.md`](reports/20260920_server_training.md)。当前推荐的是受约束
decoder-only GAN 的第 4 epoch；无约束 GAN 因随机 prior 模式坍塌被明确否决。

GAN 锐化阶段可通过 `--init-checkpoint <首轮 best_checkpoint.pt>` 仅加载生成器与 posterior
权重。浅层 latent head 的输出卷积会自动映射到深层 head，新插入的 residual decoder block
以接近恒等映射开始；优化器、判别器和 epoch 计数均从头开始，warm-start 来源会写入每个
checkpoint 和 `training_summary.json`。

若无条件 prior GAN 出现跨类别模式坍塌，使用 `lightgen_decoder_gan`：它冻结 warm-start
得到的并行光电主干、posterior 与旧 latent head，只训练新插入的 decoder residual blocks，
并用 `prior_latent_delta_weight` 将随机 prior 锚定到首轮单次生成结果。该 profile 仍然没有
迭代采样，也不会改变电子/光学并行拓扑。

服务器正式运行时可以用 `--data-dir`、`--qwen-checkpoint` 和 `--vae-checkpoint` 显式指向
仓库外的冻结资产；解析后的绝对路径会写入 `resolved_config.json`，避免 worktree 被数据文件
污染，也避免缓存阶段临时访问模型网络。

正式 run 只保留 `best_checkpoint.pt` 和 `last_checkpoint.pt`。验证集只负责选择 checkpoint；
测试集只能在协议和超参数冻结后评估。

## 未完成项

1. 生成 ABO 正式子集和 contact sheet，完成人工画面审计。
2. 生成三份共享 feature cache，先做单 batch 显存/速度 pilot，再分别训练 LightGen 和 baseline。
3. 补固定测试集的 FID/KID、CLIPScore、类别准确率和多样性评估。
4. 仿真候选稳定后再建立 DC20 硬件 profile；不得把 `compact_fft` 数值写成硬件结果。
