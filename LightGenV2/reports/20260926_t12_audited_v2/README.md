# T12 光电架构修正 v2（2026-09-26）

本目录只放此次架构修正的审计和适配结果，旧版效果权重未覆盖。
**通过架构测试不代表画质达标。短程适配权重不是最终效果版。**

## 已验证的结构

| 项目 | 大版 | 小版 |
| --- | ---: | ---: |
| 总参数（共享 token embedding 按约定单列、不计） | 144,663,450 | 11,137,746 |
| 纯相位参数 | 958,728 | 958,728 |
| 语言 Qwen-style 层数 | 2 | 2 |
| 语言隐藏宽度 / SwiGLU 中间宽度 | 640 / 1536 | 512 / 1024 |
| 图像输入 / 输出 | 256×256 | 256×256 |
| 光学 active ROI / 传播 canvas | 478 / 518 | 478 / 518 |

保留的是已有、经过训练的紧凑 Qwen-style 文字头，不是原预训练 Qwen3 的两层原权重。
词嵌入仍来自冻结 Qwen，计数约定不能把词嵌入称为 tokenizer。
图像侧沿用原型的图像编码/解码骨干，不声称使用原始 Qwen-VL 视觉塔。

语言光路与图像光路各自拥有参数，不共享；大、小版采用相同光学几何。
每侧光路包括实际光学 Router、top-2 专家、global；不是电子类别分类器。
复用 T01 的角谱传播、探测、重载、DC20 训练扰动与光学 Router 实现。

## 每侧相位参数：不能混淆物理范围和可训练区域

- Router：224×224 独立训练相位，按原项目导出为 478×478 有效面（周边填充）。50,176 个参数。
- 专家：4 个 224×224 分区，布局在 478×478 有效区域。200,704 个参数。
- Global：478×478 独立训练相位。228,484 个参数。
- 每侧共 479,364，两侧共 958,728。并不是每个分区都独占整张 478×478 可训练面。

若要求 router 与每个专家都单独使用整张、逐像素训练的 478×478 面，那是另一种光路，
不能声称这是原项目的 224/478/518 分区实现。当前修正忠实复用原分区光路。

## 语言流程

prompt → 冻结 Qwen tokenizer / token embedding → 去除 padding、保留有效 token 顺序
→ 已训练的 2048→640（大）或 2048→512（小）投影。

第一阶段：同一输入 x 分别进入 Qwen-style block 1 与光学 Router/专家传播，
经过独立电子读出后做 detached-RMS 同尺度融合。
第二阶段：同一融合结果分别进入 Qwen-style block 2 与 global 光学输入重载/传播/读出，
再进行同尺度融合。最后 RMSNorm、取最后一个有效 token。

电子 Qwen-style block 是 pre-RMSNorm、RoPE causal attention、残差、pre-RMSNorm、
SwiGLU MLP、残差。光学支路看完整条件 prompt，不宣称整个光电头是严格因果语言模型。
当前任务不做自回归文字生成，只提取完整指令条件。

大版取语言隐藏特征经已有 640→2048 bridge 与已有 condition adapter；
小版经已有 512→160 condition projection。每次训练和推理都实际执行语言光路。
只允许缓存冻结 token embeddings，不允许用缓存 pooled Qwen 特征绕过语言光电头。

## 图像流程

大版：输入 RGB → 冻结 SD-Turbo VAE encoder → reference latent；与四通道 Gaussian noise
拼接 → 已缩窄、剪 attention 的 BK-SDM UNet（128/256/512）
→ mid 处的新光电模块 → UNet decoder → 一次 latent residual 编辑 → 冻结 VAE decoder → RGB。
小版：RGB 与三通道 Gaussian noise → 已训练 CNN stem/down1/down2/down3
→ 新光电瓶颈 → 已训练 up3/up2/up1 → RGB residual 头和已有学习 source gate → RGB。
无 GT 物体掩码贴回、无检索商品贴图、无固定背景图的推理粘贴。

图像光电模块内部先将特征池化为 14×14＝196 个 token（不改物理相位范围），
注入同一文字条件。两个阶段各保留可学习电子 residual transform，与 expert/global
支路读取同一输入，再做 RMS 融合。恢复原特征尺寸时保留高分辨率特征残差。
大版保留已训练的光路外围投影；不再以 identity 的零参数充当整个电子分支。

## 约束与边界

- 四个融合 alpha 均约束到 [0.4, 0.75]。alpha 不是整网物理算力百分比。
- 同一权重覆盖背景、目标、联合修改；无 design_router 辅助商品分类器。
- 无多轮扩散循环、无 GAN。局部蒸馏仅在训练时使用旧紧凑文字头，推理不带老师。
- 保留现有 ABO-clean 数据，未引入椅子、高级材质任务或新类别。
- 固定目标监督及原数据模板边界依然存在；此次修架构不等于获得开放式随机生图能力。
- 本轮未测延迟，不把 FFT 仿真时间称作硬件时间。

## 验证和文件

`audited_v2_small.json`、`audited_v2_large.json`：GPU 实际前向、相位梯度审计。
语言/图像四阶段都用张量值比较检查 E/O 输入一致；router、选中专家、global 有梯度。
top-2 未选中专家零梯度正常。
`test_audited_unified.py`：固定物理几何、相位数量一致、拒绝超长 token 行、alpha 下限的回归测试。

代码：`LightGenV2/tasks/t12_text_to_image/audited_unified.py`；
统一 audit/train/infer 入口：`audited_unified_run.py`。

服务器新权重目录位于 `/DATA/DATA1/guest3/t12_assets/runs/abo_audited_v2_*`，
每份权重带唯一 architecture 标识，旧推理脚本不能把它无声当作旧版加载。
训练完成后进程退出，释放所用显卡；不留下后台训练。

## 此轮结果：结构通过，效果不通过

4 项新回归测试及 1 项旧文字头测试共 5 项通过。两版迁移前后均通过 GPU 前向、
四阶段并行输入检查及 router/选中专家/global 相位梯度检查。

每版从原权重先适配 100 步，再继续 1000 步，并使用旧紧凑文字头作局部特征老师。
小版最后 48 个验证前缀样本 MSE 为 0.049390，大版为 0.063549；
这是部分验证样本的诊断值，不是全验证集结果，也不是 FID 或生成质量综合指标。
预览显示物体指令串类及替换不准，**不得提升为正式效果版本**。
不能将这些语义错误解释成光学衍射。
alpha 在 0.494–0.500 范围内，均不低于 0.4；报告中的专家分布仅是最后 batch，
不能用它证明全数据集的专家利用率均衡。

- `small_adaptation.json` / `large_adaptation.json`：参数拆分、训练诊断和 alpha。
- `small_adapt1000.jpg` / `large_adapt1000.jpg`：输入、GT、修正版生成的失败分析预览。
- `audited_v2_small_adapted.json` / `audited_v2_large_adapted.json`：适配后相位梯度审计。
- 小权重：服务器 `/DATA/DATA1/guest3/t12_assets/runs/abo_audited_v2_small_adapt1000/adapted_model.pt`。
- 大权重：服务器 `/DATA/DATA1/guest3/t12_assets/runs/abo_audited_v2_large_adapt1000/adapted_model.pt`。

架构版本区别不可省略：本次修复“语言缺光、物理几何错误、没有可学习电子分支、
辅助分类器无意义”等问题，但**不是把原始 Qwen-VL 预训练 language/vision 层直接换光的版本**。
现有紧凑 Qwen-style 头与 CNN/UNet 图像路径仍保留；不能宣称已经完全实现原始 Qwen-VL 主干替换。
