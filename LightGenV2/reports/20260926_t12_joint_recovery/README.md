# 本轮最终候选：语言＋图像光电统一编辑器

2026-09-26：本轮联合训练、完整测试和权重导出完成。旧正式效果权重未覆盖。
本目录是此次恢复训练的唯一交付入口，不需要从 pilot / adapt1000 中选模型。

## 直接看结果

- [大版九组概览](large_final_overview.jpg)，[大版十八组完整预览](large_full_preview.jpg)。
- [小版九组概览](small_final_overview.jpg)，[小版十八组完整预览](small_full_preview.jpg)。
- 每行依次为输入、GT、模型输出，展示灯、桌、靠垫的背景 / 目标 / 联合编辑。
- [大版测试指标](large_final_test.json)，[小版测试指标](small_final_test.json)。
- [Tokenizer 和文字头逐层说明](TEXT_HEAD.md)。

## 只使用这两份新权重

| 用途 | 本地文件 | 服务器原始位置 |
| --- | --- | --- |
| 大版，优先用于细节演示 | `weights/large_unified_audited.pt` | `/DATA/DATA1/guest3/t12_assets/runs/abo_audited_v2_large_latent_rank64_1500/adapted_model.pt` |
| 小版，低参数量演示 | `weights/small_unified_audited.pt` | `/DATA/DATA1/guest3/t12_assets/runs/abo_audited_v2_small_joint_continue2000/adapted_model.pt` |

两份各自同时支持背景替换、目标替换、联合修改。没有高级材质分支，不是六套 task 权重。
两者输入输出都是 256×256，均保留两层可训练 Qwen-style 文字头和真实仿真光路。

当前权重文件包含完整编辑器 state_dict，但构建网络仍使用 `--source` 的旧模型配置
及大版的 UNet/VAE/adapter 初始资产路径，再严格加载新的完整 state_dict。
冻结 Qwen tokenizer/token embedding 也属于共享外部资产，不包含在这两个文件中。
因此不要声称两个 PT 是“无代码、无配置、无共享资产即可独立运行”的文件。
使用 `LightGenV2.tasks.t12_text_to_image.audited_unified_run infer`，不要用旧通用推理脚本。

大版 source 是 `abo_unified_expanded_143m_true256_v2/unified_editor.pt`；
小版 source 是 `abo_unified_expanded_qwenmini_9m_256_sourcegate_v1/best_model.pt`。
`--checkpoint` 必须指向本轮新权重；大版 PCA rank=64 会从新 checkpoint 自动恢复。

SHA256：

- 大版：`786e24ed02c9bffd12beed6249f52b89a5b1549427af4785bf348ea10e39b1f0`。
- 小版：`e889ccc32f339e2870697e7c79fd83517569090ae714c95c447dca62ad186082`。

## 完整测试集结果

测试包含 2304 个图文编辑对，每种模式 768 个。测试数据不参与最优权重选择。
最优权重由跨类别均匀抽取的 144 个验证对选择。MSE/L1 在 [-1,1] 图像范围计算，
PSNR 使用峰值范围 2，即 `10*log10(4/MSE)`。

| 指标 | 大版 | 小版 |
| --- | ---: | ---: |
| 神经网络参数，包括冻结 VAE | 144,630,618 | 11,137,746 |
| 固定条件矩阵 / 统计值 | 5,125,248 | 0 |
| 上述合计（预算口径） | **149,755,866** | **11,137,746** |
| 冻结 Qwen token embedding，按约定单列 | 311,164,928 | 311,164,928 |
| 纯光学相位参数，已包含在 NN 参数中 | 958,728 | 958,728 |
| 整体 MSE ↓ | 0.004986 | 0.004009 |
| 整体 PSNR dB ↑ | 29.043 | 29.991 |
| 整体 L1 ↓ | 0.042099 | 0.030038 |
| 整体边缘 L1 ↓ | 0.092261 | 0.087252 |
| ImageNet-Inception FID 变体 ↓ | 48.677 | 104.613 |
| KID，使用同一特征提取器 ↓ | 0.008885 ± 0.001440 | 0.023837 ± 0.003717 |

FID/KID 使用 torchvision ImageNet Inception-v3，**不是 canonical TensorFlow FID**，
不得直接与外部论文标准 FID 数字比较。这里是固定目标商品目录与组合背景的条件编辑，
不是开放式文生图 benchmark。重复 GT、简单背景也影响指标解释。
像素误差更低不代表纹理更好；本轮大版在该分布指标上明显更好，但仍需看预览。

## 实际训练，而非只训练新插入模块

1. 从上一轮光电修正权重继续训练，解冻文字头、图像主干、光路及条件接口；
   冻结共享 token embedding 和大版 VAE。
2. 小版沿最优检查点路径完成 1000＋2000 步联合训练，batch=4。
   中途因 CUDA 卡号与显示号不一致，从共享 A100 迁移到空闲 4090，采用 GPU UUID。
   原目录记录了迁移前未被选中的额外步骤，不能将其混算成最后权重的继承步数。
3. 大版先联合训练 3000 步，batch=2，再继续 1500 步，batch=4。
   第二段将 PCA rank 128→64，保留前 64 个方向及已训练的对应系数行，
   并增加冻结 VAE 对 GT 的 latent 监督。不从头初始化整个模型。
4. 损失包括 MSE、0.1×L1、0.02×边缘 L1；旧紧凑文字头仅作训练时局部特征老师。
   小版文字蒸馏权重 0.02，大版最后一段 0.01；大版最后一段 latent MSE 权重 0.1。
5. 先使用理想 DC20 仿真恢复画质，不施加随机 DC20 硬件扰动；
   没有移除光路、停止相位梯度或缩小物理 mask。
6. 每 500 步验证并保存最优权重。本轮两份均选中了各自最后一段的末尾检查点。

验证 MSE：小版 0.048688→0.003845；大版 0.096199→0.004749。
大版 rank 压缩后从 0.020390 继续恢复至 0.004749。
训练曲线见 `small_training.json` 和 `large_training.json`；前后验证值均基于固定 144 个验证对。

## 架构与真实性边界

- tokenizer、词表、chat template、冻结 embedding 不变。文字头是手写窄 Qwen-style
  Transformer，不是从原 2B checkpoint 直接剪下来的两个预训练 block。
- 语言侧两个电子 Transformer 和图像侧两个可学习电子 residual，
  分别与 expert/global 同输入并行、RMS 同尺度融合。
- 复用原项目 224/478/518 的真实角谱传播仿真和光学 Router。
  每侧 router 224²＋专家 4×224²＋global 478²＝479,364 相位参数，两侧共 958,728。
  router 的导出有效面为 478²，专家是其中的分区，不声称每个 tile 都是整张 478² 自由参数。
- 推理没有 GT 像素粘贴、商品检索或硬掩码合成。GT 构建本身仍使用程序生成背景与
  ABO 透明物体组合；网络学习这些监督目标，任务属于受控条件编辑。
- 原图参考、文本条件和 Gaussian noise 都输入网络；没有多轮扩散循环、没有 GAN。
  本轮未改变“seed 变化较弱”的问题，不宣称开放式、多样化随机生成已经解决。
- 数据仍为灯、桌、PILLOW 类靠垫/座垫，每类 4 个指定目标，共 12 款。
  PILLOW 中存在座垫/靠垫组合，外观可能像椅垫，但不包含完整椅子类别。

## 尚未解决

- 小版目标替换纹理明显较平滑，部分输入残影仍存在。
- 大版联合修改仍有局部模糊、桌腿残影和边缘变化，不能称作完全恢复或最终论文成品。
- 语言路由仍只选中专家 0、1；图像侧利用率也不是完全均衡。
  全测试集的分布在 JSON 中，不用最后一个 batch 代替全数据集统计。
- 以上缺陷不能直接解释成衍射。实际硬件验证、鲁棒扰动训练和 baseline 时间对比未在本轮执行。

本轮计算使用不超过两张卡，结束后释放本任务进程，不停止其他用户任务。
