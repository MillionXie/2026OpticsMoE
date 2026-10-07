# T07 ABO 图搜图：独立审阅版

## 这份是什么

从原 T07 中保留经过训练的数值核心，重新组织单一入口、训练目标、配置和资产校验。没有复制整个历史工程，没有运行时跨目录导入，也不会加载完整 Qwen 2B。保留数值核心和 state_dict 键名是为了验证整理没有改变预测，而不是重新设计网络。

当前交付权重目标是 `high_alpha_retrieval_20260910` 的 epoch15 EMA：**Hit@1 68.9583%，去光 64.5833%，差 4.375 个百分点**。不是旧的低 alpha 70.21% 权重；还未达到 75%。固定权重隔离复评状态见 `docs/VERIFICATION.md`。

本包支持**仿真复评及本地/服务器继续训练**，不是实验室硬件包。没有宣称包含相机、SLM SDK、LUT 标定或 BMP 播放自动化。现有模型有实测 CCD 张量注入接口，但接设备、单位标定及播放仍需单独工程。组会完整分析在 [GROUP_MEETING.md](docs/GROUP_MEETING.md)。

## 文件只分这几类

| 位置 | 用途 |
|---|---|
| `lightgen_abo/frontend.py` | 独立冻结输入权重：patch、位置、merger、所需词嵌入 |
| `model.py`、`optics.py` | 电残差、同尺度融合、真实联合衍射与光 router |
| `objectives.py`、`training.py` | 训练专用损失与单一续训流程 |
| `data.py`、`io.py`、`runtime.py` | 数据合同、资产校验、固定权重复评 |
| `configs/train.json` | 高 alpha 检索续训参数；不写服务器绝对路径 |
| `assets/` | 随包提供 best.pt、processor、SHA 清单；不进 Git |
| `runs/<run_id>/` | 每次评估/训练的独立输出，不覆盖 |
| `tests/`、`tools/` | 独立性/数值测试与显式资产打包脚本 |

## 完整推理结构

```text
一张商品 RGB 图 + 固定英文 prompt
  RGB → 中心正方形裁切 → 224×224（不是拉伸，可能截掉细长物体）
  冻结 Qwen patch Conv3d + 插值位置嵌入 → [B,196,1024]
  V 输入 Linear+LN → [B,196,192]
    电残差 E1 ───────────┐
    光 router → Top2 → 专家传播 O1 → 同尺度融合 F1
    电残差 E2(F1) ───────┐
    全局相位传播 O2(F1) ─┴→ 同尺度融合 F2
  V 输出 Linear → 1024，另有原 patch 输入 skip
  冻结 Qwen merger：每相邻 2×2 patch 合并 → [B,49,2048]
  prompt tokenizer + 冻结紧凑词嵌入，与 49 个 image token 按模板替换合并
    → [B,77,2048]（49 image，另 28 为文本/特殊 token）
  L 输入 Linear+LN → [B,77,192]
    电残差 E1 + 光 router/Top2/专家 O1 → 融合
    电残差 E2 + 全局相位 O2 → 融合 → [B,77,192]
  token mean + max 拼接 → [B,384] → LN → Linear → [B,64] → L2
  与 120 个训练商品的 12 视图中心做余弦检索 → 排序结果
```

“V/L 各三层光”在这里是各 **router、expert、global 三次光学采集**，总共六次，不是每个专家连续三块权重。四专家并排，Top2 激活；router 的衍射/探测是光学，softmax、Top2 和振幅 fanout 权重由电子计算，不称全光逻辑。

V 残差是 3×3 depthwise 二维卷积+pointwise、192→384→192 MLP；L 残差是因果 kernel5 一维卷积+pointwise、同宽 MLP。代码 `blocks` 仅为旧权重键名，不代表 Transformer。共 29,152,256 个冻结参数、2,782,485 个可训练参数，无 attention、无完整 Qwen Transformer、无 VGG。

冻结前端仍保留 Qwen 的 merger MLP（4096→4096→2048），不能说电子只有小读出头。冻结输入权重**不等于**完整预训练语义网络。固定 prompt 是：

> Represent this catalog product image for category-aware visual similarity retrieval.

manifest 中类别、商品标题不作为模型输入。输入模板/分辨率改变会明确报错，不能直接用这个 compact token 表做任意 prompt。

## 光学与融合合同（不要误读）

- 532 nm，10 cm；模拟等效像素 17 μm；478×478 有效场，FFT 扩到 518×518；四个 224×224 专家，缝隙30。224 是模拟网格，不是 8 μm 相位 SLM 原生 BMP 大小。
- latent 192 通道投影到 224 列、softplus，token 沿行放置，下方补零到224行，RMS=0.5；四槽加权场一起传播。RMS约束不等于逐像素振幅≤1，硬件映射必须另行审计。
- 相位 `2π sigmoid(raw)`；标准浮点优化，没有 8-bit 直通量化、k 空间过滤或像素偏移。
- 同尺度融合将 E/O 各自按 RMS 归一化，再做 `(1-alpha)E + alpha O` 并恢复尺度。alpha 下界0.4001、上界0.8。当前四个 alpha 约0.430–0.440；**不是整网光学能量/贡献百分比**，尤其 V 还有原输入 skip。
- CCD 当前并非只做线性归一化：`除均值 → 截到12 → log1p → pool224² → 每行LayerNorm → ReLU → 取前L行 → Linear192`。为了固定权重数值等价，本整理版没有暗改这段。语言 L=77，保留77/224行，存在需要专项验证的读出瓶颈。
- 训练中每四批一批启用光学噪声：振幅/相位未调制分量各20%–30%的场叠加，轻微截断高斯CCD噪声/增益等。正常主指标为理想仿真评估；不能称该指标已在20%–30%固定直流或真实硬件下验证。

## 数据、训练与选模

数据是10类、每类20商品、每商品12图。train=120商品1440图；val=40商品480图（当前不用）；test=40商品480图。按商品隔离，但观察到各类ID范围依次分离，不能直接声称随机划分。实际检索成功定义是**同类的其他商品**，并非找回同一个SKU。

训练 gallery 只用120个训练商品的特征中心，每epoch刷新；训练query排除自身商品，同类其他商品为正，异类为负。分类proxy、光支路辅助头只训练时用，推理没有这些额外分支。

默认续训30epoch×64step，训练batch40（10类×4商品各取一视图），`--batch-size` 只控制评估编码批量。前25epoch联合训练，后5只调读出；EMA0.99，每5epoch比较 live/EMA 的 test Hit@1（同分比较mAP@10），选best；不设独立验证集。明确属于 **test-selected** 结果，不是独立无偏最终测试。

续训是加载 best 后重新建立优化器，不是恢复 Adam/RNG 的中断续跑，不保证重训逐位一致；训练仍保存 `best.pt`、`last.pt`，没有每5epoch多份权重。原始优化试验继续在 LightGenV2，不在审阅目录堆 run。

## 如何正确读取一片mask

`best.pt` 是包含 `metadata` 与 `state_dict` 的字典，不能把整份checkpoint当一张相位图；也不能把raw参数直接当弧度。

```python
import torch
checkpoint = torch.load('assets/best.pt', map_location='cpu', weights_only=True)
raw = checkpoint['state_dict']['vision.optics.experts.0']
phase_radians = 2 * torch.pi * raw.float().sigmoid()  # [224,224], rad
print(phase_radians.shape, phase_radians.min(), phase_radians.max())
```

V/L专家键名分别为 `vision.optics.experts.0`～`.3`、`language.optics.experts.0`～`.3`；router键为`<modality>.optics.router.raw_router_phase`（224²）；global键为`<modality>.optics.global_phase`（478²）。这些是模拟相位，不是经LUT/像素尺寸变换后的可直接播放BMP。本包只含选中best，不含每5epoch演变快照。

![最佳相位概览，单位0至2pi](docs/figures/phase_masks.png)

相位预览随本地/内部ZIP提供，不以图像代替真实参数。模型从已有训练权重续训；若新建模块raw初始化为0，对应实际相位π，不能把当前已训练权重说成仍为零初始化。
