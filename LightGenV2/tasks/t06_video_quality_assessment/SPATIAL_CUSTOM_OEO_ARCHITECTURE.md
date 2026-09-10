# Spatial-4 自写卷积光电模型结构

## 一句话结构

每条视频均匀抽取 4 帧。Qwen3-VL 只在离线准备阶段提供官方图像
`patch_embed + position embedding` 和文本 `tokenizer + embed_tokens`；正式学生网络不执行
Qwen 的 Vision Transformer、Language Transformer、attention、merger 或 LM head。学生网络
只有两个主分支：一条电子残差支路和一条光学 Top-2 MoE 支路；两者在每层按同尺度公式融合，
最后由一个轻量电子读出头输出一个 Spatial MOS。

## 输入与 Qwen 边界

```text
一条视频
  -> 均匀抽 4 帧，每帧 RGB 224x224
  -> 离线 Qwen 图像前端
       每帧 14x14 个 patch token，每个 1024 维
       得到 [B,4,196,1024]
  -> Linear 1024->192
       得到学生图像 token [B,4,196,192]

固定 Spatial prompt
  -> Qwen tokenizer + chat template
  -> 冻结的 Qwen embed_tokens
       [B,38,2048]
  -> Linear 2048->192
       [B,38,192]
```

“离线”表示这些张量可一次生成后保存在 `.pt` 缓存中。训练学生、仿真评估和依据缓存导出
SLM 图案时，既不加载 2B 完整权重，也不调用任何 Qwen block。只有在换数据集、换抽帧或
重新生成缓存时，才需要单独运行 Qwen 前端缓存脚本。

## 自写电子卷积位于哪里

自写模块不是 Qwen 后面的分类器，也不是最终 MOS 旁路。它位于第一层电子残差 `E1` 内，
位置如下：

```text
Qwen patch+position token -> 192维基础图像 token V
                                      |                  |
                                      |                  +-> 光场编码 -> O1
                                      v
                               原有轻量电子映射 E1_base

原始4帧 RGB -> 自写小卷积 -> [B,4,196,192] 校正量 C

E1 = E1_base + C + 任务内质量校正
F1 = 同尺度融合(E1, O1)
F1 -> 第二层电子/光学 -> 融合 -> 文本两层电子/光学 -> 融合 -> 单一MOS头
```

因此关闭光学支路时性能会下降；自写卷积本身不能直接输出 MOS，也不能绕过后续三次光电
融合。当前质量侧缓存同样只作为 `E1` 内部的校正量，不构成第三个预测分支。

## 自写卷积的精确结构

每帧独立经过以下基本算子，所有层均由本项目直接定义：

```text
RGB [3,224,224]
 -> Conv3x3, stride2, 3->24  + GroupNorm + GELU   [24,112,112]
 -> Conv3x3, stride2, 24->48 + GroupNorm + GELU   [48,56,56]
 -> Conv3x3, stride2, 48->64 + GroupNorm + GELU   [64,28,28]
 -> Conv3x3, stride2, 64->96 + GroupNorm + GELU   [96,14,14]
 -> Conv3x3, 96->96 + GroupNorm + GELU
 -> Conv3x3, 96->96 + GroupNorm，与输入做局部残差相加，再 GELU
 -> 展平为 196 个 token
 -> LayerNorm(96) -> Linear 96->192 -> GELU -> Linear 192->192
 -> 1.4*tanh 限幅
```

该模块共 316,568 个参数。最后一个 Linear 零初始化，所以第一次接入已有模型时不会突然
破坏 E1；预训练后再联合微调。它不包含第三方 backbone、分类器、attention、Transformer
或独立质量分数头。

## 四层光电主干

四层顺序不变：

1. Vision expert：四帧 2x2 并行；每帧四个 109x109 相位专家，光 Router 选 Top-2。
2. Vision global：继续在同一 478x478 有效光场上传播和融合。
3. Language expert：图像摘要与 prompt token 进入语言侧光电层，光 Router 选 Top-2。
4. Language global：最后一次光电融合，再进入唯一 MOS 读出头。

所有融合先分别计算电子和光学 RMS，再用

```text
F = rE * ((1-alpha) * E/rE + alpha * O/rO) / rms(mixture)
```

把两支路放到可比较尺度，避免电子数值大而淹没光。四层 alpha 独立，受正式 profile 的
范围约束。正式消融只在同一个 checkpoint 上旁路光学，不能另训一个纯电子模型冒充贡献。

## 训练策略

1. 训练期教师只用于把旧轻量 E1 特征蒸馏到自写卷积；发布 checkpoint 中不含教师。
2. 先只优化自写卷积，再联合自写卷积与单一 MOS 头。
3. 前两步稳定后，才以更小学习率放开光学相位和光 Router。
4. 论文选模按约定周期查看完整 test SRCC；正式目录只保留 best 和 last。
5. 每个候选都报告同 checkpoint 的 optical-on/off、四层 alpha、mask 改变量和逐物理槽专家占比。

## 独立发布边界

仓库内由 LightGenV2 profile 锁定后端源码 SHA，避免两份模型静默分叉。给服务器或实验室的
正式 ZIP 则会把运行所需源码、配置、checkpoint、缓存合同、硬件导出/微调入口和 SHA256
manifest 一并放入包内；包内训练与评估不依赖仓库外的 `experiments/` 目录。完整 Qwen 权重
不是学生运行依赖，仅“从原始视频重建 Qwen 前端缓存”这一可选步骤需要它。
