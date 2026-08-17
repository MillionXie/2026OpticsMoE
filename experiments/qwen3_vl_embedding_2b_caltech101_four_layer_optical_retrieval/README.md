# Caltech101 四层联合光电检索

这是独立于 Language-only 消融的新工程。原始 Qwen3-VL-Embedding-2B 保持冻结，
紧凑电子 Mixer、MoE4 router、四组相位/readout 和四个 sigmoid 融合门从随机初始化
开始联合训练，不依赖纯电子 checkpoint。

四个物理 CCD 边界依次为：

1. Vision MoE4 expert；
2. Vision global；
3. Language MoE4 expert；
4. Language global。

Vision 每层是 `2D depthwise Mixer || optical CCD`，Language 每层是 causal 1D
depthwise Mixer 与 optical CCD 并行。两条支路均输出 192 维 token，通过独立可学习门控
相加。两个模态内部都是第一层融合结果重新编码后，才进入第二层光路。

```text
Vision:  [Nv,1024] -> Linear/LN -> [Nv,192]
         -> Mixer2D_1 || MoE4 expert optics -> E1 + sigmoid(gv1)*O1
         -> Mixer2D_2 || global optics      -> LN(E2 + sigmoid(gv2)*O2)
         -> Linear + gated residual -> [Nv,1024] -> Qwen main merger

Language:[Nl,2048] -> Linear/LN -> [Nl,192]
         -> causal Mixer1 || MoE4 expert optics -> E1 + sigmoid(gl1)*O1
         -> causal Mixer2 || global optics      -> LN(E2 + sigmoid(gl2)*O2)
         -> mean+max pooling [384] -> LN/Linear -> 64-D L2 embedding
```

原始 Qwen 权重始终冻结；这里的“联合训练”指上述电子 Mixer、两套 MoE4 router、
四个光学阶段、四个融合门和 64 维 readout 从第一步同时优化。训练不加载纯电子实验
checkpoint，也不启用 DeepStack 或 teacher KD。

正式目标保持精简：`1.0× supervised contrastive + 1.0× episodic prototype
retrieval CE + 0.02× CCD operating-point loss`。router 仍由检索梯度学习，但不增加
balance/importance 辅助 loss。

CCD 不做背景扣除。模型内只做逐帧均值尺度归一化、clip/log1p、478→224 pooling
和本层独立 readout。

硬件流程使用 `478×478 uint8 PNG` 作为唯一 CCD 传输格式；服务器不再保存逐样本
float32 registered CCD 或 simulation CCD。SLM 仅传输 `478×478` 紧凑 PNG，实验室按
固定规则放大两倍并居中重建完整 BMP。

详见 [DATA_PIPELINE.md](DATA_PIPELINE.md) 和 [RUN_COMMANDS.md](RUN_COMMANDS.md)。
