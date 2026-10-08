# Tokenizer 和 Qwen-style 文字头的准确含义

本轮没有更换 tokenizer、词表、chat template 或冻结的 token embedding。
缓存元数据指向同一 Qwen3-VL-2B-Instruct 快照 `89644892e4d85e24eaac8bacfd4f463576704203`。
636 条 prompt 的 embedding 形状为 [636,38,2048]，有效长度 31–38，未触及 64-token 上限。
token embedding 共 311,164,928 个参数，冻结，按约定单列；它不是 tokenizer。

## 文字头不是“只有 tokenizer”

输入 token IDs [B,L] → 冻结 Qwen embedding [B,L,2048] → 可训练线性投影
→ 两个可训练 Transformer block（分别融合光学结果）→ RMSNorm
→ 最后一个有效 token 的隐藏特征 → 图像生成条件。

| 配置 | 大版 | 小版 |
| --- | --- | --- |
| 投影 | 2048→640，无 bias | 2048→512，无 bias |
| 层数 | 2 | 2 |
| 注意力头数 / 每头维度 | 10 / 64 | 8 / 64 |
| Q、K、V、O 投影 | 各 640→640，无 bias | 各 512→512，无 bias |
| MLP gate/up | 各 640→1536，无 bias | 各 512→1024，无 bias |
| MLP down | 1536→640，无 bias | 1024→512，无 bias |
| 输出到条件 | 640→2048 bridge，再已有 adapter | 512→160 condition projection |

## 每个电子 Transformer block 的算式

1. `u = RMSNorm(x)`。
2. `q = RoPE(Wq u)`，`k = RoPE(Wk u)`，`v = Wv u`。
3. causal 多头注意力，排除 padding：`a = Attention(q,k,v)`。
4. `h = x + Wo a`。
5. `z = RMSNorm(h)`。
6. `y = h + Wdown(SiLU(Wgate z) * Wup z)`。

这是手写的窄 Qwen-style block。保留 RMSNorm、RoPE、残差和 SwiGLU 的设计思路，
但采用完整多头注意力和标准 RoPE，不逐项复制 Qwen3-VL 的 GQA、位置配置等细节。
其权重来自此前已训练的紧凑文字头，不是从原 2B checkpoint 挑选两层直接继承。
因此“仿照并缩窄的 Qwen-style block”准确，“原 Qwen block 只剪了层数”不准确。

## 光怎样加入这两层

第一层：电子 block 1 和光学 Router/专家读取同一 `x`，输出形状同为 [B,L,D]，RMS 融合。
第二层：电子 block 2 与 global 光学重载/传播读取同一第一层融合结果，再 RMS 融合。
alpha 始终在 [0.4,0.75]；每个 prompt 每次前向都实际执行这两阶段，
不使用缓存 pooled 文字特征或标签查表替代文字头。

电子注意力使用 causal mask，光学支路读取完整条件 prompt。整个文字头不是用来
自回归生成文字的严格因果语言模型，而是用于图像生成的完整指令编码器。

图像条件仍由文字头产生；目标商品标签不输入模型。不引入训练后的商品检索/贴图。
