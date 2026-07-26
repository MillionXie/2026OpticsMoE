# Architecture

## Vision

```text
RGB image + caption
→ frozen Qwen processor/chat template
→ frozen vision patch embedding
→ packed visual hidden [sum(Nv), Dv]
→ shared frozen PCA_vision encode
→ [sum(Nv), 224]
→ Optical MoE16 stage 1 → signed tap 1 [sum(Nv),224]
→ Optical MoE16 stage 2 → signed tap 2 [sum(Nv),224]
→ Optical MoE16 stage 3 → signed tap 3 [sum(Nv),224]
→ Optical MoE16 stage 4 + global phase → signed tap 4 [sum(Nv),224]
```

在三个 DeepStack provider block 和 final block 处，仅为调用 frozen Qwen
downstream module 执行 PCA decode：

```text
signed tap [Nv,224]
→ fixed PCA_vision decode [Nv,Dv]
→ corresponding frozen deepstack merger or final vision merger
```

光学 stage 的内部状态始终为 224 维；PCA decode 不是 trainable adapter。

## Language

```text
frozen token embedding + frozen visual-token injection
→ multimodal hidden [B,S,Dl]
→ shared frozen PCA_language encode
→ Optical Language MoE16 stage 1..4
→ signed taps [valid_tokens,224]
→ final fixed PCA_language decode [B,S,Dl]
→ frozen final RMSNorm
```

前三个 stage 返回 Qwen hidden 后，Qwen 原生 DeepStack 会加上视觉 embedding。
下一 optical stage 只把这个加性 delta 用同一 PCA components 投影到 latent
空间；零 injection 保持严格为零，不会错误减去 PCA mean。

## Optical stage

```text
[T,224] latent
→ LayerNorm(224)
→ Softplus
→ zero-pad token rows to [224,224]
→ one electronic top-4 routing decision
→ weighted direct amplitude loading into 16 expert apertures
→ phase-only expert plane
→ angular-spectrum propagation
→ square-law detector
→ crop 986×986 CCD ROI
→ adaptive average pool 224×224
→ non-affine LayerNorm
→ signed_readout
```

Stage 1–3 使用：

```text
reload_amplitude = ReLU(signed_readout + optional projected DeepStack delta)
```

随后沿用第一次计算出的 routing weights 重新加载到下一 expert plane。Stage 4
在 expert propagation 后经过 986×986 global phase，再传播到最终 CCD。

## Loss

```text
vision_loss   = mean(masked token-wise LN-MSE over four vision taps)
language_loss = mean(masked token-wise LN-MSE over four language taps)

total = vision_loss
      + language_loss
      + router_balance_weight × router_balance_loss
```

`train_vision` 和 `train_language` 仅包含对应 stack loss；`train_joint` 同时包含
两者。Padding token 永远不会参与 loss。
