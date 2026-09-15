# 低扰动光电融合方案

目标是保留已经验证的电子路径，把光路作为可关闭的残差修正，而不是再次替换电子 block。

推荐第一阶段只在 Vision 中接入一条光学支路：

```text
x -> Electronic 2D Mixer block 1 -> e1 ------------------------+
                              \-> optical propagation -> CCD    |
                                  -> compact electronic decoder |
                                  -> zero-init residual r_opt ---+-> e1 + tanh(g) * r_opt
                                                                  -> Electronic 2D Mixer block 2
```

`g` 初始化为 0，电子输出因此在接入瞬间严格不变；最终 decoder 的最后一层也使用零初始化。光路异常时可以把 gate 强制设为 0，回退为原电子模型。第一阶段不修改 Language Mixer、main merger、pooling 和 readout。

训练顺序：

1. 加载已训练电子 checkpoint，冻结电子主干和 main merger。
2. 在仿真噪声下只训练光学 phase、CCD 后紧凑 decoder 和 gate。phase 学习率应比 decoder 低约一个数量级。
3. 使用错位、phase dropout、强度缩放、读出噪声和可选 k-space 约束；错位范围应覆盖实测的几十像素偏差。
4. 光学残差稳定后，只以很小学习率解冻 Vision block 2 和 readout；Language 继续冻结。
5. 用实际 CCD capture 微调 decoder/gate。若硬件置信度不足，部署时关闭 gate。

损失保持任务主导：`SupCon + episodic prototype CE`。为了防止接入光路后破坏电子 embedding，可额外加入小权重部署一致性项 `1 - cosine(z_hybrid, stopgrad(z_electronic))`；它使用同一个电子 checkpoint，不需要外部 teacher。

只有当单条 Vision 光学残差在真实光路上稳定带来收益后，才考虑第二条 Vision 光学支路。Language token 没有二维结构，暂不建议接光；若以后需要，也应采用同样的并联残差和零初始化门控，不能替换已经训练好的 Language Mixer。
