# FFA：从预训练反馈到冻结完整强度响应

日期：2026-10-11（Asia/Shanghai）。本文件记录本轮研究假设与初筛协议；运行结果与实际启动参数另见本目录的运行记录。此目录是新实验，既有 `FixedFeedbackSFT` 源码、正式结果与权重保持原身份。

## 1. 从原对话保留的核心问题

用户希望沿着教授的动机研究：反馈通路可能在早期过程形成，随后主要由前向连接承担学习。教授原文的固定反馈微调结果尤其强调预训练，因此不能把合作方向改成普通随机反馈、换优化器，或将 PAT/DAT 已有的“物理前向、数字模型反向”重新包装为新方法。

已有 `FixedFeedbackSFT` 的准确结论是：**预训练源光学算子可以冻结为跨 stage 的反馈连接器，同时每层当前相位的局部导数和电子模块仍按 BP 计算。** 它并未证明冻结了完整光学更新过程，也没有证明能省去当前复光场或主要反向计算。此前对话对这些边界的纠正，本轮全部保留。

本轮先检验更强、也更容易失败的一步：**能否把一个光学 stage 从已知输入振幅到输出强度的响应结构整体固化在预训练点；后续只使用当前实数输入与当前强度误差，产生相位更新和向前一 stage 传递的误差？** 若这一步不成立，则先修正或放弃完整冻结响应，而不是用更强电子支路掩盖它。

这仍然是梯度下降类的近似学习方向研究；AdamW 只负责利用该方向更新参数。更换 AdamW 本身不会消除光场相位需求。

## 2. 已核验的工程载体

入口为 `FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone/model.py`（P11）与 `FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa/`（P12）。共享物理 stage 来自 `FixedFeedbackSFT/projects/d2nn_cifar10_high_performance_optical_backbone/optics.py`。沿用 `experiments.<project>` 公共模块名，避免同一 class 以两套模块名加载。

P11 的数据接口为：

```text
224×224 RGB
  → 冻结 Qwen Patch/Position Stem：196 个 1024-D token（无 Transformer）
  → 1024→224 adapter
  → [token-axis OEO → feature/channel-axis OEO] × 4
  → LSP 14×56×56 热图任务头
```

物理场张量为 `[batch, 3 banks, 224, 224]`，其中前 196 行是有效 token，另外 28 行是填充模式。token 轴与 feature 轴的单线长度都为 224。每层相位有 `3×224×224` 个参数，八层合计 `1,204,224`；可复用电子 body 有 `965,120` 个参数，本轮适配全部冻结。

token 轴传播前把 Qwen 的 2×2 block-major token 排列改为 row-major，传播后恢复。相位参数与源相位本身已经位于物理 row-major 平面，不再做相位置换。feature 轴没有此 token 置换。填充行参与物理传播，不能按 196 截断算子。

单个 token/feature 对是最小语义宏块，但包含两次独立传播、两次强度探测和重载。P11 假设各向异性/柱面光学系统沿一个轴传播、另一轴理想成像，不能当成普通二维自由空间。采用 1-D FFT 只是该既定模型的等价数值实现；必须先与原 2-D FFT 实现校验。

预训练源文件登记 SHA-256 为 `c3ad0b780dfbb3e5f8e1f7b7850c06fcb5c6d977e106f351b4602fcaadf210d2`，冻结 stem 为 `e3b12b274211d29f928eee95fdfc60b32d10751f1bbdc98cd63f0cccd0792485`。实际服务器路径及本次读取文件 SHA 由运行清单确认。9 月归档的旧 LSP phase-only common start 本轮核查后不可用；已适配 BP 的 best checkpoint 不能替代共同起点。本轮从 P11 源重建新头，不绕过旧 run 的身份检查、不原地恢复旧正式运行。

## 3. 当前 FA 缺失的环节

原 `_FixedFeedbackOptical` 返回当前复光场 `U_t`；`abs().square()` 在自定义算子外面。因此它接到的 `grad_output` 已经含有当前复光场。其局部相位 VJP 使用当前相位；只有输入振幅 VJP 使用源相位。

本轮将自定义学习接口放到**强度输出之后**。相机平方律之后的 mean/variance、ReLU、RMS、电子 residual、任务头仍沿用既有定义，电子计算可正常求导。自定义接口接收的是实数 `e = ∂L/∂I`，不能将旧 FA 的复误差直接误称为强度误差。

## 4. 数学定义：固定结构，动态误差

对一个光学 stage，已知输入振幅 `a` 为实数，相位为 `φ`，传播为 `P`：

\[
A(\phi)=P\,\operatorname{diag}(e^{i\phi}),\qquad
U=A(\phi)a,\qquad I=|U|^2.
\]

相机或仿真提供当前 `I_t`，电子端计算当前实数误差 `e`。定义：

\[
H_\phi(e)=A(\phi)^\dagger\operatorname{diag}(e)A(\phi),\qquad
q=H_\phi(e)a.
\]

精确局部导数为：

\[
g_a=2\operatorname{Re}(q),\qquad
g_\phi=2a\odot\operatorname{Im}(q).
\]

相位映射是 `φ=2π sigmoid(raw_phase)`，所以返回原参数的导数还需乘
`2π sigmoid(raw_phase)(1−sigmoid(raw_phase))`。phase 参数由整个 batch 共享，phase 导数对 batch 求和；输入振幅导数保留每个样本。

**source_full_response** 用冻结的预训练 `φ₀` 替换完整的 `H`：

\[
\widetilde q=H_{\phi_0}(e)a,\quad
\widetilde g_a=2\operatorname{Re}(\widetilde q),\quad
\widetilde g_\phi=2a\odot\operatorname{Im}(\widetilde q).
\]

前向仍使用当前 `φ_t`；`a` 与 `e` 仍随当前样本、当前前向与当前损失变化。固定的是响应结构，不是所有 batch 共用的一张数值 Jacobian，更不是冻结误差。除当前 raw-phase 到物理相位的已知控制映射外，光学更新式不使用当前相位或当前复场。

实现可采用因式分解：

\[
\widetilde q=
\operatorname{diag}(e^{-i\phi_0})P^\dagger
\left[e\odot P\left(a\odot e^{i\phi_0}\right)\right].
\]

它依然进行两次固定数字传播。不能宣称已经省掉 FFT、全部反向计算或证明了加速；它先移除对当前光学复场与持续更新数字光学导数的依赖。显式为每个输入/输出像素保存二次核会随单线长度立方增长；P11 八层所有独立线约需 60.4 billion 个复数元素，即约 483 GB complex64，因此本轮不构造该稠密核。

这一步可理解为逐 stage 使用源点响应的代理导数。与 PAT 的区别是本轮源响应冻结，PAT 通常在当前控制参数处由数字模型提供导数；这不构成对 PAT 的理论超越，也未证明文献新颖性。与 DAT 的“测量幅度配模拟当前相位”不同，本轮完整冻结响应无需当前光场相位估计。

## 5. 何时可能成立，何时应失败

如果传播近似酉且当前相位距离源相位较小，有：

\[
\|H_{\phi_t}(e)-H_{\phi_0}(e)\|
\le 2\|e\|_\infty\|A(\phi_t)-A(\phi_0)\|.
\]

相位屏差异可由逐像素的循环相位位移控制。因此“小漂移使源响应仍有用”是一个可检验的局部解释。该式不保证深网络全局收敛；归一化、小方差状态、深层连乘、强光路失配都可能放大误差，平均相位位移也不能替代最坏情况或任务子空间上的误差。

真正与下降相关的是同一当前状态的 `g_exact · g_candidate`。正余弦只表明存在足够小步长可能下降，不保证 AdamW 一定下降；负余弦、极端范数或长期无 phase 收益则支持失败。`e` 可含正负值，`H(e)` 不是必然正定，不以“二次响应”声称天然稳定。

需要避免把以下现象混为一谈：预训练表示可迁移；源反馈与初始前向匹配；预训练使完整冻结响应适用范围更大。当前初筛只有一个预训练源，不能单独区分三者。

## 6. 本轮五组与共同起点

| 组名 | 光学导数/学习方向 | 可训练参数 |
| --- | --- | --- |
| `head_only` | 无光学更新 | 任务头 |
| `bp_current` | 当前完整强度响应精确解析导数 | 八层 phase + 任务头 |
| `fa_source_connector` | 既有源连接器 FA；当前局部相位导数 | 八层 phase + 任务头 |
| `source_full_response` | 完整源强度响应冻结 | 八层 phase + 任务头 |
| `random_full_response` | 同形状固定随机相位的完整响应 | 八层 phase + 任务头 |

所有组前向初始相同；随机响应只改变反馈，不改变共同前向。任务头始终使用 exact BP。冻结 Qwen stem、adapter、每层 mixer、norm/gate 等电子 body，并保持 backbone eval，使冻结 mixer 内 dropout 不造成隐藏变化。任务头在适配训练时可正常进入 train 模式。

这是 FFA 新初筛表，不改写 P12 原论文四组方法表，不把新方法混入旧结果。

## 7. 短期初筛协议

使用 LSP，因为旧 phase-only 证据中 LSP 的相位适配增益清楚；Caltech phase-only 未超过 head-only，不宜先作为完整响应有效性的主要任务。旧 LSP 数值只作为任务选择依据，不作为本轮对照分数。

1. 复用 P12 校验过的 LSP/HR-LSPET 分割、可见性、裁剪与损失。按确定性样本 ID hash 选 `1024 train / 256 validation`，保存完整样本清单、来源分布及 SHA。该选样不是独立随机实验，需明确记录 seed。官方 test 保持封存，本轮不评价 test。
2. 从 P11 源创建新 LSP head。固定训练图像、冻结完整 body，缓存 float32 source final fields；以该缓存暖启动头 30 epoch，并用本轮 validation 选出新 shared start。该缓存约 600 MB 训练特征，不是光学训练数据，也不用于替代后续当前前向。
3. 五组严格从同一 shared-start 文件启动，记录文件与语义张量 SHA。共同起点只训练头，phase、源电子 body、stem 不变。
4. 先校验 1-D 当前响应与 legacy 2-D BP 的前向、输入梯度、各层相位梯度、任务头梯度，并检查 unchanged-source 时 `source_full_response` 与 `bp_current` 相符。初始随机组前向也必须相同。
5. 五组各适配 8 epoch，同一训练样本与增强流、optimizer/LR/scheduler/batch/clip、保存与选择策略。训练使用真实图像增强与当前光学前向，不能继续用暖启固定特征训练 phase。
6. phase LR、head LR、衰减等以实际 resolved config 为准，初值沿用 P12；不得按方法单独调参。只有当前 BP 共享 pilot 的数值问题才可以统一修改五组参数，并建立新 run 身份。
7. 在同一当前模型、同一 batch 上对 exact phase gradient 与候选方向做比较，保存逐层 cosine/范数比、实际相位运动；不能比较两条不同训练轨迹后把差异归因于反馈公式。最后一层在旧 connector FA 中仍为 exact local gradient，在完整响应方法中也被替换，单独报告。
8. 本轮仅报告 validation 初筛结果、实际 epoch/update/sample 数量和不确定性。它不是旧 50+50 正式协议，也不是独立泛化结论。

若 GPU 当前有外部进程即重新判断可用性，不以瞬时低利用率认定空闲。用户提供的 GPU 6/A100 与 GPU 1/4090 是可用候选；实际 UUID、所有者进程、启动 PID、退出状态及最终释放检查写入运行记录。不同型号卡独立跑同预算作业，不混用吞吐作科学性能指标。

## 8. 初筛决策门与失败记录

先判断 BP 相对 `head_only` 是否产生明确相位适配增益；若没有，初筛对反馈质量缺乏诊断性，应增加统一训练预算或换更合适样本，不能声称完整响应无效或有效。

若 BP 有增益而源完整响应大幅退化、连续诊断的主要 stage 出现负余弦，或 phase 漂移增大同时验证损失失控，暂停扩展，保留终态与失败记录。若源完整响应与 BP/源连接器近似而随机完整响应明显较差，只能说明**该源、该短期任务上完整源响应具有初步可用性**。

通过后优先补全量 LSP、更长统一适配、至少三 paired downstream seeds、第二种任务；在源完整响应与 BP 的可接受差异界值预先确定后再做充分重复。若随机完整响应也同样有效，需要明确检验 phase 贡献与初始匹配，不强行归因预训练。

每个结果应包括 initial/shared-start SHA、source/stem SHA、Git commit+dirty flag、配置摘要、实现文件摘要、数据清单、样本与更新预算、逐 epoch 历史、反馈身份、梯度诊断、验证选择规则、异常、checkpoint SHA 与 GPU 退出证据。完整权重、特征缓存和原数据不进 Git；小型报告与源码按所属 FFA 路径串行提交。数据/权重跨端以 manifest+SHA 校验，源码只走 Git。

## 9. 光路中的反馈从哪里来

本轮候选部署是：相机强度进入电脑；电脑根据当前误差、已知加载振幅与冻结的源响应计算更新；新相位再加载到 SLM。反馈因此位于电子学习模块，当前实现没有额外反向光路。

同一个 reciprocal 光路反向通光可实现相关伴随运算，但要生成合适的复误差场、匹配非线性/探测链并与局部状态组合；仅有 reciprocal 传播不能自动得到完整相位梯度。若装置已经便于精确物理伴随且代价低，原位 BP 是有竞争力的方案。本研究价值需要通过实测获取成本、校准频率和鲁棒性比较体现，不能把“反向同光路不可做”当作前提。

本轮模拟中源 `φ₀` 与 `P` 已知；真实装置怎样形成该源响应尚未实现。可选择预训练后一次标定/数字孪生作为上界，但它仍是模型依赖。不能因为后续不用相位探测，就声称初始化也完全不用标定。

## 10. 通过初筛后的发育规则：真正对接教授动机

复制解析源响应是机制上界，尚不等于“基因编码动态形成反馈”。后续要把任务改成：在固定的早期活动窗口，通过少量共享规则参数和活动轨迹形成反馈，之后冻结反馈、更新前向，并跨任务复用规则。

优先考虑结构化形成机制：沿 token/feature 两轴共享低维规则，规则只接受已知加载振幅、已知相位控制扰动与相机强度；在发育期拟合源强度响应的误差作用，而不是逐连接保存与前向完全对称的复矩阵。可以用已知控制扰动的强度差分作为训练信号，其代价必须记录，不能伪称没有探测成本。

这一阶段至少比较：解析 source response 上界、活动形成的结构化 response、无活动随机规则、相同规则在未预训练 body 上形成的 response。每次部署只运行共享规则与限定长度的发育活动，不利用下游标签或 test；外层规则训练与评价源/任务应分开。规则参数量、发育采样数、反馈存储量、后续更新成本与跨源复用能力都成为主要测量。

如果反馈生成规则每个任务都重新用 BP 全量监督、或者仍完整复制百万级前向参数，就不能支持低维遗传编码的强叙述。若发育形成与解析源响应性能相近、参数显著更少、跨新源/新任务有效，且适配期间反馈保持冻结，才真正将“预训练形成可迁移学习能力”与“早期形成反馈、前向承担后续学习”连接起来。

这里的 genetically encoded process 只作为计算研究假设，不写成人脑已被证实的机制。顶刊潜力取决于形成规则的新意、可证伪机制、跨条件证据和真实光路效果，不由一个短期仿真初筛保证。
