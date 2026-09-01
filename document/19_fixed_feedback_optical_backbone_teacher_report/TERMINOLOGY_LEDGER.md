# 术语账本

本文档用于保证代码、汇报和论文中的同一对象始终使用同一名称。

| 推荐中文 | 推荐英文 | 精确定义 | 避免使用 |
|---|---|---|---|
| 混合光电计算 | hybrid optoelectronic computing | 光学主体与轻量数字电子模块协同 | 混合精度（除非真的比较位宽） |
| 光电异构协同 | optoelectronic co-processing | 不同物理计算介质分工 | 纯光网络 |
| 源算子固定反馈 | source-fixed feedback / FA-source | 冻结 source checkpoint 的层间 optical error connector | 全网络无 BP |
| 当前算子精确反传 | current-operator exact BP | 层间 connector 使用当前光学算子 | normal BP（正文中定义后可简写 BP） |
| 随机固定反馈 | fixed random feedback / FA-random | 谱和范数匹配的随机 optical connector | random forward |
| 当前物理前向 | current physical forward | 所有训练方式都用当前 phase 执行 forward | pretrained forward |
| 本层精确相位导数 | exact local phase derivative | 本 stage phase 梯度的局部导数取自当前 forward | 整条光路 exact BP |
| 语义轴光学 mixing | semantics-aligned axial optical mixing | token 和 feature 映射到正交光学坐标并交替传播 | 发明 token mixing |
| 潜在光学 bank | latent optical bank | Qwen stem 后的三个潜在表示路径 | RGB 三通道光路 |
| 冻结 Patch/Position Stem | frozen Qwen patch/position stem | patch convolution + positional embedding，无 Transformer/LM | Qwen vision encoder、VLM backbone |
| No-ImageNet body initialization | No-ImageNet body initialization | 保留冻结预训练 stem，随机 body，其三个 run 共用一个 body init | full scratch、三个独立 backbone seeds |
| 光学可训练 body 参数占比 | optical share of trainable body parameters | 排除冻结 stem 与临时任务头后的参数计数比例 | 光功率占比、MAC 占比、能耗占比 |
| 数值融合门 | numerical fusion gate | 光/电张量合并的系数 | 物理光功率比例 |
| 函数保持式扩深 | function-preserving growth | `x+alpha(Stage(x)-x)`，alpha=0 保留旧函数 | 已证明深层性能 |
| 源初始化反馈 | source-init feedback | growth 新层刚插入时捕获的初始算子 | pretrained deep feedback |
| 预训练深层反馈 | pretrained deep feedback | 完整深层 source 预训练后捕获并用于下游的算子 | 当前 P13 growth feedback |
| LSP PCK@0.2 torso | PCK@0.2 normalized by torso | 当前 LSP 主指标 | PCK@0.1 |

写作强度约束：

- `matches/reaches the performance level of BP`：当前可作描述性表述；
- `equivalent/non-inferior to BP`：需要预定义等效界值与更充分重复；
- `scales to 100 stages`：当前只能限定为计算图、参数量和全深度梯度；
- `effective 100-stage backbone`：完成 ImageNet、同预算 control、层贡献与下游迁移后才可使用；
- `large-model optical training`：当前不可使用，因为只有冻结 Qwen Patch/Position Stem。
