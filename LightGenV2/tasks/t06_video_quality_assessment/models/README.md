# T06 model ownership

当前计算图的唯一实现位于：

```text
experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54/
```

LightGenV2 已提供唯一运行入口并接管新的 runs、报告和 releases。此阶段不复制
`modeling.py`，因为本地与源服务器源码 SHA256 已核对一致，复制会造成两份模型悄悄
分叉。正式 profile 已锁定核心源码和 config 的逐文件 SHA256，若后端被修改，入口会
停止而不是悄悄产生不可比较结果。等第二个 LightGenV2 任务迁移后，再把真正复用的
传播器提升到 `common/`。

当前 36 帧正式结构：冻结 Qwen 图像/文本前端缓存，36 个 7×7 视觉 token 网格，
6×6 lane 并行，四个 37×37 光专家，光学 Top-2 router；视频级串行部分保留四个
109×109 专家。学生网络不含 Attention 或 Transformer block，最后由目标专属电子
读出头输出单个连续 Temporal MOS。
