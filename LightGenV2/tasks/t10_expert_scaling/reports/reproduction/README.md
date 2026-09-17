# T10 复现入口

当前只有设计产物，无新训练成绩或已验证模型权重。

- [任务与当前状态](../../README.md)
- [实验协议](../../docs/PROTOCOL.md)
- [数据来源与许可](../../dataset/registry.json)
- [设计参数](../../configs/study.json)
- [自动生成的矩阵与校验摘要](../design/summary.json)

从仓库根目录执行：

```bash
python LightGenV2/tasks/t10_expert_scaling/plan.py
```

只需Python标准库；输出为设计表，不读取数据、不加载checkpoint、不使用GPU。
已核查数据集公开许可；尚未下载/校验本任务原始数据及split，registry中的SHA256保持null。
不能把几何自检通过写成新模型复现成功。未来结果必须在此登记run ID、配置、选模与证据链接。

此前病理代码结构的核查依据（不是本任务实现）：

- `LightGenV2/demo_check/adrenal_softsign_code_export_20260915_145336/code/optical_reference/model.py`：
  `between_expert_stages`与`global_fcs[-1]`明确包含中间及末端global phase。
- 同目录`models.py::RelayFanoutMoE.global_fanout_convolution`：理想图像复制及振幅L2归一化。
- 同目录`optical_reference/prompt.py::_topk_routing`：旧Top-k梯度及概率/功率口径。
- `LightGenV2/tasks/t08_abo_image_text_retrieval/reports/optical_router_moe_20260907/main/config.yaml`：
  正式任务224²专家、pitch254的尺寸来源。
