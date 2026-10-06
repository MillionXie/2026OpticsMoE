# T07 独立版验收：2026-09-10

这次是整理与等效复现，不是重新优化出更高的候选。交付保持原 epoch10 best；没有用微调检查权重替代它。

## 性能

同一 RTX 4090、1440 个训练视角建立 120 商品中心、480 个测试 query，按同类别相关性检索：

| 版本 | Hit@1 | mAP@10 | 同权重去光 Hit@1 |
| --- | ---: | ---: | ---: |
| 原工程 best 复评 | 70.2083% | 0.68003894 | 67.2917% |
| 独立版隔离目录复评 | 70.2083% | 0.68001116 | 67.2917% |

独立版光学移除下降 2.9167 个百分点。测试特征与旧版平均余弦 0.99999756，非逐位相同；
显式前端及混合精度运算有微小差别，不能说所有排序完全一致。
旧 A100 原记录为 70.0000%，跨 GPU 差一个 query；不把这一差异算作模型提升。
这是 test-selected 权重，不是独立无偏测试；也不是同商品实例检索、实测 CCD 性能或抗噪保证。

## 独立性检查

- 验收运行时源码 commit：`172c8092f6cd9e5131fe2eee10fd3c307dc1c653`；之后只补交付说明/打包清单。
- 源码白名单复制到新的目录，带完整数据、assets；Python `-I`，清空 PYTHONPATH。
- import guard 禁止导入 `experiments` 和 `LightGenV2`；仍完成全部正常/去光评估。
- 运行前逐文件 SHA256 校验；最终 ZIP 的 `MANIFEST.json` 记录交付源码 commit 和文件摘要。
- 模型审计：0 Transformer、0 attention；只保留 Qwen patch/position/merger、固定 prompt 的词嵌入行。
- 29,152,256 个冻结参数、2,782,485 个可训练参数；模型/processor/训练目标合计 85,580,305 字节。
- 只使用 AutoProcessor 处理文本/图片，不创建 AutoModel，也不加载完整 2B 权重。原 tokenizer 保留。
- 固定 prompt 和图像布局有严格检查；修改 prompt 需重新导出对应词嵌入，不能当通用 Qwen 服务使用。

## 可训练性与资源

16 项本地测试通过。单卡做了 1 epoch × 2 steps 的端到端短微调，包含训练、EMA、best/last 保存与重载、全量评估。
10 个专家/全局 raw phase 参数 RMS 更新约 0.000994～0.001210，确认相位能训练。
短检查 Hit@1=69.7917%，只用于接口验收，不是性能优化结果；交付不包含其权重。

单张 4090 串行完成；训练 batch=20，PyTorch peak allocated 2039 MiB、reserved 2138 MiB；
评估 batch=4，peak allocated 187 MiB、reserved 230 MiB。这些数值不含完整 CUDA 上下文/他人显存。
最终验收 PID=3511037 已退出，nvidia-smi 确认不再占卡；没有停止他人进程。

## 文件与用途

源码位置：`LightGenV2/tasks/t07_abo_image_retrieval/`。交付包：
`releases/t07_abo_standalone_70_20260910.zip`（接收人解压后先读 COMMAND.md）。
服务器证据位于本任务 `runs/simulation/standalone_isolated_verify_20260910/`；
原始资产在 `runs/simulation/standalone_assets_20260910/`；短微调检查在 `runs/smoke/standalone_finetune_20260910/`。
接收人不需要这些服务器路径，ZIP 已含 best、processor、训练目标、2400 张图、数据清单及 reference 报告/相位预览。

不包含旧工程、优化器历史、候选 runs、全 Qwen 或其他任务。未删除共享旧后端和历史证据，避免破坏其他任务。
本包是**独立仿真/微调工程，不含 SLM/CCD 厂商 SDK 和已验收自动采集流程**。
CCD 实测注入接口不等于完整硬件闭环。相位预览是可视化，不能直接当作 8 μm SLM 播放 BMP。
