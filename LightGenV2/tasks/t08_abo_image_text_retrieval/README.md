# T08 商品检索（图搜文）

- 当前数据集：ABO，后续可替换。
- 正式子集：`abo_easy100_dataset_20260906`，100 个商品、4,800 张 train、2,400 张
  test；本次冻结 Qwen baseline 不使用 train。
- 任务方向：每张 test 商品图像检索 100 个商品的官方英文标题，即 image-to-title，
  不是图搜图。

## 冻结大模型 baseline

- 模型：`Qwen3-VL-Embedding-2B`，完整 2048 维输出；全部参数冻结，不微调，也不训练
  读出头。
- batch 固定为 1；图像和标题均逐条进入模型。
- 每个标题只编码一次，形成固定 100-title candidate bank。
- 指标：R@1、R@5、R@10、MRR、mean rank、median rank。
- 在线延迟：从第一个原生 Vision Transformer block 输入开始，到图像 embedding 归一化、
  与 100 个标题计算相似度并完成排序为止。图像读取、processor、patch embedding、模型
  加载和标题库预计算均不计入。
- 功耗：标题库计算后先等待 GPU 冷却，再与正式 query 推理同时以 100 Hz 请求频率记录
  RTX 5090 D `power.draw`；保存 active mean、
  peak、idle-subtracted J/image 和全部原始采样。
- 正式测量不做显式 warm-up，第一张 test 图片也进入统计。

```bash
python -m LightGenV2.tasks.t08_abo_image_text_retrieval.baseline_5090d \
  --model /root/autodl-tmp/models/Qwen3-VL-Embedding-2B \
  --data-root /root/autodl-tmp/datasets/abo_easy100_dataset_20260906 \
  --run-dir LightGenV2/tasks/t08_abo_image_text_retrieval/runs/simulation/qwen_frozen_5090d_easy100
```

输出包括报告 JSON、逐图片时间、逐图片 Top-10、逐商品指标、原始功率采样、图像/标题
embedding 和汇总图。正式运行前必须确认 5090 D 没有其他进程；共享占卡时不得测速度
或功耗。
