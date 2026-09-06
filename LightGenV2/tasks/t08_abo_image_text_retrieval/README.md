# T08 商品检索（图搜文）

当前任务是 ABO easy100 的单张商品图像到官方英文标题检索，不是图搜图：

- 100 个商品、4,800 张 train、2,400 张 test；本冻结基线完全不使用 train。
- 候选库为 100 个商品标题，每个标题只编码一次。
- 查询图像逐张推理，固定 `batch=1`、224×224 输入、完整 2048 维归一化 embedding。
- 模型为 `Qwen3-VL-Embedding-2B`，所有参数冻结，不微调，也不训练额外读出头。
- 性能在全部 2,400 张 test 上计算 R@1、R@5、R@10、MRR 和 rank。

## 5090D 正式测量口径

性能与计时分开执行。计时从第一个原生 Vision Transformer block 的输入开始，到图像
embedding 归一化、与 100 个预计算标题 embedding 求相似度并完成完整排序为止。文件读取、
图像解码、processor/tokenizer、patch embedding、标题库预计算和模型加载均不计入。

正式速度/功耗采用固定协议：50 次不计时预热，随后测量类别均衡的 200 张 test（每个商品
2 张）。功率使用 `nvidia-smi power.draw` 以 10 ms 请求间隔保存全部原始采样，只把上述模型
在线窗口标为 active。共享 5090D 有其他计算进程时不得测速度或功耗。

```bash
python -m LightGenV2.tasks.t08_abo_image_text_retrieval.baseline_5090d \
  --model /root/autodl-tmp/models/Qwen3-VL-Embedding-2B \
  --data-root /root/autodl-tmp/datasets/abo_easy100_dataset_20260906 \
  --run-dir LightGenV2/tasks/t08_abo_image_text_retrieval/runs/simulation/qwen_frozen_5090d_easy100_controlled \
  --warmup-forwards 50 --timing-samples 200
```

输出包括总报告、逐图 Top-10、逐商品指标、200 张计时记录、原始功率采样、图像/标题
embedding、汇总图和全部文件 SHA256。

最新正式结果与测量说明见 `reports/QWEN5090D_BASELINE.md`。
