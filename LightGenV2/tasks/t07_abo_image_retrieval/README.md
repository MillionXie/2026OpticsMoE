# T07 商品检索（图搜图）

## 正式数据合同

本任务使用 `abo_similarity10_data_only.zip`：10 个类别、200 个商品、每商品 12 个视角，
商品身份在 train/validation/test 间严格不重叠。

- train/gallery：120 个商品、1440 张图片；每个商品的 12 个 Qwen 图像特征取均值并归一化，
  得到一个固定 gallery centroid。
- validation：40 个商品、480 张图片；当前 frozen-Qwen baseline 不用它选模型。
- test/query：40 个未见商品、480 张图片。
- 对每张 query，gallery 中同类别的 12 个商品均为相关项；它不是精确商品 ID 检索。
- 主指标：unrestricted R@1（这里等于 Hit@1 和 P@1）；同时报告 category-route
  accuracy、P@5/P@10、positive Recall@K、Hit/R@K、mAP@10 和 NDCG@10。

## 冻结 Qwen baseline

`baseline_a100.py` 使用完整 `Qwen3-VL-Embedding-2B`，2048D、224×224，所有参数冻结，
没有 LoRA、微调或训练读出头。性能在完整 480 张 test 上评估。在线核心计时从第一个原生
Vision Transformer block 开始，到 2048D 归一化、120 个 gallery cosine、10 个 category
prototype cosine 和排序结束；图片读取、解码、processor、patch embedding、gallery 构建和
模型加载不计入核心时延。

正式运行必须指定物理 A100：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m LightGenV2.tasks.t07_abo_image_retrieval.baseline_a100 \
  --model /path/to/Qwen3-VL-Embedding-2B \
  --data-root /path/to/abo_similarity10_data \
  --dataset-archive /path/to/abo_similarity10_data_only.zip \
  --run-dir LightGenV2/tasks/t07_abo_image_retrieval/runs/simulation/qwen_frozen_a100_20260907 \
  --timing-samples 200 --warmup-forwards 50 \
  --expected-gpu 'NVIDIA A100-PCIE-40GB'
```

运行目录保存实际命令、Git commit、环境、数据哈希、480 条预测、200 条计时、原始功率采样、
gallery index、JSON 总报告和概览图。原始数据、模型与生成的 run 不提交 Git。
