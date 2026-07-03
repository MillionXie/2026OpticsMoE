# Qwen3-VL-8B 图像分类基准

这是一个独立于 `qwen_vl_cifar10` 的工程。模型固定为
`Qwen/Qwen3-VL-8B-Instruct`，冻结完整主干，只训练末尾的两层 MLP 分类头：

```text
Image -> Qwen3-VL-8B Vision Transformer -> visual-token mean pooling
      -> Linear -> GELU -> Dropout -> Linear -> class logits
```

## 支持的数据集

- `cifar10`
- `cifar100`
- `stl10`
- `svhn`
- `fashionmnist`
- `imagefolder`：目录结构为 `<data_root>/train/<class>/...` 和
  `<data_root>/test/<class>/...`

增加其他 torchvision 分类数据集时，只需在 `datasets.py` 中注册训练集、测试集和类别名。

## 安装

```bash
python -m pip install -r experiments/qwen_vl_8B/requirements.txt
```

建议将 Hugging Face 缓存放在容量足够的数据盘：

```bash
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_CACHE=/root/autodl-tmp/hf_cache/hub
```

### 无卡模式只下载模型

无卡实例不要执行完整实验。使用专门的下载阶段，它不会初始化数据集、构造模型或调用 CUDA，
并默认禁用可能返回 CAS 401 的 Xet 下载客户端：

```bash
python -m experiments.qwen_vl_8B \
  --config experiments/qwen_vl_8B/configs/cifar100.json \
  --phase download \
  --cache-dir /root/autodl-tmp/hf_cache \
  --download-workers 2
```

如果仍提示访问限制，运行 `hf auth login` 并粘贴 Hugging Face read token。下载完成后，
有卡模式使用 `--local-files-only` 可禁止程序再次访问网络。若显式传入了 `--cache-dir`，
下载和推理必须使用同一个值；程序也会通过运行目录中的 `download.json` 自动恢复该路径。

## 运行

先用小规模配置验证完整链路：

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen_vl_8B \
  --config experiments/qwen_vl_8B/configs/cifar100_smoke.json \
  --device cuda --local-files-only \
  --cache-dir /root/autodl-tmp/hf_cache
```

完整 CIFAR-100 实验：

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen_vl_8B \
  --config experiments/qwen_vl_8B/configs/cifar100.json
```

可以分阶段执行，便于断点复用特征：

```bash
python -m experiments.qwen_vl_8B --config <config.json> --phase extract
python -m experiments.qwen_vl_8B --config <config.json> --phase train
python -m experiments.qwen_vl_8B --config <config.json> --phase inference
python -m experiments.qwen_vl_8B --config <config.json> --phase visualize
```

本地模型目录可写入配置的 `model_id`。目录必须包含真实的 safetensors 权重，不能只有
Git LFS 指针。`cache_dir`、`data_root` 和 `output_dir` 也可通过命令行覆盖。

## 科研计时定义

推理批次先预热，预热结果不进入统计。每个 GPU 阶段前后执行 CUDA 同步，使用
`time.perf_counter()` 记录墙钟时间：

- `data_loading_sec`：等待 DataLoader 返回图像和标签。
- `image_preprocess_sec`：PIL 图像经过 Qwen image processor；不调用文本 tokenizer。
- `host_to_device_sec`：视觉输入从 CPU 传输到 GPU。
- `vision_forward_sec`：`get_image_features()` 的视觉主干前向。
- `pooling_sec`：视觉 token 均值池化。
- `mlp_forward_sec`：分类头前向。
- `model_inference_sec`：视觉主干、池化和 MLP 的合计，不包含图像预处理。
- `postprocess_sec`：argmax、top-5 和结果转回 CPU。
- `pipeline_sec`：图像预处理开始到得到 CPU 分类结果，不包含 DataLoader 等待。
- `end_to_end_sec`：从请求下一批数据到得到 CPU 分类结果。

报告包含 mean、standard deviation、median、P90、P95、P99、单图延迟及吞吐率。

## 输出结构

```text
runs/<run_name>/
├── config_resolved.json
├── environment.json
├── dataset.json
├── model.json
├── summary.json
├── checkpoints/best_mlp.pt
├── features/train.pt
├── features/test.pt
├── metrics/
│   ├── feature_extraction_*_batches.csv
│   ├── training_history.csv
│   ├── inference_batches.csv
│   ├── inference.json
│   ├── predictions.csv
│   └── confusion_matrix.csv
└── figures/
    ├── training_curves.{png,pdf}
    ├── classification_metrics.{png,pdf}
    ├── latency_breakdown.{png,pdf}
    ├── confusion_matrix.{png,pdf}
    └── per_class_accuracy.{png,pdf}
```

`model.json` 保存总参数量、视觉/语言主干参数量、MLP 参数量和关键 hidden size；
`inference.json` 保存精度、特征形状、逐阶段计时汇总和 CUDA 峰值显存。
