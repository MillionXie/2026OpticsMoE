# Qwen3-VL-2B CC3M general distillation: PCA224 + Optical MoE16

## Automatic CC3M preparation

The formal configuration uses the public WebDataset snapshot
`chaocq/cc3m-wds` pinned at revision
`28cde01364d7e3b180681f8c448935edf47e2fd5`. It contains 576 training
shards and 2,905,954 successfully recovered image-caption pairs from the
original CC3M training URLs. The snapshot is public and does not require
gated-dataset approval.

`--phase prepare_data` downloads shards with resumable Hugging Face transfers,
extracts only the paired images and captions, deletes each source TAR after a
successful extraction by default, and creates `data/cc3m/cc3m.jsonl`. Per-shard
markers and manifests live under `data/cc3m/.cc3m_prepare`, so rerunning after
a network interruption continues from completed shards. The generated
`cc3m.jsonl.metadata.json` records the source repository, pinned revision,
shard/sample counts, byte size, and manifest SHA256.

The server-oriented formal config uses `https://hf-mirror.com`; the endpoint,
download worker count, archive retention, and optional smoke shard limit are
all configurable under `dataset.prepare`. A manually created JSONL remains
supported when `auto_if_missing` is disabled.

这是一个独立的、无任务标签的通用蒸馏实验。冻结的
`Qwen/Qwen3-VL-2B-Instruct` 读取图像和 caption，student 用彼此独立的
Vision Optical MoE16 与 Language Optical MoE16 拟合教师的多阶段 hidden
features。

本实验不包含分类/回归 head，不读取任务标签，不缓存 teacher logits，也不做
文本生成。

## 数据

数据由 JSONL manifest 提供，每行必须是：

```json
{"sample_id": "unique_id", "image_path": "/path/to/image.jpg", "caption": "caption text"}
```

相对 `image_path` 按 manifest 所在目录解析。默认 formal manifest 是
`data/cc3m/cc3m.jsonl`。代码不会静默下载或替换 CC3M；路径或图片缺失时会列出
实际尝试的路径。

若没有单独的 validation manifest，代码用 seed 42 对 manifest 行做确定性
train/validation 划分，并把结果持久化到 `data_split.json`。Smoke 配置只读取
前 100 个样本，但执行与正式实验完全相同的 PCA、teacher cache 和训练接口。

## 固定 PCA

每个 stack 只拟合一套共享坐标系：

- `PCA_vision` 同时拟合 vision stack input、3 个原生 DeepStack tap 和 final tap；
- `PCA_language` 同时拟合 language stack input、decoder layer 0/1/2/final taps。

encode 与 decode 分别为：

```text
z = (LayerNorm(hidden) - mean) @ components
hidden_reconstructed = z @ components.T + mean
```

`mean` 和 `components` 是 buffer，不参与训练。Vision 与 language 各自的四个
stage 都共享对应 stack 的这一套 PCA，不允许 stage-specific PCA。新 student
中没有可训练的 `D→224` 或 `224→D` Linear adapter。

## signed readout 与 reload amplitude

每个 optical stage 的 detector 路径为：

```text
complex field
→ square-law intensity
→ crop physical CCD ROI
→ adaptive pool to 224×224
→ non-affine LayerNorm
→ signed_readout
```

`signed_readout` 可正可负，直接用于 stage distillation 和固定 PCA decode。
只有准备下一 OEO stage 时才计算 `reload_amplitude = ReLU(signed_readout)`。
因此 PCA loss 不会误用 ReLU 后的非负振幅。

## Qwen 原生部件

Qwen tokenizer、token embedding、vision patch embedding、四个 frozen vision
merger（3 个 DeepStack merger + final merger）、原生 DeepStack 注入和 final
RMSNorm 全部保留且冻结。Language teacher taps 使用 layer
`[0, 1, 2, final]`，以匹配 Qwen 在前三个 decoder layer 后执行 DeepStack
注入的真实时序。

任何 visual token count 或完整 language sequence 超过 224 都会直接报错。代码
不会 crop、truncate、pool hidden 或改变 token-row mapping。

## 流程

`all` 严格依次执行：

1. `fit_pca`
2. `pca_oracle_check`
3. `precompute_teacher`
4. `train_vision`
5. `train_language`
6. `train_joint`

Oracle 会报告 hidden cosine、normalized MSE、relative reconstruction error，
以及 frozen vision merger / final RMSNorm 后的 cosine 与 MSE。即使结果较差，
代码也只告警，不会偷偷添加 trainable adapter。

缓存位于本实验的 `cache/`，runs 位于本实验的 `runs/`，两者均被 `.gitignore`
排除。Projected teacher cache 对 manifest、prompt、processor pixel budget、
PCA digest 和 tap indexes 做严格身份校验。
