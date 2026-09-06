# T08 商品检索（图搜文）

本任务是 ABO easy100 的单张商品图像到官方英文标题检索，不是图搜图：100 个
商品、4,800 张 train、2,400 张 test、100 个唯一标题候选。性能均在完整 test
上计算；训练期间每 5 个 epoch 看一次 test，并按 EMA test R@1 选择 checkpoint。
因此该结果适用于当前工程的选模口径，但属于 test-selected，不应伪装成 sealed test。

## 2026-09-07 光 Router MoE 结果

| 方法 | 输入 / 维度 | R@1 | R@5 | R@10 | MRR |
| --- | --- | ---: | ---: | ---: | ---: |
| 冻结 Qwen，固定光场输入 | 224×224 / 64D | 0.5358 | 0.7958 | 0.8458 | 0.6559 |
| 冻结 Qwen，动态长宽 | 动态 / 64D | 0.5979 | 0.8517 | 0.9063 | 0.7151 |
| 冻结 Qwen，动态长宽 | 动态 / 2048D | 0.7371 | 0.9338 | 0.9604 | 0.8230 |
| 光 Router MoE，性能优先 | 224×224 / 64D | **0.7988** | 0.9479 | 0.9825 | 0.8617 |
| 光 Router MoE，强均衡 | 224×224 / 64D | 0.7983 | **0.9538** | **0.9858** | **0.8676** |

性能优先版只比强均衡版多命中 1/2,400 张 Top-1；强均衡版的其余排名指标和
平均名次更好，所以硬件部署优先推荐强均衡版，同时保留性能优先版作为 R@1 报告值。

两种 64D Qwen 基线的差别不是模型权重：光学网络要求固定正方形光场，故先做
224×224 居中裁剪；原冻结基线让 processor 保留动态长宽。二者必须分开列出。

## 光电推理合同

- Qwen3-VL-Embedding-2B 原始权重全冻结；训练光 Router、相位、紧凑电子支路、
  同尺度融合门和 64D 读出头。
- Vision 与 Language 各两次 O/E/O 特征传播；每次由物理能量 Router 选 Top-2，
  专家相位为 224×224、2×2 排布在 478×478 有效场中。
- 逻辑采样 17 µm、传播距离 10 cm；推理图不改变 Caltech T01 的硬件尺寸和 ROI。
- 融合为 RMS 对齐后的 `(1-alpha)E + alpha O`，而不是让电子数值范围淹没光支路。
- 训练加入 20%–30% 相干未调制强度、截断偏置高斯 CCD 噪声、±16 px 位移、
  k 空间角度扰动和相位 DC 约束；确定性仿真评估不随机加噪。
- 图像查询经过 Vision+Language 光支路；100 个文本标题仅经过 Language 支路，
  可预计算后作为固定候选库。检索使用余弦相似度完整排序。

最佳 epoch 的 Vision 专家几乎均衡。Language 呈现“共享专家 0 + 三个轮换专门
专家”的 Top-2 结构；强均衡版图像侧选择占比约 50.0%/19.8%/15.4%/14.8%，
不是只剩两个专家的坍缩。

## 实验室服务器复现

性能优先：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m LightGenV2.tasks.t08_abo_image_text_retrieval.optical_moe \
  --config LightGenV2/tasks/t08_abo_image_text_retrieval/configs/optical_router_moe_dc20.yaml \
  --device cuda:0 --seed 42 --epochs 40
```

强均衡：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m LightGenV2.tasks.t08_abo_image_text_retrieval.optical_moe \
  --config LightGenV2/tasks/t08_abo_image_text_retrieval/configs/optical_router_moe_dc20_kd1_balance1.yaml \
  --device cuda:0 --seed 42 --epochs 40
```

每个 run 只保存 `best_checkpoint.pt` 与 `last_checkpoint.pt`，另含完整预测、
训练曲线、相位总览、融合诊断与数据 SHA256。实验室服务器结果位于：

`/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t08_abo_image_text_retrieval/runs/simulation/`

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

冻结 Qwen 的 5090D 正式速度/功耗结果与测量说明见
`reports/QWEN5090D_BASELINE.md`；光学 MoE 的仿真结果不冒充硬件功耗测量。
