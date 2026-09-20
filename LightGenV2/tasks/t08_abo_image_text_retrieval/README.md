# T08 商品检索（图搜文）

## 2026-09-20：15 cm 真正文搜图合同

正式重训配置为
`configs/optical_text_to_image_64_15cm_true.yaml`。三个容易混淆的口径固定如下：

- 英文商品标题是 query，TEST 图片是 gallery document；不是把旧图搜文相似度矩阵转置。
- 正例按完全相同的 `product_id`/SKU 判断，不按 bed、chair 等宽泛类别判断。
- 100 个已登记 SKU；每个 SKU 有 48 张 TRAIN 图片和 24 张 TEST 新视角图片，
  即 4800 TRAIN、2400 TEST。每个文本 query 有 24 个正确图库候选。

全部特征传播与两个光 Router 的传播距离均为 0.15 m。10 cm checkpoint 只迁移
电子残差、适配器和 64D 读出头；Vision/Language 的专家、global 与 Router 相位
全部重新初始化并在 15 cm 下重训。checkpoint 架构字符串也显式含 `15cm_17um`，
因此 10 cm/15 cm 权重不能静默混用。

### 冻结 Qwen 维度口径（不要混表）

64D 是 Qwen3-VL-Embedding 的 Matryoshka 截断口径，用于和 64D 光电输出做同维度、
同检索预算比较；2048D 是完整输出强参照。64D 不是官方默认维度。已复核的结果为：

| 任务/协议 | 预处理 | Qwen 64D Top-1 | Qwen 2048D Top-1 |
|---|---:|---:|---:|
| T07 图搜图，同一已登记 SKU 新视角 | native | 85.125% | 85.625% |
| T07 旧版宽松类别匹配 | native | 94.375% | 95.2083% |
| T08 图搜文 | 动态长宽 | 59.79% | 73.71% |
| T08 图搜文 | 固定 224 光场 | 53.58% | 未单独报告 |
| T08 真正文搜图 | 动态长宽 | 65% | 82% |
| T08 真正文搜图 | 固定 224 白边 | 66% | 80% |

论文主表可统一报告 64D，以避免把输出维度收益算到架构收益里；2048D 同时作为不隐藏的
完整大模型上限放在附表/补充列。图搜文与文搜图不是同一任务，不能因相似度矩阵可转置就
复用提示词编码结果。

## 2026-09-20 纯文搜图（独立协议）

新增 `text_to_image_baseline.py`：冻结 Qwen3-VL-Embedding-2B，100个官方英文标题为纯文本query，
2400张test图为gallery，每query有24张同SKU相关图；无查询图片、无标题附着于gallery图像。
先测动态长宽/固定224白边两种图像预处理，各64D与2048D；主同预算对比使用64D，完整2048D另列，不能隐藏。
报告Hit@1/5/10、真正多正例Recall@K、MRR、全库mAP；100个query意味着Hit@1每次跳1个百分点。
不将旧图搜文指标反转当作新baseline，使用文搜图指令重新编码全部输入。
不微调Qwen、不按分数选择商品或prompt；本次不计时/功耗。

```bash
CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m LightGenV2.tasks.t08_abo_image_text_retrieval.text_to_image_baseline --model /path/to/Qwen3-VL-Embedding-2B --data /DATA/DATA1/guest3/2026OpticsMoE/data/abo_easy100_dataset_20260906 --output LightGenV2/tasks/t08_abo_image_text_retrieval/runs/simulation/text_to_image_frozen_20260920
```

光学待用户确认：图库离线V+L六阶段编码；文本在线只用L router/expert/global三阶段，输出相同64D。
保留原光路、Top2与同尺度融合，文本长度可变须独立设计padding/mask，不能套T07固定77token输入。
训练文本-TRAIN图像的同SKU多正例目标，test只参与既定评估/选模；同标题参与训练只代表已登记目录的新图检索，
不声称未见文本泛化。若需要自然新描述泛化，应预先制作独立描述，或采用SKU互斥划分作为另一协议。
完成结果见对应 run 的 `final_report.json`。

已完成固定冻结权重原图推理（源码`1e218991a`，物理GPU4 RTX4090；PID1202253已退出释放）：

| 图库预处理 | 维度 | Hit@1 | Hit@5 | Hit@10 | MRR | mAP |
|---|---:|---:|---:|---:|---:|---:|
| 动态长宽 |64|65%|75%|83%|0.7051|0.5796|
| 固定224白边 |64|66%|77%|84%|0.7090|0.5691|
| 动态长宽 |2048|82%|90%|96%|0.8560|0.7719|
| 固定224白边 |2048|80%|92%|93%|0.8498|0.7644|

证据：`runs/simulation/text_to_image_frozen_20260920/report.json`与四份逐标题predictions JSON。
单一query文本无图片。报告含模型权重/配置、数据清单及每张图库照片SHA；完整隐藏向量可重算64/2048指标。
64维是取冻结embedding前64维再L2，不学习投影。
建议预注册同预算64D主对照并同时保留2048D强参照，未来若追求完整大模型差距≤5pp，需以82%为参照目标≥77%。

### 64D 光电文搜图正式候选

训练沿用同一份 easy100 合同：100 个已见商品，train 每商品 48 张、test 每商品
24 张；100 条官方英文标题作为文本 query，2,400 张 test 图作为 gallery。该协议衡量
“同一已见商品的新视角检索”，不代表未见 SKU 泛化。checkpoint 仍按周期 test Hit@1
选择，因此属于 test-selected。

| 方法 | α | Hit@1 | Hit@5 | Hit@10 | MRR | mAP | 同权重去光 Hit@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 冻结 Qwen，动态长宽 / 64D |—|65%|75%|83%|0.7051|0.5796|—|
| 冻结 Qwen，固定 224 白边 / 64D |—|66%|77%|84%|0.7090|0.5691|—|
| 光 Router MoE，渐进 α 后低学习率精修 |0.40|**81%**|**92%**|**95%**|**0.8532**|**0.7327**|61%|
| 冻结 Qwen，动态长宽 / 2048D 强参照 |—|82%|90%|96%|0.8560|0.7719|—|

正式候选为精修 epoch 14，checkpoint SHA256
`f81ce7d0ea3da31de09ea01392c47312969b84cc39244c43772d380224faba0cb`。
四个融合门均为 `(1-0.40)E + 0.40O`，且先做 RMS 同尺度对齐；同权重去光后
Hit@1 从 81% 降至 61%，即光支路贡献为 20 个百分点。Vision Router 的四专家
选择占比为 26.05%/24.00%/25.52%/24.43%；Language 为一个共享专家固定占据
Top-2 的一个槽，另一槽在其余三专家间轮换。训练使用
`configs/optical_text_to_image_64_alpha_curriculum.yaml` 后接
`configs/optical_text_to_image_64_alpha04_refine.yaml`，保持 64D、Top-2、原光学几何、
20%–30% 训练期相干未调制强度和鲁棒噪声合同不变。

曾把旧“图搜文”checkpoint 的相似度矩阵转置做过诊断：文搜图 Hit@1 为 89%，
但同权重去光仍为 89%。这不是此前独立训练过的文搜图结果，也不能作为正式光学
结果；它仅说明旧共享嵌入空间的电子残差很强，并用于确定渐进提高 α 的必要性。

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
`reports/QWEN5090D_BASELINE.md`。强均衡光学 MoE 的六次光场临界路径、CCD 后电子
处理、5090D 分量功率，以及与冻结 Qwen 的同协议对照见
`reports/5090d_optical_and_qwen_20260907/README.md`。其中光学 MoE 能耗明确标为“80.388 W
光学设备 + 5090D 分量实测”的组合代理，不冒充实验台整机功率计实测。
