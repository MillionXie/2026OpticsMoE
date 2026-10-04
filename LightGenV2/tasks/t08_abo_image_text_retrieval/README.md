# T08 商品检索（图搜文）

## 2026-10-03 整理：区分两个检索方向

本页现有训练命令及 `optical_moe.py` 是**图搜文**入口。ABO 图搜图最终版另属
T07；**文搜图**曾在同名 T08 的另一个运行工作树实现，不能直接将其 PT 换入
本页命令。三方向总索引见 [ABO 版本图](../../../maintenance/storage/ABO_VERSION_MAP_20261003.md)。

| 保留版本 | 仿真 Top-1 | 未微调实拍 | 用户采用的末端微调实拍 |
| --- | ---: | ---: | ---: |
| 图搜文，性能优先 | R@1 .79875 | 本次未核验，不填 | 本次未核验，不填 |
| 图搜文，强均衡对照 | R@1 .7983333333 | 本次未核验，不填 | 本次未核验，不填 |
| 文搜图，10cm / alpha约.40 | Hit@1 .86；去光 .73 | Hit@1 .79 | Hit@1 .85，10epoch、VAL选第9轮 |

本轮逐文件核验了服务器两份图搜文 best/last、文搜图主体 best/last、easy100
四份 CSV，以及本机文搜图主体与10轮读出 best/last。身份、SHA、来源和待办见
[结构化核验记录](reports/reproduction/FINAL_IDENTITIES_20261003.json)。这是身份核验，
不是重新训练或重评；服务器原图搜文6项CPU合同测试通过，不代表反向 PT 完整复现。

文搜图实际入口在服务器既有 `.worktrees/t08_text_to_image_20260920`，当前
源码 commit `d0662a7d240340817948a3496c2cd43f4240e76d` 的 `optical_moe.py`
与服务器工作文件、本机交接源码 SHA 相同；历史 run manifest 仍如实保留
`9485674...` 加 dirty 的记录。两份共享几何/架构身份依赖与 main 不同，未覆盖。
文搜图在 main 的独立固定评估入口见下节；原运行工作树和 Windows 工程保护。

## 2026-10-04 文搜图固定评估入口

`text_to_image.py` 加载封存的 10cm 主体 `cc977b83...`，调用单独保留的
`reverse_runtime.py`；不替换本页图搜文 `optical_moe.py`。21 层旧配置继承已
机械展开为 `configs/text_to_image_10cm_adopted_eval.yaml`，保留原结构和光学参数。
入口只允许固定评估，不训练、不重置相位或融合门，拒绝覆盖已有输出目录；
模型、数据、教师缓存和权重须显式指定，不自动下载模型。先只读检查：

```bash
python -m LightGenV2.tasks.t08_abo_image_text_retrieval.text_to_image \
  --model /absolute/path/to/local/Qwen3-VL-Embedding-2B \
  --data-root /absolute/path/to/abo_easy100_dataset \
  --checkpoint /absolute/path/to/adopted/best_checkpoint.pt \
  --teacher-cache /absolute/path/to/teacher_cache.pt \
  --run-dir /absolute/path/to/new_fixed_eval_run --inspect
```

去掉 `--inspect` 才执行固定主体评估；默认 CPU，可显式指定空闲设备。
固定入口现在先核验既有教师缓存的方向、提示词、四份数据CSV、样本顺序、
模型路径和64维张量；缓存缺失或不符时，在大模型加载与输出创建前停止，
不会静默重新生成缓存。旧缓存的逻辑模型名与显式本地路径不自动视为同一身份，
不能改缓存身份字段绕过检查。当前两处历史任务run目录均未找到原教师缓存；
完整复评仍需定位或另行明确重建正确方向的缓存，不声称依赖已经全部齐全。
原图7,200张、划分及三组正式best/last和报告的内容身份见
`maintenance/storage/T08_CONTENT_ASSETS_20261004.json`；只读核验命令为
`python maintenance/storage/check_t08_assets.py --repo-root /absolute/path/to/2026OpticsMoE`。
这不是 Windows 光路采集或 .85 读出适配版部署命令。迁移验证已完成真实主体 PT
严格加载和有界合成 token 的旧/新后端逐值等价，尚未新重评全数据或迁移 Windows
流程，不将这些检查冒称新的实拍或精度复现。
证据见仓库 `maintenance/storage/T08_BACKEND_COMPATIBILITY_20261004.json` 和
`maintenance/storage/T08_EVAL_PROFILE_IDENTITY_20261004.json`。

当时服务器实际运行的导出、真实CCD回放、末端读出微调七项工具也已归入
[文搜图实拍工具](physical10cm/README.md)，不再依赖散落的服务器临时源码目录。
仅重新绑定包内导入和正确反向实现，历史计算函数未改；有界CPU合同检查通过。
Windows采用版的DVP逐层采集入口也已归入同包 `capture_full_stage`：显式指定
既有项目资产目录和外部SDK/LUT配置，不再从几个外部工程寻找设备或模型源码。
18项纯CPU合同测试通过，采样、六帧丢弃、曝光/增益、ROI及阶段方向保持原合同。
统一入口另支持原TEST首层的序号／样本ID清单；只将元数据映射为原 `image_0000`
至 `image_2399` 文件键，拒绝重复、错序号范围、混格式和错误文件名，不改原清单或幅度。
这是源码归并，不是新的硬件回归；实验室现用工程未替换，真实SDK/LUT与三端运行
目录切换仍待核验，不能因此声称整套硬件已可直接接管。详情及新命令见实拍工具页。

原Windows资产也已逐文件读取核验：TEST六层14,700张CCD，TRAIN六层5,100张CCD，
对应幅度图各自齐全；39,631份文件内容身份、全部采集日志中的相位／幅度／曝光／ROI
和六帧合同相符。语言侧每层另含100标题，因此不是图搜图的14,400帧合同。
证据见 `maintenance/storage/T08_PHYSICAL_CONTENT_IDENTITY_20261004.json`；PNG检查为
头部形状／8位灰度和整文件SHA，不是重新解码重评或新光路稳定性验收。旧低亮度统计
保留，不套用其他任务的新暗帧阈值否定或重新采这些历史结果。

用户采用的是主体 `cc977b83...` + 10轮读出 `89e25360...` 的 .85 版本。
更长微调对照、旧相位/CCD/预测和所有测速文件原位保留；另有 .88 仿真的计时
PT `8a96132d...`，不是这份实拍主体，不拼接其精度和时间。本项目历史 TEST
曾用于开发，不能把10轮训练内未用TEST选模说成全项目独立泛化验证。

以下原始图搜文结果、baseline与测量协议保持原含义。

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
