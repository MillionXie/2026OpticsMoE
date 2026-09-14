# ABO 冻结 Qwen baseline：给老师、复现同学及其 AI

本目录可以脱离整个 LightGenV2 独立运行。只有一个评估脚本 `baseline.py`，没有光学网络、
学生权重、其他任务依赖或训练入口。ZIP 中的 `PACKAGE_MANIFEST.json` 固定源码 commit 及每个文件 SHA256。
这不是硬件部署包。模型和图像需另行提供；已附原始特征缓存，可先用 CPU 检查计分。

## 1. 为什么 94.375% 变成 85.125%？不是“Qwen 被我们微调过拟合”

两次均使用冻结的 Qwen3-VL-Embedding-2B，没有反向传播、优化器、LoRA、训练读出头、PCA拟合或reranker。
**旧版与新版不只是改变 train/test 划分，而是改变了正确答案定义和图库构建。**

| 项目 | 旧版：未见商品的类别相似检索 | 新版：已登记商品的实例检索 |
|---|---|---|
| 任务名 | `legacy_category` | `enrolled_sku` |
| 商品 | train120 / val40 / test40，身份互斥 | 同200个SKU都在图库中 |
| 图像 | 每商品12张；1440/480/480 | 每商品按固定哈希分8图库+4查询 |
| 查询 | 40个test商品的480张图 | 200个商品的800张图 |
| 候选库 | 120个train商品中心，每个中心平均12视角 | 1600张独立图库图，不取商品平均 |
| 正确答案 | 同类别的任意商品均算对，每查询12个正例 | 必须同一个SKU，每查询8个正例 |
| 举例 | 查椅子A，返回椅子B也对 | 查椅子A，返回椅子B就是错 |
| Qwen native64 Hit@1 | 453/480 = **94.375%** | 681/800 = **85.125%** |

“已见”只表示光电学生的训练集包含该SKU的其他照片；冻结Qwen根本不在这个训练集上学习。
给Qwen换图库并不等于让它学会这200件商品；官方预训练中是否出现某件商品也不能由这里证明。
旧查询商品在旧图库中没有同SKU正例，因此不能把旧任务直接按同SKU计分并称为公平对照。

关键诊断：在新版原有64维缓存中，**不改变特征、图库、排名或Top1预测**，
仅将正确性从“同SKU”放宽为“同类别”，得到790/800=**98.75%**。
119个实例错误中，109个是同类别其他商品，只有10个跨类别错误。
这证明本批预测主要错在同类别内的SKU区分，支持“判对标准变严格”的解释；
但不能据此精确分摊图库、查询组成、视角、特征维数各自导致了多少差异。
98.75%仅是诊断值，**绝不替代新版85.125%的正式成绩，也不与旧94.375%当成同协议直接比较**。

原旧split在新查询中分布：原train480张，同SKU412/480、同类471/480；原val160张为131/160、159/160；
原test160张为138/160、160/160。新旧查询数量和构成也不同。
未见商品不等于未见类别：旧版10个类别在train/test两边都存在。

## 2. Qwen 到底执行什么？

模型是 [Qwen/Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B)，
不是 Qwen3-VL-2B-Instruct，也不是光电模型的紧凑前端。固定HF revision：
`9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda`。
脚本会校验模型和处理器的10个关键文件SHA；主权重SHA为
`c73fa9caeddeb3ff831d46c085a7a5708343248ca777e90f2d486964464509c1`，不只检查文件夹名称。
官方支持64～2048维输出；这里64维是我们预先采用的比较口径，不代表完整2048维的上限。

计算链：图像 → 原生processor/视觉patch embedding → **完整冻结Vision Transformer** →
视觉merger/图文序列 → **完整冻结语言Transformer及最终归一化** → 最后有效token的2048维向量 →
取前64维 → L2归一化 → 余弦相似度排序。
不是平均所有token，不是用前64个token，不是额外训练一个2048→64的Linear。
代码使用Transformers提供的 `Qwen3VLForConditionalGeneration` 加载Embedding checkpoint，
实际调用 `.model(...).last_hidden_state`，不调用词表生成头，不生成标题或答案。
类名含ConditionalGeneration不表示加载了Instruct权重。

固定system指令（图库、查询完全相同）：

> Represent this catalog product image for category-aware visual similarity retrieval.

user只含一张图，无商品标题、类别名、SKU ID或答案。使用模型自身chat template，
`add_generation_prompt=True`；按attention mask找最后非padding位置，兼容左右padding。
该prompt是历史沿用的类别感知指令，可能不是实例检索的最优指令；本次复现不偷偷修改prompt。

主结果为 **native**：EXIF方向矫正→RGB，processor最小/最大像素预算均50176（224²），
保持原生长宽比并按模型网格对齐，**不是所有图都强制224×224裁剪**。
旧版还提供square对照：RGB、中心裁剪224²、BICUBIC，不EXIF转置；旧报告中64维为92.7083%、2048维为93.9583%。
旧native2048为95.2083%。不能把这些不同模式的数字混在一起。

数值细节：旧流程将归一化2048维特征以FP16保存后再截64维并归一化，本脚本保留此舍入步骤；
新版直接将最后token的前64维转换FP32、L2归一化。默认旧batch1、新batch4，BF16/SDPA。
GPU/torch版本和批大小变化可能使临界相似度排名略变，不能为凑分删样本或更改正确答案。

## 3. 文件与依赖

ZIP解压后：

```text
baseline.py                 独立数据校验、完整Qwen推理和指标计算
test_baseline.py             无GPU的逻辑测试
requirements.txt
README.md
PACKAGE_MANIFEST.json        每个文件的SHA和源码commit
evidence/
  abo_similarity10_manifest.csv
  enrolled_protocol.json    固定8/4名单，不要重新随机分
  legacy/features.pt        历史native/square特征；不是模型权重
  legacy/baseline_report.json
  legacy/run_manifest.json
  enrolled/features.pt      历史新版64维特征
  enrolled/final_report.json
  independent/              如存在：本次从原图重新推理的报告和逐图预测
```

Python3.11。服务器原环境torch2.6.0+cu124、transformers4.57.3；其余依赖见requirements。
先安装适合显卡/驱动的CUDA版torch，再 `python -m pip install -r requirements.txt`。
不同显卡例如5090不能盲目沿用4090的CUDA构建。CPU缓存核算不需要GPU或下载Qwen。

```bash
python -m unittest discover -s . -p test_baseline.py
```

所有命令从解压目录执行，输出目录必须不存在；想重做时改一个明确run名，不覆盖历史证据。
不要将下方历史缓存审计称为“重新跑了大模型”。

## 4. 第一步：CPU复算两个原始baseline（无需原图/模型）

下面单行命令也可直接用于PowerShell。

```bash
python baseline.py audit --protocol legacy_category --manifest evidence/abo_similarity10_manifest.csv --features evidence/legacy/features.pt --expected-features-sha256 38f77637e48cf28fb7b1e077119bcf4c1ea37dd3480cbd1b6ce5707f05b38324 --output results/legacy_cached
python baseline.py audit --protocol enrolled_sku --manifest evidence/abo_similarity10_manifest.csv --enrolled-manifest evidence/enrolled_protocol.json --features evidence/enrolled/features.pt --expected-features-sha256 c6eb631c268d2446a2f783854c86d8493cdbcaa04c0669148b16d9785016d8d7 --output results/enrolled_cached
```

旧版预期 `metrics_by_dimension.64.hit_at_1=0.94375`、2048为0.9520833333；
新版预期64为0.85125，`same_ranking_category_hit_at_1_diagnostic=0.9875`。
每份输出 `report.json` + `predictions_64d.csv`（旧版另有2048D）。
新版历史缓存只有64维，不能靠补零“还原2048维”。

## 5. 第二步：从原图重新执行完整冻结Qwen

需要原数据包：服务器 `/DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data_only.zip`。
其SHA256应为 `c8f0f79cbd8ceb3c42162092136254a1f44f6d13b3329e0e3b1c5dfc458c1485`。
解压后的DATA根应直接含 `data/abo_similarity10_manifest.csv`，不要多套一层根目录。
该CSV SHA固定为 `2949a4035150a9f8718f2a6cace164c17394613d24fb9d0234c553bee8d77c97`。
MODEL指向完整Embedding模型快照，至少包含safetensors、config、tokenizer、processor、chat template文件。
**这些数据和4.26GB模型权重不在小ZIP中；特征缓存不能代替从原图验证。**

Linux实验室服务器（先确认GPU空闲，只用一张，不碰他人进程）：

```bash
export DATA=/DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data
export MODEL=/DATA/DATA1/guest3/.cache/huggingface/hub/models--Qwen--Qwen3-VL-Embedding-2B/snapshots/9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
nvidia-smi
# GPU0仅作示例；必须先确认空闲。顺序执行，不需要同时占两张卡。
CUDA_VISIBLE_DEVICES=0 python baseline.py infer --protocol legacy_category --manifest evidence/abo_similarity10_manifest.csv --data "$DATA" --model "$MODEL" --output results/legacy_native
CUDA_VISIBLE_DEVICES=0 python baseline.py infer --protocol enrolled_sku --manifest evidence/abo_similarity10_manifest.csv --enrolled-manifest evidence/enrolled_protocol.json --data "$DATA" --model "$MODEL" --output results/enrolled_native
```

Windows：只需把变量换为实际解压位置，命令仍用相同脚本：

```powershell
$env:CUDA_VISIBLE_DEVICES = '0'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$Data = 'E:\datasets\abo_similarity10_data'
$Model = 'E:\models\Qwen3-VL-Embedding-2B'
python baseline.py infer --protocol legacy_category --manifest evidence/abo_similarity10_manifest.csv --data $Data --model $Model --output results/legacy_native
python baseline.py infer --protocol enrolled_sku --manifest evidence/abo_similarity10_manifest.csv --enrolled-manifest evidence/enrolled_protocol.json --data $Data --model $Model --output results/enrolled_native
```

均禁止联网自动切换模型。两组都会额外输出2048维结果供审查，不把补充维度替换为原64维口径。
旧square可另加 `--preprocessing square --output results/legacy_square`，其他不改。
输出记录模型文件/图像SHA、源码/环境、精度、prompt、完整逐图结果、可重算的 `features.pt`。
推理无optimizer/epoch；进程退出会释放GPU。墙钟耗时含加载/读盘，不能当论文推理速度。

## 6. 指标与核验顺序

1. 对上模型Embedding身份、源码版本、数据与两份manifest SHA；不是Instruct或学生模型。
2. 对上64/2048维、native/square、BF16、batch、归一化和最后有效token。
3. 对上480×120商品中心（同类）或800×1600单图（同SKU），不做类别预筛选。
4. 旧版每视角先L2，12视角平均后再L2。新版直接对单图排序，不用商品投票或多图平均。
5. Hit@K：前K是否出现至少一个正例。集合Recall@K：前K正例数/全部正例数，二者不同。
   本文R@1沿用Hit@1；新版命中1张时集合Recall@1只记1/8。指标不使用标题文本。
6. 若临界分数不同，检查逐图预测和环境；不能只改汇总值。GPU显存不足可以减小batch，但须记录。

来源：旧原run `frozen_qwen_20260912`，commit `09835251d7dc0072bd7653f0529b97c9b0306049`；
新原run `abo200_enrolled_qwen64_20260913`，commit `5f731fae979ef81a09770c1dd721b5cecd8b3392`。
旧 `BASELINE_METHODS.md` 是更早的A100/square版本历史说明，不能拿来覆盖这两次native基准。
本次独立代码是否从图像复跑、使用什么GPU/源码，以及实际结果，以ZIP内独立报告为准。

## 7. 边界与不能下的结论

- 可确认两次本地任务都没有训练Qwen；不能声称官方预训练完全未见过ABO。
- 新任务“学生见过SKU”与“必须认出具体SKU”同时发生，不构成难度严格单调下降的实验。
- 109/119同类混淆是当前样本证据，不是Qwen架构的普遍定律，也不是单因素因果实验。
- 比较光电模型与Qwen必须在各自同一协议内部进行。不能用光电新协议81.25%对旧Qwen94.375%。
- 光电模型周期TEST选模的偏差仍需披露；冻结baseline没有在本协议做参数训练或挑epoch。
- 数据版权以原ABO授权和实际再分发范围为准，小包不包含原图，不宣称“保证Nature版权通过”。
