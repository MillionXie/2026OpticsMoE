# T07 独立工程：日常操作顺序

以下从本任务文件夹或解压后的工程根目录执行。**不需要2026OpticsMoE、T01、experiments、完整Qwen权重或网络访问。**
当前支持固定ABO检索指令、一张224×224商品图；不是任意prompt的大模型服务。

## 1. 环境与文件

先激活已有Python环境，不创建/覆盖conda环境。根据本机显卡先安装适配的CUDA版PyTorch，再执行：

```powershell
python -m pip install -r requirements.txt
python run.py verify --assets assets
```

`assets`包含best.pt、processor、train_targets.pt和SHA256清单。完整包另有`data`，内含
`data/abo_similarity10_manifest.csv`以及它引用的图像；若数据已在其他位置，后续仅修改`--data`，不移动原数据。
不要修改assets内容；图像尺寸、prompt、朝向和相位物理参数不通过猜测更改。

## 2. 固定best：完整评估与去光对照

```powershell
python run.py evaluate --assets assets --data data --device cuda --batch-size 4 --output runs/simulation/evaluate_01
```

全量1440训练视角重建120个图库中心，480测试图全部查询；然后同权重去光复评。
输出`final_report.json`、逐图CSV、`retrieval_features.pt`、`phase_masks.png`。
`--output`必须是新目录，不覆盖已有结果；CPU调试可改`--device cpu`，但全量会慢。
这不是硬件采集命令；新版核心保留实测CCD注入接口，但本次包不声称已含设备SDK/自动播放流程。

## 3. 可选：独立微调

```powershell
python run.py finetune --assets assets --data data --device cuda --epochs 30 --steps 48 --batch-size 4 --output runs/simulation/finetune_01
```

`--batch-size`是评估batch；训练固定10类×2个不同商品=20。`--steps`是每epoch训练步数。
使用包内**训练集**教师目标，不加载完整Qwen；训练光学相位、Router、电子残差及读出，前端冻结。
保存best.pt和last.pt，EMA=0.99，每5epoch完整test选best；这是test-selected而非独立无偏估计。
新微调为fresh optimizer继续训练，不承诺与历史训练随机轨迹逐步相同。历史70%是固定best复现目标，
不是承诺任意微调都提升。先保留原assets；别用微调last直接替换交付best。

## 4. GPU使用规则

默认一个进程只用一张GPU，无DDP/多卡自动扩张。服务器先查看`nvidia-smi`，用UUID选择空闲卡：

```bash
CUDA_VISIBLE_DEVICES=空闲GPU的UUID python run.py evaluate --assets assets --data data --output runs/simulation/evaluate_02
```

日常只用一张；同一操作者最多两张，勿占用他人GPU进程。命令完成或Ctrl+C会退出进程，释放CUDA上下文；
结束后用`nvidia-smi`确认自己的PID消失。不要为清理显存杀别人的进程。

## 5. 可选：任务内教师预热 + 联合训练（新试验，不替换正式best）

需最新 Git 代码（旧 ZIP 未包含此新增 profile）：

```powershell
python run.py finetune --profile teacher_curriculum --assets assets --data data --device cuda --epochs 30 --steps 48 --batch-size 8 --output runs/simulation/teacher_curriculum_01
```

训练 batch=40（10类×4个不同商品）；`--batch-size 8`仍然只控制评估。默认30epoch包含4epoch预热、
22epoch光电联合、4epoch无蒸馏收尾。前端始终冻结，不添加TF/attention，不加载大模型。
仅使用原训练集教师缓存，并非新增外部数据集预训练。配置在`standalone/curriculum.json`。
alpha固定为输入best的四个实际系数（约0.087～0.104，不是0.4）；预热只更新电子，随后相位也更新。
联合阶段75%的batch保留原未调制/CCD噪声；收尾25%，其余为干净仿真。推理图和原评估口径不变。
每轮记录全部batch的专家选择比例、相位/电子参数更新、alpha及独立训练样本数。
开始先评估/保存epoch0为保底best，之后用EMA候选test选模；只保存best.pt、last.pt，不保证优化一定提升。
