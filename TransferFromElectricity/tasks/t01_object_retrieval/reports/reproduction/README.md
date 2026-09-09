# 固定专家库实验复现入口

本入口覆盖第五轮 `spatial_v4` 和第六轮 `unseen_v1`（未见类别操作见文末）；此前文本生成、随机相位初始化和短计时实验仍以各自历史报告及 run 为准，不能混合数值。

## 模型、训练和指标合同

完整说明见 [空间生成结构与训练说明](../spatial_v4_20260908/空间生成结构与训练说明.md)。五组分别是直接相位优化、CLIP视觉空间生成、Qwen视觉池化消融、Qwen视觉空间生成、Qwen同时生成expert/global。Qwen生成器只执行视觉塔，训练q/v LoRA及空间解码器；光电学生仍保留Vision/Language两侧结构。固定参考图来自训练划分，每类一张，始终不输入当前查询。

共十二个可训练相位平面从raw零开始，经sigmoid约束为π；间隙和保护区的物理相位为0。所有组共用非相位电子warmstart，融合系数固定0.6。训练分为4轮experts、8轮optics、18轮joint，每轮120个batch、每batch十类各三张图，seed42。直接相位及视觉生成不是同一参数化，因此不声称有效像素步长相等。

任务是图像到十个类别原型的检索。类别原型来自独立gallery，每类图像embedding取均值；Top-1/Top-3表示正确类别原型的排名，不是任意实例检索的Recall@K。标签用于训练损失和构造类别原型。只按validation Top-1、其次MRR选择live checkpoint；结束后重新载入best，计算其test成绩。EMA不用于本轮选模。

Caltech使用原十类，CIFAR-100使用预先固定类别编号 `[3,13,14,17,28,31,35,81,86,94]`，Imagenette使用官方十类。各run的 `split.json` 包含完整训练/验证/gallery/test样本身份，`data_hashes.json` 是逐图片SHA256。分割交集必须为空。Caltech历史warmstart和测试集曾被本项目使用，不能称为新独立测试。

CIFAR十类和Imagenette的test也在此前轮次被本项目查看过。当前run内部没有使用test选checkpoint，不等于整个跨轮次研究从未接触test；三套数据均不能称为首次盲测。

CIFAR的训练、选模validation和gallery来自官方train，test来自官方test。Imagenette使用现有`imagenette2-160`缓存，官方train划出9,119张train、300张validation与50张gallery；官方val的3,925张图作为本项目test。Caltech对应2,525/100/30/200张图。不要将Imagenette的官方val同时用作选模集合，也不要把本结果标成原始高分辨率版本的评测。

## 固定源码和资源

正式训练源码：`c024b9280433f6e7fe31fc0122a1b8aadf342b38`。报告代码后续提交不会改变这批训练的源码SHA。

服务器Python为 `/home/guest3/miniconda3/envs/xml/bin/python`，Python3.11.15、PyTorch2.6.0+cu124、Transformers4.57.3；逐run的环境和实际设备见 `environment.json`。确定性算法开启，cuDNN benchmark关闭，`CUBLAS_WORKSPACE_CONFIG=:4096:8`。模型均离线读取已有缓存。

本次服务器依赖核验还包括 torchvision0.21.0+cu124、NumPy1.26.4、Pillow12.2.0、PyYAML6.0.2、safetensors0.8.0、accelerate1.14.0、SciPy1.17.1，以及转换官方CLIP缓存时使用的openai-clip1.0.1。不要误用服务器系统的`python3`，其Python3.8环境没有安装本任务的Torch/Transformers。

| 资源 | 身份 |
|---|---|
| Qwen3-VL-2B-Instruct snapshot | `89644892e4d85e24eaac8bacfd4f463576704203` |
| Qwen model.safetensors SHA256 | `7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0` |
| 非相位warmstart checkpoint SHA256 | `6a27f54d8c869cce46150583383a127b0ba47b3d34503f5753aa23974ac1e55d` |
| 原OpenAI CLIP ViT-B/16缓存SHA256 | `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f` |

Qwen各配置文件及完整权重的SHA保存在Qwen组的 `generator_source_manifest.json`；CLIP组同名文件记录转换后的视觉权重SHA，转换manifest记录原始缓存身份与逐patch token一致性误差。实际warmstart路径和SHA见 `initialization.json`，零相位重置发生在其载入之后，见 `zero_initialization.json`。最终权重和数据清单均以run中记录及报告 `evidence_manifest.json` 为准，不用文件修改日期判断版本。

## 重新训练

先从GitHub获取上述SHA，创建独立detached worktree；仅链接已有数据、基座权重缓存和warmstart所需runs。不要在其他任务运行的工作树上切换版本。所有命令在仓库根目录执行。

```bash
python -m TransferFromElectricity.tasks.t01_object_retrieval.run_spatial_suite \
  --gpu-uuids GPU-4d8bfdb9-8777-05a6-3811-ab18ff4eadfd \
  --datasets caltech --prefix <新的唯一run前缀> --wait-for-gpus
```

GPU参数必须换成当前确认空闲的完整RTX UUID；启动器核验型号并拒绝A100。同一数据集五组共用同一卡顺序执行。将 `--datasets` 改为 `cifar` 或 `imagenette` 可运行另一个数据集；不指定 `--methods` 就执行五组。`--smoke` 仅运行每阶段一轮、每轮两batch并且validation-only，不能用于正式数值。

单组等价命令：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=42 \
python -m TransferFromElectricity.tasks.t01_object_retrieval.launch_rtx \
  --gpu-uuid <完整RTX-UUID> --method qwen_vision_global_lora \
  --config TransferFromElectricity/tasks/t01_object_retrieval/configs/spatial_v4/imagenette.yaml \
  --steps-per-epoch 120 \
  --run-dir TransferFromElectricity/tasks/t01_object_retrieval/runs/simulation/<新的唯一runID>
```

正式run的实际参数、模型路径和继承后的配置以 `environment.json` 的command、`gpu_execution.json` 的完整启动命令、`protocol.json` 和 `config.yaml` 为准。原始命令保存在每个run中；不能使用 `--resume` 覆盖已完成实验来冒充独立重跑。

## 固定权重检查、产物和汇总

每个正式run保存 `best_checkpoint.pt`、`last_checkpoint.pt`。结束时已经实际重载验证集所选best、重建生成相位，并检查其与checkpoint内相位一致；再执行test和干预评估。随后将expert/global物化成可部署相位，检查生成模式与物化模式embedding一致。`expert_bank.pt` 包含两侧expert/global的raw和物理相位，但完整推理仍需best checkpoint里的Router、电子接口和readout。

上述属于本次训练末尾的固定权重评估与导出校验；尚未声称在独立新进程、其他机器或实验台完成第二次固定权重复评。图中相位由实际bank绘制，不是示意生成图。

服务器原始产物位于 `/DATA/DATA1/guest3/2026OpticsMoE/TransferFromElectricity/tasks/t01_object_retrieval/runs/simulation/20260908_v4_formal_<dataset>_<method>_s42`。轻量产物通过 `collect_results` 回传；每个文件均与远端独立 `sha256sum` 比对，收据在 `transfer_manifest.json`。大checkpoint保留服务器，不提交Git。收集时加 `--checkpoint-hashes`，将两份大权重的远端SHA256及字节数记录到本地 `checkpoint_hashes.json`，不下载权重；汇总清单以 `evidence_type=remote_checkpoint_sha256` 明确区分这些记录与已回传文件。

从报告版本所在仓库根目录收集正式产物的Bash命令如下，密码通过终端提示输入，不写入命令或文件：

```bash
runs=()
for dataset in caltech cifar imagenette; do
  for method in direct clip_vision_lora qwen_vision_pooled_lora qwen_vision_lora qwen_vision_global_lora; do
    runs+=("runs/simulation/20260908_v4_formal_${dataset}_${method}_s42")
  done
done
python -m TransferFromElectricity.tasks.t01_object_retrieval.collect_results \
  --host 202.120.62.181 --port 24096 --user guest3 \
  --remote-root /DATA/DATA1/guest3/2026OpticsMoE \
  --require-gpu-audit --checkpoint-hashes --runs "${runs[@]}"
```

短计时产物用同一命令收集，将run列表换为五个`runs/smoke/20260908_v4_timing_cifar_<method>_s42`。源代码仍通过Git获取，不能用这个产物收集器覆盖服务器源码。

收齐十五组后执行：

汇总使用包含`report_spatial.py`与`report_spatial_timing.py`的报告版本`e35328e7`（其中短计时脚本来自`c547e0f8`），在本地读取已校验的run产物。正式训练SHA `c024b928`早于这两个汇总入口，不能直接在该训练worktree调用新报告脚本；也不要为了出报告切换正在训练的worktree版本。

本机实际生成报表使用`C:/Users/Xml12/.conda/envs/qwen3vl-cifar10/python.exe`。Windows PowerShell中可将下方`python`替换为`& 'C:/Users/Xml12/.conda/envs/qwen3vl-cifar10/python.exe'`；产物收集器只需要带Paramiko的Python环境，读取相位张量和绘图则需要Torch与Matplotlib。

```bash
python -m TransferFromElectricity.tasks.t01_object_retrieval.report_spatial
```

汇总脚本拒绝不完整或不兼容证据，输出全部Top-1/Top-3、450个epoch时间、45组阶段均值、固定validation阈值首次达到时间、完整拼接专家/global层及来源清单。各项计时是CUDA同步的仿真训练墙钟时间；不表示实验台耗时、光传播时间或功耗优势。单seed、共享CPU/磁盘和可能的外部GPU进程限制见具体结果报告。

正式15组已完成，汇总得到450条epoch记录。阅读 [完整结果](../spatial_v4_20260908/完整结果.md)、[结构与训练说明](../spatial_v4_20260908/空间生成结构与训练说明.md)；[证据清单](../spatial_v4_20260908/evidence_manifest.json)保留关键文件校验与30份远端best/last checkpoint哈希，[执行回传清单](../spatial_v4_20260908/execution_transfer.json)对应六份队列/设备审计。模型权重不提交Git，已有run与固定训练worktree均保留。

## 补充的同卡短计时复测

短计时固定源码为 `c558fd03054c9abc6e5db245a1017106a2549a1b`。与正式SHA相比，模型、训练循环、数据集及所依赖光路实现没有变化；新增配置把三个阶段改成各两轮，并设为validation-only。配置为 `configs/spatial_v4/timing_cifar.yaml`。不传 `--smoke`，因为该开关会把每轮缩成两个batch。

```bash
python -m TransferFromElectricity.tasks.t01_object_retrieval.run_spatial_suite \
  --gpu-uuids <当前空闲的完整RTX-UUID> --datasets cifar \
  --config TransferFromElectricity/tasks/t01_object_retrieval/configs/spatial_v4/timing_cifar.yaml \
  --run-kind smoke --prefix <新的唯一计时前缀> --wait-for-gpus
```

本次使用 `20260908_v4_timing` 前缀，通过 `allocate_spatial_dataset` 的 `--reservation-prefix 20260908_v4_formal` 等待正式Caltech或CIFAR五组全部完成，再从GPU3/5中选择空闲卡。分配记录和每组GPU审计记录实际设备与源码SHA，不能仅凭队列名称判断显卡。

产物位于 `runs/smoke/20260908_v4_timing_cifar_<method>_s42`。收齐后运行：

```bash
python -m TransferFromElectricity.tasks.t01_object_retrieval.report_spatial_timing
```

脚本同时核对正式/短计时的配置、数据哈希、固定参考图、初始化权重和模型结构一致，仅允许阶段长度及validation-only开关不同。结果保留全部30条epoch记录，并固定比较epoch 2/4/6；不采用各组最快一轮，也不将短复测的validation成绩写成正式test结果。

## 第六轮：CIFAR未见类别迁移

训练源码固定 `0d854376cca29e16143ad1eb7762860770e59459`，服务器独立worktree `/DATA/DATA1/guest3/worktrees/static_unseen_v1`；基础环境和冻结Qwen权重与第五轮相同。依赖的学生模型、光路与评估实现已核对相对源训练SHA无变化。共享data和本任务runs的既有路径，不切换其他任务checkout。

源模型仅使用 `20260908_v4_formal_cifar_direct_s42` 和 `20260908_v4_formal_cifar_qwen_vision_lora_s42` 的best，来自源十类训练SHA `c024b928`。它们的选模只看源validation；每个新run的 `source_checkpoint.json` 保存实际完整权重SHA256及源选择轮次。源训练从raw0开始，本轮续训保持源光学相位。两方法各自继承不同的源电子/相位参数，不能宣称跨方法起点相同。

`configs/unseen_v1/cifar.yaml`锁定两组互斥新十类、每类5/20个支持样本、支持seed101/202/303。全部支持图来自官方train，且同时用于gallery及梯度训练；同seed的5-shot是20-shot子集，不再提供额外gallery或validation标签。每任务官方test的1,000张图均作为查询，仅在固定的初始/最终评估时使用。新任务本地标签映射为0–9，原始fine ID及图像身份保留在 `split.json`。所有训练/参考图均不得来自query。

Qwen仅视觉塔，q/v rank8 LoRA及空间decoder更新；直接组仅八张expert raw phase更新。两组router/global/电子/readout/融合系数均冻结；支持参考图固定，每类一张，所有query共用同一专家库。适配固定5轮×20步、PK10×3，不增强、不依据新query选模。`best_checkpoint.pt`在本协议中就是预先固定的末轮，与last权重相同，不表示按test挑选的最好轮次；脚本不提供恢复训练入口，失败重试必须新run ID。

每个run首先用源bank计算新任务检索。Qwen另执行一次仅换支持参考图、保持源offset的无梯度生成诊断；之后将offset设成 `G适配开始(新参考)-源expert`，保证正式适配从源mask继续。诊断需要新类别支持图，不是完全无样本生成；正式适配的起点也不是这一未校准输出。

在独立worktree中核验Git版本与空闲RTX UUID后运行（系统Python不能替代xml环境）：

```bash
python -m unittest TransferFromElectricity.tasks.t01_object_retrieval.tests.test_unseen_protocol -v
python -m TransferFromElectricity.tasks.t01_object_retrieval.run_unseen_suite \
  --gpu-uuid <空闲RTX完整UUID> --class-set a --prefix <唯一smoke前缀> --smoke
python -m TransferFromElectricity.tasks.t01_object_retrieval.run_unseen_suite \
  --gpu-uuid <空闲RTX完整UUID> --class-set a --prefix <唯一正式前缀>
python -m TransferFromElectricity.tasks.t01_object_retrieval.run_unseen_suite \
  --gpu-uuid <另一张空闲RTX完整UUID> --class-set b --prefix <同一正式前缀>
```

每个class-set队列依次运行12组；smoke只运行两方法、5-shot/seed101，每组1轮2步，查询也来自官方train且与支持集不重叠。已有前缀为 `20260909_unseen_smoke` / `20260909_unseen_formal`。队列遇到占卡会等待，GPU PID采样每5秒记录实际时间戳，不占用A100。

单run等价命令：

```bash
python -m TransferFromElectricity.tasks.t01_object_retrieval.run_unseen \
  --gpu-uuid <空闲RTX完整UUID> --class-set a --method qwen_vision_lora \
  --shots 5 --support-seed 101 \
  --run-dir TransferFromElectricity/tasks/t01_object_retrieval/runs/simulation/<唯一run_id>
```

使用本入口前述 `collect_results --require-gpu-audit --checkpoint-hashes` 经SFTP及独立SHA256收集，run路径为 `runs/simulation/20260909_unseen_formal_<a或b>_<direct或qwen_vision_lora>_<5或20>shot_s<101或202或303>`。逐查询预测/混淆矩阵、数据与权重哈希、固定参考、任务梯度、冻结变化和逐轮时间均保留在run；大型源权重与适配best/last不提交Git。

收齐24组后，使用包含汇总入口的报告版本运行 `python -m TransferFromElectricity.tasks.t01_object_retrieval.report_unseen`。报告检查两方法数据/显卡配对、支持集嵌套、所有查询未入训练、源权重一致、导出误差、冻结参数及回传哈希。均值/标准差来自三次支持抽样，不能标成三个源训练seed，也不能将单源任务的条件外推称为经过跨任务元训练。当前过程包括源checkpoint重建、固定末轮评估与部署mask一致性核验；尚未另启全新进程对适配checkpoint重评，不能声称已做该项复现。
