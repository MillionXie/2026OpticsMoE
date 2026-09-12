# 同一权重去光复评（不重新训练纯电子模型）

目的：测量当前权重对光学分支的依赖，不把alpha或相位更新幅度直接当成性能贡献。
标准仍为完整SALICON官方val2014的5000张，每图CC取平均，公开测试参与过选模。
不改变图像预处理、标签、读出归一化或评价精度；正常测试关闭随机光学扰动。

`recheck_aligned --ablation remove_optical`调用已有运行时消融：
跳过光router、专家传播及全局传播，两级融合分别直接返回原电子残差E，系数为1。
不是将O置零后仍保留(1-alpha)衰减，不重训电子分支、不改相位文件、不修改保存的alpha。
正常方法仍要求alpha≥0.4；旁路仅用于这个明确标记的消融，不能用于满足正常模型的0.87目标。
Qwen不接受去光选项；默认`--ablation none`保持原复评行为。

## 命令与身份

在已配置SALICON数据/Qwen本地缓存的Git工作树运行，选择可用GPU，保留已存在的复评目录。
每次命令的输出目录必须尚不存在。示例候选是空间CC KD2的epoch5；两次必须核对相同SHA。

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
TASK=LightGenV2/tasks/t03_saliency
CONFIG="$TASK/configs/moe_alpha40_sam_spatialcc_kd2.yaml"
CHECKPOINT="$TASK/runs/simulation/moe_alpha40_sam_spatialcc_kd2_seed42/best_checkpoint.pt"
python -m LightGenV2.tasks.t03_saliency.recheck_aligned --system optical --config "$CONFIG" --checkpoint "$CHECKPOINT" --run-dir "$TASK/runs/simulation/aligned_recheck_20260910_kd2_ablation_normal" --batch-size 32 --ablation none
python -m LightGenV2.tasks.t03_saliency.recheck_aligned --system optical --config "$CONFIG" --checkpoint "$CHECKPOINT" --run-dir "$TASK/runs/simulation/aligned_recheck_20260910_kd2_remove_no_ccd" --batch-size 32 --ablation remove_optical
```

报告的`checkpoint_sha256`绑定实际读入的字节。示例候选应为：
`87ad4db51e3f58f9a41d6df09092439e5a09e81e93f88a3bfe2d3d008fafb29a`。
若活动训练已更新best，请选择新目录，并明确这是另一个候选；不可沿用旧SHA/指标。
两份`reproduction.json`中的数据ID SHA也须一致，逐图差异按`per_image_cc.csv`中的sample_id连接。

报告CC绝对差`normal-removed`，若给相对下降则为`100*(normal-removed)/normal`。
相对下降不是“光学参数/功耗/信息贡献占比”，也不能由消融断言光学硬件优于重新训练的纯电子网络。
不测速度/功耗，不将旁路结果填入普通D2NN baseline。

正式数值与产物见下文；临时终端诊断不替代含逐图指标和SHA的正式产物。

## 首次独立旁路发现的接口问题

初版入口0c185bae在全新进程运行去光时，被继承的LSP `forward` 拒绝：
`Vision global CCD readout is unavailable`。正常预测`head(spatial)`不消费该CCD返回值，
但旧接口仍要求它存在；先正常再去光的同进程诊断会残留旧CCD缓存，不能据此宣称独立旁路验证通过。
失败目录`aligned_recheck_20260910_kd2_ablation_remove`保留，不包含成功的reproduction.json，不能作结果引用。

修复限定于T03的去光forward：主动清空旧CCD诊断值，保持电子latent/读出计算，第三返回值明确为None。
不伪造全零CCD，不执行一次“隐藏的正常光学预热”；正常模式直接调用原父类forward，不改正常推理。
新增测试覆盖全新进程等价的无CCD状态、旧NaN缓存、变化batch大小和正常路径直接委托。
去光模式的调用方必须接受“没有CCD诊断图”；它不是硬件CCD为空的容错开关。

## 完整5000张的正式结果

2026-09-10，A100、torch2.6.0+cu124、batch32，每种模式独立进程加载同一epoch5 EMA权重。
正常复评源码`0c185bae53a3ac65d75702593cbe7a3d708d02d9`；
修复后去光源码`e5e1ded1437235adb14e86ae929c91fa62efb0b4`，99项测试通过并push后执行。
两者正常路径保持原父类forward，修改仅处理旁路时不存在的CCD诊断返回值。
正常结果与更早的独立复评差约2.3e-8，不是新的训练改进。

|指标|正常光电|同权重去光，无重训|
|---|---:|---:|
|独立float64 CC|0.8620495784|0.8411720440|
|KLD，低好|0.11424452|0.13784606|
|SIM|0.82407412|0.80784451|
|NSS|0.96543062|0.93996568|
|AUC-Judd|0.76997777|0.76576228|
|MAE，低好|0.08014799|0.08443406|

按sample_id配对，正常模型在3717/5000张上CC更高；去光CC绝对下降.0208775344，
相对正常CC下降2.42184846%。这些数字支持该权重依赖光分支，但不能分解出独立、可相加的光电贡献。
因同时移除了光router/专家/全局计算，不能把全部差异归给某一张mask或某一层。
这也不是单独训练纯电子模型后的性能，不能替代普通D2NN或Qwen baseline。
目标.87仍按正常光电模型衡量，尚未达到。

两份报告加载的checkpoint SHA均为87ad4db5…8fafb29a（完整值见上），
test IDs SHA均为`625dec6bc15b2d737d39bc252cfa0c354de217fec0266dcda568913f4a3496d0`。
各自的`per_image_cc.csv`含5000个唯一且完全相同的ID，无按结果挑图。

- 正常目录：`aligned_recheck_20260910_kd2_ablation_normal`；`reproduction.json` SHA256：
  `c518af8c9d5cddbe9b0efe6a5f74fbeb4f0e5667f6ab2a0b59d23445e431d124`。
- 去光目录：`aligned_recheck_20260910_kd2_remove_no_ccd`；`reproduction.json` SHA256：
  `370acc6c89e4c024edc432cc1c334a4cd0f8d7647504b0ad32ecfe5ad27ec3f5`。

两目录均在本任务`runs/simulation/`下，保存实际命令、配置、源码及环境；
未删除先前失败目录，未更改正式训练best/last。公开测试选模偏差与标准clean-eval边界仍适用。

## 2026-09-13：batch8候选同权重旁路

run `moe_alpha40_sam_batch8_20260913_seed42`的epoch1 EMA已独立复评，
SHA `e4930c1264442d7056fd0451385cf8fb301e83dd580200e15c6b7a7fe934725c`。
训练在last第11轮停止，并非完成20轮。以下结果不覆盖上面的87ad历史消融。

|指标|正常光电|同权重去光，无重训|
|---|---:|---:|
|独立float64 CC|.8623960523|.8423025094|
|KLD|.11422249|.13743438|
|SIM|.82427605|.80846425|
|NSS|.96467616|.94043377|
|AUC-Judd|.76998876|.76592108|
|MAE|.08168397|.08534979|

CC绝对下降.02009354，相对下降2.32997%；3744/5000张正常光电CC更高。
alpha=.43068656/.44104564不是上述性能下降百分比，也不能解释为独立可加的贡献比例。
两份逐图CSV完整身份及checkpoint SHA一致，test IDs SHA仍为上文625dec6b…3496d0。
batch48、固定权重独立进程，无重训练；标准eval关闭随机光扰动，不是实测硬件结果。

在该run中查阅`candidate_recheck/reproduction.json`与`candidate_remove_optical/reproduction.json`，
以及各自`per_image_cc.csv`；摘要`candidate_summary.json`。
完整专家分布审计位于`candidate_selected_evaluation/selected_checkpoint_test_evaluation.json`：
2343/2629/2313/2715次、份额23.43/26.29/23.13/27.15%，有效专家数3.98050，无闲置专家。
源码fa4647d7；复评PID/PGID604876、606813及各自子进程均已退出，GPU1释放。
表中MAE沿用本项目原实现：prediction/GT各按自身最大值归一化后求逐像素绝对误差，
不是sum=1密度直接相减的MAE。两种模式同口径，不改历史指标定义。
