# T13 schema=4 当前复现命令

四组正式训练已完成；下列仅是操作说明，不自动重训或采集。
事实见[唯一复现入口](reports/reproduction/README.md)，
指标见[正式报告](reports/FULL2250_TESTBEST_RESULTS.md)。旧schema3不用于当前部署。

进入任务目录；历史Python是`/home/guest3/miniconda3/envs/xml/bin/python`，
依赖见requirements.txt和requirements-dev.txt。

## 不读取正式数据的检查

```sh
python -I verify_source.py
python -m pytest tests -q
```

paths.local.json仅登记原视频、冻结Qwen、manifest、两种缓存和TRAIN-only软标签的
本机路径，不含凭据、不进Git。新输出必须新建/空目录，不能覆盖旧run；
run.py的plan/preflight/smoke也会写输出，不是只读盘点命令。

## 已执行的最后15轮来源（历史记录，不复制即运行）

原supervisor通过train_four.py使用下列参数：

```text
--reuse-cache --gpus 1,2,5,6
--run-id full2250_testbest_phase_only15_s163_uuid1256_20260927
--finetune-parent-run <任务根>/runs/simulation/full2250_testbest_low_lr30_s163_uuid1256_20260927
--finetune-epochs 15 --finetune-lr-factor 0.03 --finetune-phase-only
--dataset-root /DATA/DATA1/lixinyue/xyli/data/LGVQ
--qwen-model /DATA/DATA1/guest3/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203
--manifest /DATA/DATA1/guest3/2026OpticsMoE/experiments/lgvq_four_stage_optical_electronic_109_no_attention_vqa/artifacts/lgvq_train2250_test558.csv
--soft-targets /DATA/DATA1/guest3/2026OpticsMoE/experiments/lgvq_four_stage_optical_electronic_109_no_attention_vqa/artifacts/training_only_teacher_predictions.pt
```

GPU编号不是当前空闲卡保证，也不授权现在用4卡。完整逐子进程命令、GPU UUID及退出状态
在原supervisor.json，训练commit为1aa3842c01ddb54c8b0d61f8428de391e11cdaaa。
不恢复旧optimizer；2250TRAIN/558TEST，不划VAL，每5epoch按无噪声TEST SRCC选best。

## 固定权重评价（需原数据和缓存）

以r0为例，其他组换自己的group与PT；输出用新的空目录。

```sh
python -I run.py --group r0_post --phase evaluate --paths paths.local.json --checkpoint <正式run>/r0_post/best_checkpoint.pt --device cuda --eval-split test --noise-scale 1 --noise-seed 20260927 --output <新的空评价目录>
```

默认共同8µm/DC30；noise-scale=1对应主结果表，不是选模noise-scale=0。
诊断--noise-scale 0对应final_test_clean_dc30；再加--eval-eta .20对应
final_test_clean_dc20，不再据此选PT。--phase theory只允许r0，复用G2权重的
17µm/noDC/noise0理想条件。未标定Poisson-Gaussian pilot不能当真实曝光或功率。

## 交付资产暂存

四组真best已存在。构建器通过--checkpoints的group→PT JSON映射核验身份；
不能将导师reference PT改名当四组权重。

```sh
python -I build_lab_package.py --group r3_ccd_dc_intrain --checkpoint <正式run>/r3_ccd_dc_intrain/best_checkpoint.pt --output <新的暂存目录>
```

状态仍是staged_not_capture_ready，不连接设备、不宣称LUT/ROI/BMP已验收。
现有生成工程保留，本轮单main治理不新增独立工程副本。

## 源码同步边界

遵守根AGENTS.md及LightGenV2/AI_RULES.md。审核过的main commit发布后，各端只在
干净状态及无运行占用时通过Git同步；保护旧有效工作树和未提交overlay。
不整仓覆盖、不强推、不自动建分支/worktree；私有连接脚本不进新源码交付，
数据/PT/缓存按manifest+SHA另存，用户暂停的外部上传不恢复。
