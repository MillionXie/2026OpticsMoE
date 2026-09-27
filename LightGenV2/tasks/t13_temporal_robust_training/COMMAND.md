# 独立入口与命令（当前schema=3）

当前共用`tanh(a/0.5)`振幅图、DC30；旧schema=2产物保留作历史，不覆盖。`train_four.py`在用户授权四卡后，先重建缺失缓存、严格验证identity，再四组训练，最后评估实际训练子集/validation/原test。只有自己的子进程由supervisor回收。

```bash
python -I train_four.py --gpus 3,4,5,6 --run-id bounded_dc30_ccd_v2_s163_20260927 --dataset-root /absolute/LGVQ --qwen-model /absolute/frozen_Qwen3VL2B --manifest /absolute/frozen_manifest.csv --soft-targets /absolute/training_only_teacher_predictions.pt
```

每台机器先核验空闲GPU；以上索引不是固定硬件要求。`supervisor.json`记录PID/退出/GPU释放；`comparison.json`记录最终三种划分的回归指标。重建缓存不使用测试标签拟合Qwen，冻结patch/position/embed_tokens；正式测试标签只用于最终评价。

进入本任务目录；Python 3.11 + torch2.6/PyYAML/numpy。使用 `python -I run.py`，不把仓库根放入 PYTHONPATH。以下训练/实采命令是后续操作，不是本轮已执行。

```bash
python -I verify_source.py
python -m pytest tests -q
python -I verify_reference.py --package /absolute/path/teacher_release_final_v2/lgvq_temporal_08044 --output runs/smoke/teacher_one_field.json
python -I run.py --group r0_post --phase plan --output runs/smoke/r0_plan
python -I run.py --group r3_ccd_dc_intrain --phase smoke --device cpu --output runs/smoke/r3_smoke
python -I run.py --group r0_post --phase preflight --paths paths.local.json --output runs/smoke/data_preflight
```

`paths.local.json` 仅包含机器私有资产路径，示例见 `configs/paths.example.json`；不含账号密码，不提交Git。相对路径基于任务根，绝对路径保持不变。所有输出目录必须新建/空目录，不能覆盖旧run。

## 训练（资产、映射和标定确认后，单GPU串行）

```bash
CUDA_VISIBLE_DEVICES=0 python -I run.py --group r0_post --phase train --paths paths.local.json --output runs/simulation/r0_post_s163
CUDA_VISIBLE_DEVICES=0 python -I run.py --group r1_ccd_post --phase train --paths paths.local.json --output runs/simulation/r1_ccd_post_s163
CUDA_VISIBLE_DEVICES=0 python -I run.py --group r2_ccd_dc_post --phase train --paths paths.local.json --output runs/simulation/r2_ccd_dc_post_s163
CUDA_VISIBLE_DEVICES=0 python -I run.py --group r3_ccd_dc_intrain --phase train --paths paths.local.json --output runs/simulation/r3_ccd_dc_intrain_s163
```

上述 CCD 组在经验参数尚未标定时会拒绝启动。若用户明确允许经验模型pilot，加 `--allow-uncalibrated-noise`；记录pilot身份，不称标定模型。GPU0只是命令占位，运行前查空闲卡并指定一张，不自动排队或占满多卡。

## G1理论参照与统一部署仿真

```bash
# G1复用G2权重，17um理想网格，无DC/噪声
python -I run.py --group r0_post --phase theory --paths paths.local.json --checkpoint runs/simulation/r0_post_s163/best_checkpoint.pt --output runs/simulation/g1_theory

# G2共同8um部署条件；默认无CCD随机噪声，但有共同eta=.20
python -I run.py --group r0_post --phase evaluate --paths paths.local.json --checkpoint runs/simulation/r0_post_s163/best_checkpoint.pt --output runs/simulation/g2_device_clean

# 同一个模型的噪声扫描，全部组使用同一noise-scale和noise-seed
python -I run.py --group r0_post --phase evaluate --paths paths.local.json --checkpoint runs/simulation/r0_post_s163/best_checkpoint.pt --noise-scale 1 --noise-seed 20260927 --output runs/simulation/g2_noise1
```

G3/G4/G5使用自己的group与checkpoint，其余测试参数相同。pitch测试用 `--device-pitch-um 17|12|8`；eta测试用 `--eval-eta 0|0.1|0.2|0.3`；都不增加训练组。schema=2 使用独立电子单位Poisson-Gaussian。noise-scale=s时，k'=k/s²、sigma_e'=sigma_e/s，得到信号shot及读噪声标准差约随s变化；暗电子均值保持不变。不是标定曝光倍数，不能将横轴写成真实光功率。使用 `--ccd-profile /absolute/path/calibrated.json` 指定同一相机参数；各组必须一致，参数及SHA写入run。

## 生成四份独立工程

```bash
python -I audit_teacher.py --zip /absolute/path/LGVQ_Temporal_08044_teacher_final_v2_20260923.zip --package /absolute/path/lgvq_temporal_08044 --output runs/smoke/teacher_zip_audit.json
python -I build_projects.py --output projects/temporal_four_v2 --teacher-package /absolute/path/lgvq_temporal_08044
cd projects/temporal_four_v2/01_baseline_post
python -I project.py plan
python -I project.py reference --device cpu --fields 1 --output runs/reference_smoke.json
```

构建后每组都有完整导师参考PT及35个输入；训练缓存仍需另配。其余三组同样使用 `project.py`，不用自己写 `--group`。若只需要服务器训练代码、没有导师资产，则构建时省略 `--teacher-package`；不能声称这种包支持旧导师推理。

重新训练后，用JSON映射四个group到其best checkpoint路径，并传 `build_projects.py --checkpoints /absolute/path/checkpoints.json`。构建器核验group身份，拒绝将旧导师PT放入四组weights。

## 暂存单组部署资产

```bash
python -I build_lab_package.py --group r3_ccd_dc_intrain --checkpoint runs/simulation/r3_ccd_dc_intrain_s163/best_checkpoint.pt --output releases/g5_staged
```

该命令仅暂存，不连接设备，不自动换mask，不生成宣称已验证的LUT/BMP。真正实采待hardware contract与幅相独立raster数值审计通过后接入。

全部真实结果到齐后，按tools/plot_results.py文档的JSON格式整理身份与指标：

```bash
python tools/plot_results.py --input reports/comparison.json --output reports/main_figure
```

绘图器拒绝把G2-G5纯仿真值标成hardware，拒绝G1/G2不同权重；不生成假想性能数字。

## Git服务器同步

管理机需paramiko；密码交互输入、已登记SSH host key。

```bash
python tools/server_sync.py --phase inspect
python tools/server_sync.py --phase sync --commit <本地已push的40位SHA>
python tools/server_sync.py --phase test --commit <同一SHA> --python <服务器实际Python绝对路径>
```

固定服务器为guest3@202.120.62.181:24096，仓库 `/DATA/DATA1/guest3/2026OpticsMoE`。代码目录为其 `.worktrees/t13_temporal_robust_20260927`，共享根工作树不修改。数据路径私有配置在新worktree单独登记，不使用SCP覆盖源码。
