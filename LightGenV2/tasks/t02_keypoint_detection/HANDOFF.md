# LSP 关键点检测交付（2026-09-09）

先进入本文件所在包根目录（应包含 LightGenV2、experiments、data）。
源码未改模型、光路、损失；仅整理完整依赖交付。不要覆盖旧工程或历史 run。

## 已有模型

| profile | 内容 | 原记录 test PCK@0.2 |
|---|---|---:|
| main_dc20_no_shift_warmstart | 最新候选：旧光电body/head/feature phase初始化，新训光Router，无像素平移 | 0.7305 |
| main_dc20 | DC20、随机公共初始化、有位移扰动 | 0.5773 |
| main_dc20_no_shift | 随机公共初始化、关闭位移 | 0.5562 |
| d2nn_dc20 | 两层dense D2NN对照 | 0.6751 |
| Frozen Qwen Vision + DeconvPoseHead | 仅监督训练姿态头的baseline | 0.7217 |

光学四种 profile 的 best/last、原配置与测试结果均已打包。Qwen baseline 训练好的
teacher_best_train_loss.pt、两种初始化权重也包含在内。数字来自已有结果，非本次重新训练。
数据包含 LSP 2000 张及完整 HR-LSPET；实际协议 train10428/test1000，14关节。
用标注确定人体裁剪范围，是给定人体定位的姿态估计，不含人体检测。

## 环境与本地Qwen路径

原实验 Transformers 4.57.3。5090 请使用支持其硬件的 PyTorch（如服务器已有 cu128），
不要覆盖师姐原环境。示例独立环境：

```bash
python -m venv --system-site-packages ../env
source ../env/bin/activate
python -m pip install 'transformers==4.57.3' scipy pillow pyyaml matplotlib tqdm pytest
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p cache/qwen/models--Qwen--Qwen3-VL-Embedding-2B/snapshots
ln -s /root/autodl-tmp/0_xyli/abo/Qwen3-VL-Embedding-2B cache/qwen/models--Qwen--Qwen3-VL-Embedding-2B/snapshots/local
```

环境与链接只需首次创建；已有链接不要重复运行 ln，也不要把 model_id 改成本地路径。
链接仅引用服务器现有Qwen，不复制或改变权重。换机器只改链接目标。

## 先检查，再评估最新光学权重

```bash
python -m LightGenV2.tasks.t02_keypoint_detection.run --help
python -m pytest LightGenV2/tasks/t02_keypoint_detection/tests -q
python -m LightGenV2.tasks.t02_keypoint_detection.run --profile main_dc20_no_shift_warmstart --phase prepare --run-dir LightGenV2/tasks/t02_keypoint_detection/runs/smoke/sister_data_check
python -m LightGenV2.tasks.t02_keypoint_detection.run --profile main_dc20_no_shift_warmstart --phase evaluate --checkpoint LightGenV2/tasks/t02_keypoint_detection/runs/simulation/moe_router_scale_dc20_no_shift_warmstart0713_seed42/best_checkpoint.pt --run-dir LightGenV2/tasks/t02_keypoint_detection/runs/simulation/sister_warmstart_eval01
```

evaluate 不训练。输出目录使用新的run ID，勿指向原训练结果。
本次包内 dataset 已解压，不需要再下载zip；若仍提示下载，先检查当前工作目录及数据结构。

## Qwen baseline 固定权重复评（5090）

```bash
python -m LightGenV2.tasks.t02_keypoint_detection.baseline_5090d --model /root/autodl-tmp/0_xyli/abo/Qwen3-VL-Embedding-2B --data-root data/lsp_pose --config experiments/qwen3_vl_embedding_2b_lsp_pose_optical_moe16/configs/lsp_pose_opt2.yaml --checkpoint experiments/qwen3_vl_embedding_2b_lsp_pose_optical_moe16/runs/lsp_pose_optical_moe16_opt2/checkpoints/teacher_best_train_loss.pt --run-dir LightGenV2/tasks/t02_keypoint_detection/runs/simulation/sister_qwen_eval01
```

完整冻结Vision Transformer、merger前196×1024 token、1.103M参数反卷积头，
输出14×56×56热图，不执行语言网络。此入口含计时功耗；GPU有其他任务时不要当正式计时。

## 修改及重新训练

```bash
python -m LightGenV2.tasks.t02_keypoint_detection.run --profile main_dc20_no_shift_warmstart --phase all --run-dir LightGenV2/tasks/t02_keypoint_detection/runs/simulation/sister_warmstart_train01
```

训练100epoch，周期test按EMA PCK选best。新方法用新的profile/run ID；不要用改过的配置
加载旧权重后冒称原模型性能。其他版本修改 --profile 和对应checkpoint即可。

完整方法及baseline训练说明：
LightGenV2/tasks/t02_keypoint_detection/reports/reproduction/BASELINE_METHODS.md。
光学主干是两级192宽的卷积/MLP电子支路+光学支路，并非纯光；读出头与Qwen baseline不同。
性能比较时需同时披露初始化、扰动、读出头及选模差异。

PACKAGE_MANIFEST.json记录源码commit及每个文件SHA256；ZIP旁另有整包校验文件。
包内完整兼容源码不意味着包含其他任务的原始数据。Qwen大模型复用服务器现有目录。
