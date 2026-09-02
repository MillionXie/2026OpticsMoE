# 服务器运行命令

所有命令从仓库根目录执行。正式实验只运行 `alpha20/alpha35/alpha50` 三个固定 Top-2 光路由配置；不要再启动 LSP、电子 Router、Top-1 或 Top-4。

## 1. AutoDL：生成完整 Qwen3-VL-Instruct main-merger 缓存

```bash
cd /root/autodl-tmp/workspace/opticsmoe
conda activate xml
```

确认以下两个目录存在：

```text
/root/autodl-tmp/workspace/LGVQ
/root/autodl-tmp/models/Qwen3-VL-2B-Instruct
```

然后执行：

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_lgvq_o2_109_highalpha_vqa.cache_qwen_inputs \
  --dataset-root /root/autodl-tmp/workspace/LGVQ \
  --model-path /root/autodl-tmp/models/Qwen3-VL-2B-Instruct \
  --manifest experiments/qwen3_vl_2b_lgvq_spatiotemporal_optical_router_vqa/artifacts/lgvq_prompt_group_split.csv \
  --output experiments/qwen3_vl_2b_lgvq_o2_109_highalpha_vqa/artifacts/qwen3vl_instruct_full_main_merger_196x2048.pt \
  --batch-size 4 \
  --device cuda
```

必须确认报告为：

```text
vision: [2808,4,196,2048]
language hidden width: 2048
feature_contract: qwen3vl_full_visual_main_merger_196x2048_v1
alignment: false
split: train=2250, validation=0, test=558
```

若模型实际放在别的绝对路径，只修改 `--model-path`；不要换回 Embedding 模型缓存，也不要使用旧的 `196×1024` pre-merger 缓存。

## 2. 将缓存同步到主服务器

登录主服务器后执行：

```bash
cd /DATA/DATA1/guest3/2026OpticsMoE
mkdir -p experiments/qwen3_vl_2b_lgvq_o2_109_highalpha_vqa/artifacts
scp -P 37626 root@connect.westd.seetacloud.com:/root/autodl-tmp/workspace/opticsmoe/experiments/qwen3_vl_2b_lgvq_o2_109_highalpha_vqa/artifacts/qwen3vl_instruct_full_main_merger_196x2048.pt experiments/qwen3_vl_2b_lgvq_o2_109_highalpha_vqa/artifacts/
```

不要把缓存提交 GitHub。两台服务器都必须保留 manifest、缓存和初始化 checkpoint；preflight 会检查形状、SHA、split 和架构合同。

## 3. Smoke 与三档 preflight

在任意一台服务器的仓库根目录执行：

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_lgvq_o2_109_highalpha_vqa \
  --config experiments/qwen3_vl_2b_lgvq_o2_109_highalpha_vqa/configs/release/alpha20.yaml \
  --phase smoke
```

```bash
python -m experiments.qwen3_vl_2b_lgvq_o2_109_highalpha_vqa --config experiments/qwen3_vl_2b_lgvq_o2_109_highalpha_vqa/configs/release/alpha20.yaml --phase preflight
python -m experiments.qwen3_vl_2b_lgvq_o2_109_highalpha_vqa --config experiments/qwen3_vl_2b_lgvq_o2_109_highalpha_vqa/configs/release/alpha35.yaml --phase preflight
python -m experiments.qwen3_vl_2b_lgvq_o2_109_highalpha_vqa --config experiments/qwen3_vl_2b_lgvq_o2_109_highalpha_vqa/configs/release/alpha50.yaml --phase preflight
```

只有三份报告全部为 `status=ready` 才进入正式训练。

## 4. 正式训练：三档各跑一次

推荐把 alpha20 放在 AutoDL，把 alpha35、alpha50 放在主服务器两张空闲 4090；三个实验同 seed，不重复跑无关对照。

AutoDL：

```bash
cd /root/autodl-tmp/workspace/opticsmoe
conda activate xml
mkdir -p experiments/qwen3_vl_2b_lgvq_o2_109_highalpha_vqa/logs
CUDA_VISIBLE_DEVICES=0 nohup python -u -m experiments.qwen3_vl_2b_lgvq_o2_109_highalpha_vqa \
  --config experiments/qwen3_vl_2b_lgvq_o2_109_highalpha_vqa/configs/release/alpha20.yaml \
  --phase train \
  > experiments/qwen3_vl_2b_lgvq_o2_109_highalpha_vqa/logs/alpha20.log 2>&1 &
```

主服务器：

```bash
cd /DATA/DATA1/guest3/2026OpticsMoE
conda activate xml
mkdir -p experiments/qwen3_vl_2b_lgvq_o2_109_highalpha_vqa/logs
CUDA_VISIBLE_DEVICES=0 nohup python -u -m experiments.qwen3_vl_2b_lgvq_o2_109_highalpha_vqa \
  --config experiments/qwen3_vl_2b_lgvq_o2_109_highalpha_vqa/configs/release/alpha35.yaml \
  --phase train \
  > experiments/qwen3_vl_2b_lgvq_o2_109_highalpha_vqa/logs/alpha35.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python -u -m experiments.qwen3_vl_2b_lgvq_o2_109_highalpha_vqa \
  --config experiments/qwen3_vl_2b_lgvq_o2_109_highalpha_vqa/configs/release/alpha50.yaml \
  --phase train \
  > experiments/qwen3_vl_2b_lgvq_o2_109_highalpha_vqa/logs/alpha50.log 2>&1 &
```

查看状态：

```bash
tail -f experiments/qwen3_vl_2b_lgvq_o2_109_highalpha_vqa/logs/alpha35.log
nvidia-smi
```

训练在 epoch 1、每 5 epoch 和 epoch 60 测试。每档输出：

```text
runs/o2_109_alphaXX/best_observed_test_checkpoint.pt
runs/o2_109_alphaXX/metrics_best_observed_test.json
runs/o2_109_alphaXX/train_history.json
runs/o2_109_alphaXX/training_summary.json
```

## 5. 对最佳 checkpoint 做完整评估

以 alpha35 为例：

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_2b_lgvq_o2_109_highalpha_vqa \
  --config experiments/qwen3_vl_2b_lgvq_o2_109_highalpha_vqa/configs/release/alpha35.yaml \
  --phase evaluate \
  --checkpoint experiments/qwen3_vl_2b_lgvq_o2_109_highalpha_vqa/runs/o2_109_alpha35/best_observed_test_checkpoint.pt
```

alpha20、alpha50 只需替换配置和目录名。正式移交时同时保留 `test_metrics.json`、`test_predictions.csv`、`fusion_diagnostics.json` 和 `router_diagnostics.json`。

## 6. 明确禁止的旧命令

以下内容不属于当前任务，不再启动：

```text
qwen3_vl_embedding_2b_lsp_pose_optical_router
electronic_power_topk1 / topk2 / topk4
LGVQ e1 / e2 / e4 electronic-router runs
alignment target training
```


