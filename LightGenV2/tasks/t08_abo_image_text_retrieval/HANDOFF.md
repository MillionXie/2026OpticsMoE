# ABO 光 Router MoE 核对包

从解压后含 `LightGenV2/`、`experiments/`、`data/` 的根目录运行。
旧包不要覆盖：保留师姐已经修改的版本。此包是图搜文，不是图搜图。

包含完整受 Git 管理的兼容后端源码及配置、ABO easy100 的 4,800 train / 2,400 test
图像和 100 标题、两组 ABO best/last 权重及结果、教师缓存、训练必需的 Caltech
warmstart 初始化权重。历史 run 内 config 是原始证据，不要直接用于新机器训练。
Qwen 大模型权重不重复打包，使用服务器已有的 Qwen3-VL-Embedding-2B 目录。

## 1. 环境与完整性

原训练环境 PyTorch 2.6.0+cu124 / Transformers 4.57.3。5090 需支持其 GPU 的
PyTorch（服务器已有 2.8.0+cu128）；不要为了安装下面的依赖覆盖 torch。
建议在独立环境中安装 Transformers 4.57.3、numpy、scipy、Pillow、PyYAML、matplotlib、tqdm。
不要直接用 Transformers 5.x 冒充原实验环境；底层 Qwen 接口可能不同。

```bash
# 本次已在师姐服务器建立独立环境；不修改原 base 环境。
source /root/autodl-tmp/abo_handoff_deps_20260907/env/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m LightGenV2.tasks.t08_abo_image_text_retrieval.handoff verify
python -m LightGenV2.tasks.t08_abo_image_text_retrieval.handoff prepare --model-path /root/autodl-tmp/0_xyli/abo/Qwen3-VL-Embedding-2B
python -m LightGenV2.tasks.t08_abo_image_text_retrieval.handoff check
```

`prepare` 只首次运行，生成 `configs/handoff_balance.yaml`；重复运行会拒绝覆盖。
`qwen.model_id` 必须保留官方名称，它参与 checkpoint 身份校验，不应改成本地路径。
prepare 会在包根 `.handoff_hf/` 建立只引用已有模型目录的符号链接，不复制或修改大模型。
搬到其他机器后需重新生成该配置与链接。所有模型/光路设置继承正式强均衡配置。

## 2. 已训练模型核对（不训练）

```bash
python -m LightGenV2.tasks.t08_abo_image_text_retrieval.handoff smoke --device cuda:0 --run-dir LightGenV2/tasks/t08_abo_image_text_retrieval/runs/smoke/handoff_check
python -m LightGenV2.tasks.t08_abo_image_text_retrieval.handoff evaluate --device cuda:0 --run-dir LightGenV2/tasks/t08_abo_image_text_retrieval/runs/simulation/handoff_full_test
```

smoke 仅 2 张 test 加 100 标题，用来检查模型、初始化、ABO checkpoint 和前向是否可用，
不是正式性能。evaluate 完整评估 2,400 张，输出逐图预测、embedding、R@1/5/10、MRR。
输出目录必须尚不存在，重跑请用不同 run ID。历史强均衡参考 R@1=0.7983，
性能优先参考 0.7988；它们是 periodic-test-selected，非 sealed test。

## 3. 修改或重新训练

修改独立配置（保留模型、划分、ROI 合同用于公平对比），不要改历史结果目录。
以下会重新训练，不是只做评估，默认 40 epoch，best 按周期 EMA test R@1 保存。

```bash
python -m LightGenV2.tasks.t08_abo_image_text_retrieval.optical_moe --config LightGenV2/tasks/t08_abo_image_text_retrieval/configs/handoff_balance.yaml --device cuda:0 --seed 42 --epochs 40 --run-dir LightGenV2/tasks/t08_abo_image_text_retrieval/runs/simulation/sister_balance_trial01
```

方法、光路、损失与原始结果请看本任务 README。此包没有改模型、光路或损失；
新增 handoff 入口仅修复跨机器配置和补上不训练的评估入口。

## 维护者重打包

提交源码后，在干净 Git 工作树运行（asset-root 指已有数据及训练产物的仓库根目录）：

```bash
python -m LightGenV2.tasks.t08_abo_image_text_retrieval.build_lab_package --asset-root /DATA/DATA1/guest3/2026OpticsMoE --output LightGenV2/tasks/t08_abo_image_text_retrieval/releases/abo_handoff.zip
```

包内逐文件 SHA256 在 PACKAGE_MANIFEST.json，包外另有 ZIP SHA256。源码包括历史兼容
模块，以覆盖传递导入和配置继承；不代表其他任务的原始数据也已打包。
