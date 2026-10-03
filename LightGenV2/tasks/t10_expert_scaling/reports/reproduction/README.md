# T10 复现入口

当前已进入seed17验证校准训练，无最终test成绩。完整专家数/Top-k矩阵尚未启动。

## 当前运行

- 服务器Git工作树：`/DATA/DATA1/guest3/t10_scaling_20260917`，训练源码`752d2995`。
- run：`runs/simulation/kather_calibration_s17_17um_uuid_20260917`。
- GPU：物理0、1、6，分别为RTX4090、RTX4090、A100；使用UUID绑定，最多同时3个训练子进程。
- coordinator PID：490723；首组子进程PID：490816、490822、490954。PID只代表启动时身份，以run状态为准。
- 12配置：N=9、k=9；MoE+OEO、总相位参数匹配D2NN+OEO、等孔径D2NN+OEO；L=4/6、lr=.001/.002。
- 60epoch，seed17，有效batch16、microbatch2，梯度累积8次；宏类权重仅来自train。
  路由均衡项按每个microbatch的平均概率计算，所有本批运行固定microbatch2，不能无说明改变它。
- train3496 / val752，原Kather固定split；不读取test。该缓存来自登记的HF镜像，
  与Zenodo原ZIP逐像素等价尚未验证，且患者独立性未验证，不将这些写成已通过。
- 每epoch依据EMA validation macro-NLL保存best，训练所有相位，OEO无可训练电子残差。

```bash
python -u -m LightGenV2.tasks.t10_expert_scaling.schedule \
  --data /DATA/DATA1/guest3/demo_reproduction_data/kather2016/kather2016_fixed_split.npz \
  --out LightGenV2/tasks/t10_expert_scaling/runs/simulation/kather_calibration_s17_17um_uuid_20260917 \
  --gpus 0 1 6
```

依赖：Python3.11、PyTorch2.6.0+cu124、NumPy；数据获取另用Pillow、gdown、curl。
每个run的metadata保存数据/cache与manifest SHA256、GPU身份、命令、实际配置、源码commit；
只有best/last两个checkpoint。队列正常完成或收到终止信号时等待子进程退出，写`release_check.json`。

## 已完成的检查

- `runs/smoke/physics_n4_v2_full`：三架构6层分类损失反传通过，Top-1 Router梯度非零；
  ASM 2倍/4倍补零的相对光场误差0.0011902、强度误差0.0009036。
- `runs/smoke/physics_n49_v2_full`：49专家及两个D2NN6层反传通过；对应误差0.0001822、0.0000740。
- `runs/smoke/train_n4_k1_v2`：32train/32val流程检查完成6epoch，覆盖dense预热到hard Top-1切换；
  不将小样本准确率当成实验结果。上述冒烟源码`f2be649a`。
- 原`kather_calibration_s17_17um_20260917`短跑为修复CUDA序号映射而终止，记录保留；
  三个旧子进程退出后GPU进程表为空，再启动UUID版本，没有同时运行六张卡。

## 下一阶段

先检查校准曲线、相位更新与验证结果，锁定共同深度/各模型学习率，之后生成并运行N/k矩阵。
DeepWeeds已固定作者revision `da084f62bcd1a2b0afeb8b81f1a27be3186391a1`及官方第1折（文件索引0）；
标签已通过作者API获取，图像下载和场景重复审计尚未完成，不能提前声称该任务开始训练。
数据准备在独立工作树`/DATA/DATA1/guest3/t10_data_20260917`执行，避免改动正在训练的源码。

- [任务与当前状态](../../README.md)
- [实验协议](../../docs/PROTOCOL.md)
- [数据来源与许可](../../dataset/registry.json)
- [设计参数](../../configs/study.json)
- [自动生成的矩阵与校验摘要](../design/summary.json)

从仓库根目录执行：

```bash
python LightGenV2/tasks/t10_expert_scaling/plan.py
```

只需Python标准库；输出为设计表，不读取数据、不加载checkpoint、不使用GPU。
已核查数据集公开许可；Kather缓存已核验，其他任务原始数据审计尚未完成，登记中的待定SHA256保持null。
不能把几何自检通过写成新模型复现成功。未来结果必须在此登记run ID、配置、选模与证据链接。

此前病理代码结构的核查依据（不是本任务实现）：

- `LightGenV2/demo_check/adrenal_softsign_code_export_20260915_145336/code/optical_reference/model.py`：
  `between_expert_stages`与`global_fcs[-1]`明确包含中间及末端global phase。
- 同目录`models.py::RelayFanoutMoE.global_fanout_convolution`：理想图像复制及振幅L2归一化。
- 同目录`optical_reference/prompt.py::_topk_routing`：旧Top-k梯度及概率/功率口径。
- `LightGenV2/tasks/t08_abo_image_text_retrieval/reports/optical_router_moe_20260907/main/config.yaml`：
  正式任务224²专家、pitch254的尺寸来源。
