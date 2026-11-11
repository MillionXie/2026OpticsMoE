# T22：时间一致性公开替代模型试验

2026-11-11 用户批准独立试跑。原论文 T06/T13 模型、权重、数据和结果不改。本目录由本对话负责；不修改图表整理对话负责的 E 盘归档或其他任务。

这是单独训练、单独评价的替代模型，不冒充论文原实现。目标为原仿真 SRCC 0.8043868643 的绝对差不超过 0.02。当前状态：实现和预检中；尚无正式训练结果。

## 已批准的架构

输入保留 16 视频/场、4 帧/视频、49 个视觉 token/帧、固定 Temporal prompt。仅使用冻结 Qwen patch+position 特征与固定 prompt；删除 quality14、prompt 到视觉的额外调制、四条电子卷积旁路及 RMS 融合。用户进一步指定所有原 Softplus 位置改 ReLU。

串行链路：Qwen front -> LN1024/Linear1024→192 -> 帧 MLP router/Top2 experts -> 帧 optical global -> Flatten49×192/LN9408/Linear9408→192/GELU -> 四帧 token 与 38 个 prompt token 拼接 -> 视频 MLP router/Top2 experts -> 视频 optical global -> Flatten8064/Linear8064→1024/GELU/Linear1024→1 -> MOS 尺度还原。

- Router：LN192、Flatten、Linear→128、GELU、Linear128→4、softmax(T=1.1)、Top2、选中权重 L1 归一化。帧 router 读49 token，视频 router 只读4帧 token。
- 每级4个专家：LN192、Linear192→384、GELU、Linear384→192；按Top2加权。专家在视频/帧/物理槽之间共享，同一级4专家互不共享，帧级与视频级不共享。当前为易核对实现，实际执行4个MLP，仅2个贡献输出，不宣称稀疏执行速度。
- 两次 global 保持原17μm逻辑光路：532nm、10cm、518整场/478有效区、0.5°截止、16视频布局、111×111每视频相位、评测相干漏光系数0.2。不是改变传播网格的8μm/v2结果。
- 帧 global 编码：Linear192→27/ReLU；token轴Linear49→27/ReLU；原RMS幅值归一化；原整场传播、探测、区域读取；强度除均值/clip8/log1p；pool49×96/LN96/ReLU/Linear96→192。
- 视频 global 编码：Linear192→56/ReLU；序列补零至56行；原RMS归一化、整场传播、探测及读取；pool42×96/LN96/ReLU/Linear96→192。
- 末端没有时序卷积、均值/方差/差分统计、独立文本统计头或额外旁路。前端固定空间降采样和global物理归一化仍明确保留。

`model.py` 独立声明全部前向层，不 import 原 T06/T13/experiments，不带旧 router/expert 相位实现。`front.py` 从本机已有 safetensors 定向抽取实际需要的 patch/position 参数，固定 prompt 直接保存 embedding；不实例化完整 Qwen。不把2B权重或词表打入包。

## 数据与评价

沿用原2250 TRAIN / 558 TEST，无独立VAL；TEST只用于周期评价和选模，不参与梯度，结果注明开发评价。每视频一标签；padding仍参与整场传播但从损失/指标剔除。训练可使用已核验的冻结前端缓存，不读取quality14作为输入。初次试验不蒸馏、不读取原模型TEST预测或MOS进入训练损失。

只保存 best/last；配置、Git源版本/文件SHA、环境、逐样本预测、SRCC/KRCC/PLCC/RMSE/MAE、路由统计和状态写入独立run。两级global均保留整幅相干传播，不把逐视频独立传播拼接冒充多视频光场。

## 独立权重

`LightGen.pt` 为结构配置 + state_dict + 固定prompt/位置buffer + MOS尺度。含提取后的冻结视觉前端，可对新的RGB输入推理；不需要完整Qwen目录/模型/Tokenizer。固定prompt限定此Temporal任务。原Qwen预训练参数的来源及必要归属保留。

当前工作仅为内部试跑。公开远端仓库存在，因此不自动将本轮源代码或权重push公开。服务器仅从带提交身份的私有Git包取出本任务路径，记录发布状态；不更新/覆盖其已有脏工作目录或修改其他窗口文件。

## 检查与运行

`python -m unittest LightGenV2.tasks.t22_public_temporal_mlp.tests.test_contract -v` 检查串行拓扑、精确Top2、全零ReLU场的有限梯度、两级相位梯度、独立PT重载。

首次导出前端：`python front.py --qwen-dir <原本机快照目录> --language-cache <固定prompt缓存> --output assets/front.pt`。只定向读取小组件，不构建完整Qwen。`paths.local.json` 指定 vision_cache、language_cache、manifest、front，均不提交Git。

试跑：`python run.py --paths paths.local.json --out runs/simulation/relu_s163_20261111 --device cuda`。必须设置单个CUDA_VISIBLE_DEVICES，并在入场前、缓存核对后两次检查GPU为空闲；不抢占、不终止其他任务。

`wait_for_gpu.py` 是本次试验的有界单任务等待器，只候选空闲RTX4090、连续三次检查后启动一个训练任务；超时只记录未开始，不自动换占用GPU或转A100。它不是无限轮询或后台超参数搜索。

正式独立推理：`python infer.py --weights <LightGen.pt> --videos <video1.mp4> <video2.mp4> --output result.json --device cuda`。新视频按输入顺序组成16槽位，最后一组复用已有视频padding但只输出有效条目。槽位敏感性另由每次正式run的16次循环换位审计记录。
