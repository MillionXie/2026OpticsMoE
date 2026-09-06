# 冻结 Qwen baseline：RTX 5090 D 待测合同

本文件是交给后续实验 AI 的强制口径；当前共享服务器不得填写结果。

## 模型与性能

- 模型：`Qwen/Qwen3-VL-2B-Instruct`，权重冻结，不微调。
- 输入是同一 OpenMoji 图像和完整文本指令，使用同一个 5,000/1,000 split contract。
- 输出必须解析为 `6×6` 类别网格和编辑网格；报告 changed-cell accuracy、foreground category accuracy、edit-grid IoU、object F1 和 scene exact match。
- 若模型自由文本不能稳定解析，必须单独报告 parse-failure rate，不得删掉失败样本。

## 速度

- GPU：RTX 5090 D；记录驱动、CUDA、PyTorch、精度、功耗上限和 batch size。
- 计时起点：图像/文本 hidden state 即将进入第一个原生 Transformer block。
- 计时终点：可解析的 `6×6` 语义/编辑结果输出；包含全部 Transformer blocks 和生成/读出过程。
- 不计文件读取、PNG 解码、tokenizer 和 block 之前的 embedding。
- batch=1，模型只加载一次，完整 test 连续运行；不显式 warm-up，第一条 test 也进入
  统计。CUDA Event 与同步 host 计时均保留；因终点包含 CPU JSON 解析，论文速度采用
  host mean，并同时报告 CUDA mean、median、P5、P95。

## 功耗

- 与速度同一次测试，用 NVML 至少 20 Hz 采样整卡功率。
- 报告 idle、mean、peak，以及扣除 idle 后的 `J/sample`；不能使用 TDP 代替实测值。

待填字段：`performance=null`、`speed_ms=null`、`power_w=null`，完成 5090D 实测后再替换。
