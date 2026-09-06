# 冻结 Qwen baseline：RTX 5090 D 待测合同

本文件是交给后续实验 AI 的强制口径；当前共享服务器不得填写结果。

## 模型与性能

- 模型：`Qwen/Qwen3-VL-Embedding-2B`，权重冻结，不微调。
- 输入与主方法使用同一份 `split_contract.json`。
- 以 Qwen 最后一层视觉空间 token 接同一个轻量显著性 decoder；只训练 decoder，报告 CC、KLD、SIM、NSS、AUC-Judd、MAE。
- checkpoint 仍按每 5 epoch 的 public-test CC 选择；报告必须标注 test-used-for-selection、no-validation。

## 速度

- GPU：RTX 5090 D；记录驱动、CUDA、PyTorch、精度、功耗上限和 batch size。
- 计时起点：输入进入第一个原生 Vision Transformer block 之前。
- 计时终点：`224×224` 显著性图已经输出之后。
- 不计文件读取、JPEG 解码和 block 之前的 patch/token embedding；不得只计单个 block。
- batch=1，模型只加载一次，完整 test 连续运行；不显式 warm-up，第一条 test 也进入
  统计。用 CUDA Event + `torch.cuda.synchronize()` 报告 mean、median、P5、P95
  （ms/样本）。

## 功耗

- 与速度同一次推理测试，NVML 至少 20 Hz 采样整卡功率。
- 报告空闲功率、推理平均功率、峰值功率，以及扣除空闲后的每样本能量 `J/sample`。
- 不能用 TDP 代替实测功率，也不能把主方法的光学 9.084 ms 当作 GPU 功耗结果。

待填字段：`performance=null`、`speed_ms=null`、`power_w=null`，完成 5090D 实测后再替换。
