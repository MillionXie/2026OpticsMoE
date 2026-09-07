# ABO 强均衡光学 MoE 与冻结 Qwen：RTX 5090 D 正式对照

## 先确认任务方向

这次是**图搜文**，不是文搜图：每张 ABO test 商品图作为 query，在 100 个固定官方英文
标题中检索唯一正确标题。完整 test 有 2,400 张图；训练集 4,800 张图只用于光学 MoE，
冻结 Qwen baseline 不使用训练集。

## 结果

| 方法 | R@1 | R@5 | R@10 | MRR | 时间 | 平均功率 | 能量 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 光 Router Top-2 MoE，强均衡 | **0.7983** | **0.9538** | **0.9858** | **0.8676** | **10.138 ms/query** | **125.19 W**（组合代理） | **1.269 J/query**（组合代理） |
| 冻结 Qwen3-VL-Embedding-2B | 0.7371 | 0.9338 | 0.9604 | 0.8230 | 25.691 ms/query | 154.20 W（实测） | 3.962 J/query（实测） |

在当前合同下，光学 MoE 的 R@1 高 6.125 个百分点；模型核心时间比为 2.53×，Qwen
实测能量与光学 MoE 组合能量代理之比为 3.12×。这里没有把组合代理伪装成实验台电表
实测，具体口径见下文。

## 光学 MoE 的时间边界

强均衡 checkpoint 与性能报告来自实验室服务器完整训练，checkpoint SHA256 为
`b64689eb792da483486f721d4adf879f0a083107d7cbee63ad8ed7718c8b7144`。权重值不会改变
算子形状，因此 5090 D 计时使用同一冻结推理图的精确张量尺寸：

- 两次光 Router；
- Vision 专家层和全局层；
- Language 专家层和全局层；
- 64D 检索读出头；
- 每次物理光场固定为 `0.714 ms 传播 + 0.100 ms 相位 SLM + 0.500 ms CCD = 1.314 ms`。

计时不运行角谱 FFT 来冒充真实光路。每个 CCD 后电子分量先预热 50 次，再计时 1,000
次，共 8,000 个正式分量调用。电残差与光路并行，且 Vision/Language 残差均小于
1.314 ms，故由光路覆盖；串行 CCD 读出、融合、下一层 SLM 重建和任务头按实测中位数
加入临界路径。结果为：

- 六次物理光场：7.884 ms；
- 电子算子总工作量：3.396 ms/query，其中一部分与光路并行；
- 完整临界路径估算：**10.138 ms/query**；
- 任务头：约 0.076 ms/query；
- 其余逐分量 median/P95 见 `optical_moe/optical_moe_electronics_5090d.csv`。

该边界对应“第一个替换 block 到 64D embedding 输出”，不含文件读取、图像解码、
processor、Qwen patch embedding、模型加载，也不含预计算的 100 个标题库；与 baseline
的第一个原生 Vision Transformer block 到完整排序边界相对应。

## 功率和能量口径

光学 MoE 不是一块 5090 D 独立跑完整仿真的耗电。报告将实测/给定的两部分组合：

- 光学设备：80.388 W；只算六次物理光场为 0.634 J/query；假设设备在完整 10.138 ms
  临界路径持续上电，则为 0.815 J/query。
- 5090 D 电子分量：10 ms 采样；idle 15.92 W；稳定分量负载平均 101.28 W、峰值
  107.72 W。每个分量额外维持 2 秒稳定负载，每项约 200 个功率样本，避免 GPU 从 idle
  唤醒造成低估。
- GPU 在电子算子执行期间按对应分量功率、其余临界路径按 idle 计，得到 GPU 板卡
  0.454 J/query；与光学设备相加为 **1.269 J/query**，折合平均 **125.19 W**。
- 若极端假设 5090 D 在完整临界路径始终达到额定 575 W，再加光学设备，绝对上界为
  655.388 W、6.644 J/query。它只是保守上界，不是实测平均功率。

冻结 Qwen 是直接板卡测量：50 次预热，类别均衡 200 张计时/功率样本；CUDA mean / median /
P95 为 25.691 / 25.075 / 27.657 ms，active mean / peak 为 154.20 / 158.05 W，实测
active energy 3.962 J/query，575 W 额定上界 14.772 J/query。

## Baseline 合同

baseline 使用原始 `Qwen3-VL-Embedding-2B`：2,127,532,032 个参数全部冻结，可训练参数
为 0，没有 LoRA、额外读出头或微调。100 个标题 embedding 预先计算；在线查询执行原生
Vision/Language blocks、2048D L2 归一化、100 个余弦相似度和完整排序。性能对完整 2,400
张 test 计算。本轮复测的性能与此前正式证据逐项完全一致，延迟从 26.052 ms 变为
25.691 ms（约 -1.39%，属于同卡运行波动）。

## 文件

- `summary.json`、`comparison_summary.csv`：可直接进表格的汇总。
- `abo_5090d_comparison.png/.pdf`：性能、时间、能量三联图。
- `optical_moe/`：8,000 次分量计时、1,601 个 active 功率样本和原始 JSON/CSV。
- `qwen_baseline/`：2,400 条预测、200 条计时、原始功率样本和总报告。
- `evidence_manifest.json`：关键文件 SHA256。

重新绘图：

```powershell
python LightGenV2\tasks\t08_abo_image_text_retrieval\reports\5090d_optical_and_qwen_20260907\plot.py
```

重新测光学 MoE 电子边界与功率（必须独占 RTX 5090 D）：

```bash
python -m LightGenV2.scripts.profile_optical_electronics_5090d \
  --tasks t08 --warmup 50 --repeats 1000 --measure-power \
  --idle-sample-seconds 5 --power-dwell-seconds 2 \
  --output-dir /root/autodl-tmp/abo_t08_optical_profile_5090d
```

冻结 Qwen 的精确复测命令保存在 `qwen_baseline/command.txt`。
