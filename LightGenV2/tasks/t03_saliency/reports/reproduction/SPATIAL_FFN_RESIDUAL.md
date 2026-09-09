# 现有电子残差内部的轻量空间改造

## 结构边界

只修改两个现有电子残差内部的通道MLP；不增加第三分支，不扩大解码头，不引入attention或完整Transformer。
此前3×3→5×5和GRN没有提升。新假设是在通道展开之后混合邻域，比仅入口扩大感受野更适合密集预测。

```text
现有输入token混合（192通道DW3×3等）保持不变
现有LN → Linear192→384 → [新增DW3×3] → 现有GELU/Dropout → Linear384→192 → 原残差相加
```

新增DW卷积逐通道，不混合不同样本；单层384×9=3456，两层共6912个参数，额外MAC约135万/图。
原头仍85412参数。宽度、层数、光学Router Top2、alpha≥0.4、同尺度融合、478有效ROI、
224专家、10cm传播与20%–30%直流分量均不改变，pixel位移仍为0。

Qwen token顺序是2×2块优先，不能直接reshape成普通14×14网格；新增卷积先还原空间坐标，
再卷积并恢复原顺序。配置固定SALICON的196个无padding token，不宣称可直接用于视频/变长输入。

参考[MobileNetV2, CVPR2018](https://arxiv.org/abs/1801.04381)的展开空间深度卷积和
[SegFormer, NeurIPS2021](https://arxiv.org/abs/2105.15203)的Mix-FFN局部空间混合。
这里只独立实现Linear-DWConv-GELU-Linear部分，不引入两篇论文的完整网络。
与原方法差异包括：保持原有2倍展开、无新bias、恒等初始化、可选dilation2、Qwen块序还原。
并非复制SegFormer的Transformer/attention；论文不保证本任务一定获益。

## 三组对照及训练

- `control`：原结构，同本轮训练日程。
- `d1`：两个电子MLP内新增普通DW3×3。
- `d2`：同样参数量的DW3×3，dilation=2，覆盖5×5区域（非25个采样点）。
- 新增核中心为1、其余0；加载同一0.85812016源，严格保留所有已有张量及alpha。
  源SHA256=`36333e6ba013cac3a2801bce4babcc2fb5cd36adf02969e019437b40ceebaffd`。
- 最多40轮，前5轮冻结原electronic组，新DW（有则训练）、相位/Router/读出/头可更新；之后联合训练，31轮起精修。
- LR：原电子1e-5、新DW1e-4、相位5e-5、Router1e-5、CCD读出1e-5、头2e-5。
  新DW无weight decay，旧组规则不变。不同组率为了让新核学动，属于明确的训练策略。
- KD固定0.6，GT损失不变，无增强，EMA=.995；batch32/test48/workers2。
- train10000/public-test5000，不设独立validation。起点、epoch1、每2轮、末轮测试。
  从12轮起连续4次没有至少0.0001提升则LR再减半；两次后仍平台则停止。
  最高CC保留best（含起点），只保存best/last；指标选模/调速/早停使用public test，有选择偏差。

## 命令与证据

仓库根目录执行，GPU编号仅示例，先检查空闲容量：

```bash
conda activate xml
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
TASK=LightGenV2/tasks/t03_saliency
python -m pytest "$TASK/tests" -q
CUDA_VISIBLE_DEVICES=2 python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config "$TASK/configs/moe_alpha40_cffn_control.yaml" --phase all
CUDA_VISIBLE_DEVICES=6 python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config "$TASK/configs/moe_alpha40_cffn_d1.yaml" --phase all
CUDA_VISIBLE_DEVICES=6 python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config "$TASK/configs/moe_alpha40_cffn_d2.yaml" --phase all
```

结果固定在本任务`runs/simulation/moe_alpha40_cffn_<control|d1|d2>_seed42/`。
核对warmstart复评约0.858120；结构标签`_cffn_d1`和`_cffn_d2`不可混用。
查看`metrics/training_history.csv`的`lr_electronic_ffn_spatial`、自动调速事件、初始化迁移报告、
best完整复评及可视化。正式比较同时看原结构本轮对照、历史0.85812、Qwen0.88968。
这份文件是协议，不是已经取得提升的声明。
