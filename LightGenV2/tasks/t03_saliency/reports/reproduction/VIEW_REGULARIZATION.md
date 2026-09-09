# SALICON 泛化优化：同步弱增强与早期重新适应

目标是完整5000张public-test平均CC达到0.87；目标不是结果承诺。
前一轮cffn三组均34轮早停，更新后最高CC为0.857841/0.857759/0.857757，
均保留epoch0的0.858120源。训练CC约0.881而测试下降，提示该续训方案泛化退步。

## 本轮不变的边界

- 冻结Qwen patch前端，不执行原生Transformer/attention；两级光电同尺度融合。
- 光router四专家Top2；alpha至少0.4；20%–30%随机相干未调制分量；pixel位移0。
- 原478有效面积、224专家、17微米、10cm传播不改；解码头85412参数不变。
- control是原电子残差；cffn只在现有两残差内部加6912个DW3×3参数，无新分支。
- 单图单次原始输入推理；不加测试增强、集成或使用真值的测试后处理。

## 数据和训练

train2014=10000，val2014=5000作为public test，无独立validation。
epoch0、1、每5轮、末轮完整测试，按CC选best，自动调速也使用public test，存在选择偏差。
只保留best/last及日志，不产生每5轮权重。测试集不进入增强、teacher缓存或反向传播。

三组共同从`moe_alpha40_refine_weakaug_seed42/best_checkpoint.pt`开始（CC约0.85468765），
这比反复无增强蒸馏续训的0.85812来源早，但并非从头训练/全新数据。
源SHA256：`de477b8c13c46c50cb9f17eb0b62bc577aaefef5512887e1e17a86d0affb5eea`。
不重置alpha，所有已有张量严格迁移，新卷积恒等初始化。

|配置后缀|电子结构|KD权重|
|---|---|---|
|viewreg_control|原结构|固定0.6|
|viewreg_cffn|增加展开空间DW3×3|固定0.6|
|viewreg_cffn_kd2|同上|2.0线性降至0.6，60轮到达|

增强在已统一224×224的训练图上执行：边长保留95%–100%的随机裁剪、缩回224、
50%水平翻转、亮度/对比度各±5%。图像、GT密度、fixation、teacher用同一空间变换。
密度双线性缩放后重新归一化为和1；fixation最近邻缩放，裁掉全部fixation时回退整幅图。
教师必须先softmax成概率密度，再变换、归一化、取log返回KD接口（温度固定1）。
**这是近似视图一致性正则化，不是教师重新推理增强图；裁剪后注视与光度不变性只是训练假设。**
不使用MixUp：现有NSS把fixation二值化，直接混合会丢失混合权重，本轮避免改动损失语义。

教师仍是同头Qwen：SHA `531c4a330fc05e2c27b037784588716d248a29d1ab8da951d443165dcce552f8`。
缓存`runs/simulation/generalize_alpha40_20260909/teacher_train_logits.pt`必须精确包含排序后10000个train ID；
代码校验身份/尺寸/来源，每个run记录缓存SHA及变换说明，推理不运行教师。

最多80轮，前5轮冻结原电子组，6–60联合训练，61–80关闭增强精修；EMA=.995。
初始LR：电子3e-5、新DW2e-4、相位5e-4、router5e-5、CCD读出3e-5、头5e-5。
weight decay=.03，相位/router/新DW为0；GT KL1+CC1.5+SIM.25−NSS.1。
从30轮开始每6次测试无至少.0001改善则额外LR减半，最多两次，之后再平台则早停。
三组增强种子、样本顺序、batch32/test48一致；训练CC是在增强图上，不能直接当作原图泛化差距。

## 操作命令

先通过Git拉取包含本文件的已发布commit；依赖同之前xml环境（Torch2.6.0+cu124、Transformers4.57.3）。
在仓库根目录，检查GPU容量后逐项启动；并行使用不同终端/受控调度，不重复同一run目录。

```bash
conda activate xml
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python -m pytest LightGenV2/tasks/t03_saliency/tests -q
TASK=LightGenV2/tasks/t03_saliency
CUDA_VISIBLE_DEVICES=6 python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config "$TASK/configs/moe_alpha40_viewreg_control.yaml" --phase all
CUDA_VISIBLE_DEVICES=6 python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config "$TASK/configs/moe_alpha40_viewreg_cffn.yaml" --phase all
CUDA_VISIBLE_DEVICES=3 python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --config "$TASK/configs/moe_alpha40_viewreg_cffn_kd2.yaml" --phase all
```

产物：本任务`runs/simulation/moe_alpha40_viewreg_<control|cffn|cffn_kd2>_seed42/`。
查看run_manifest的commit/命令、resolved_config、teacher_cache_provenance、初始化SHA、
metrics/training_history.csv（augmentation_active/KD/LR/测试曲线）、training_report以及selected_checkpoint_test_evaluation。
最佳可视化在best_visualization。比较本轮更新后的best、历史0.85812以及Qwen0.88968，
不得把warmstart保留下来的分数标为本轮提升。获得改善后仍需完整权重复评和光router/alpha审计。
