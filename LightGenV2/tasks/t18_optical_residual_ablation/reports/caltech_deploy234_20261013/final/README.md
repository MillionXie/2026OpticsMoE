# 主干2/3/4层可部署合同实验：最终交付（2026-10-13）

六组各完整150轮已结束，独立全六CPU审核后排他锁定，仅按首次最低验证EMA balanced NLL选择，每组获选EMA首次单次test。六次test及CPU/GPU父子均实际退出；项目GPU已释放，全程未使用A100。原数据、所有旧run/best/last/测试和封存ABO均保留。

**4层rho.3最终test总体准确率46.4147%，75%验收目标未达到。** 不因test成绩换权重、重训、换类别或追加预算。本轮是有负结果的完整实验交付，不能说精度目标已经实现。

|主干|rho|获选epoch|train|val|test总体|test宏召回|val bNLL|test bNLL|train-val差|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|2|0.0|147|30.90%|30.39%|27.86%|29.71%|2.349547|2.395744|0.52pp|
|2|0.3|150|36.22%|36.04%|53.68%|36.56%|2.412189|2.330161|0.18pp|
|3|0.0|148|32.65%|30.04%|26.31%|29.91%|2.386002|2.420238|2.62pp|
|3|0.3|150|47.81%|35.69%|7.41%|40.51%|2.037389|2.107678|12.12pp|
|4|0.0|147|52.84%|48.76%|50.19%|43.33%|2.010080|2.095928|4.08pp|
|4|0.3|150|57.22%|46.64%|46.41%|42.81%|1.869109|1.983065|10.57pp|

## 精度与过拟合解释

4层残差test总体比无残差低3.7791pp、宏召回低0.5270pp，但balanced NLL更低（1.983065对2.095928）。残差train/val差距10.57pp，对照4.08pp；3层残差差距12.12pp，泛化差距和末段平台迹象仍在。2层训练也低（30.90/36.22%），不能把所有低精度归因为过拟合。后期val NLL仍缓慢下降，不能称充分收敛或证明容量上限。当前更强MixUp/早停等没有保证高精度或小差距。

测试支持极不均衡：airplanes680、Motorbikes678、Faces315合计1673/2064（81.0562%），而这三类在val只占60/283。3层残差在这三个大类的test召回均为0；因此总体7.4128%，而20类宏召回40.5068%。2层残差总体53.6822%也不能当作所有类别同样有效，其宏召回36.5647%。支持分布和各类别召回必须一起看，不能仅由总体train/test差断言全部是严重过拟合。

[类别支持/召回/正确数](test_support_breakdown.csv)、[前三大类与其余17类贡献](test_support_groups.csv)、[全混淆矩阵CSV](full_confusion.csv)、[三划分support与召回](per_class_support.csv)。这些是保存结果的CPU计算，无新模型推理、无test调参。类用验证筛选、旧test已观察过，因此本结果为探索性开发；没有单seed误差条，不能与旧9专家/旧编码结果作单因素正则或深度因果结论。

## 真实曲线及审核

[完整train/val与单test点PNG](learning_curves.png)、[SVG](learning_curves.svg)、[PDF](learning_curves.pdf)、[曲线CSV](curve_data.csv)；train为每5轮及端点的unaugmented评估，val每轮，test星形点仅在各获选epoch。
[汇总PNG](selected_summary.png)、[SVG](selected_summary.svg)、[PDF](selected_summary.pdf)、[CSV](selected_summary.csv)。
[2层test CM](L2_test_confusion.pdf)、[3层test CM](L3_test_confusion.pdf)、[4层test CM](L4_test_confusion.pdf)，全部按true-class行归一化，并保留PNG/SVG。

[最终CPU审核](audit.json)、[全六拟合审核](cpu_fit_audit.json)、[全局锁](global_selection_lock.json)、[pair锁](pair_locks.json)、[选择锁](selection_locks.json)、[各test claim与源/部署收据](server_audit_receipt.json)、[实际进程退出/GPU释放/部署校验](release_receipt.json)。
全局锁SHA d45a5f9ffdbc1024843bed7b901aec3ebfc1a1b0d5b1b312d884701258689f2d；all6/each-test1/test_used_for_selection=false。训练7源当前及原4406c010f95f60a959d732ad2200eaa34657519a Gitblob逐项一致。CPU最终审核只读预测和保存结果，model_forward_calls=0。

## 后续光路部署身份与顺序

布局参考封存ABO rank72：2×2四个224专家、gap30、active478、数值canvas518，532nm、10cm、逻辑17um。主干2=E/G、3=E/G/E、4=E/G/E/G，router另算，实际3/4/5次顺序捕获；四专家在同一个相位面并行。没有CNN/Linear、预训练或可训练电子分类头。

原RGB uint8固定luminance、bilinear224、sqrt幅度、RMS.25；每次相位之前保零tanh(abs/.5)。Router捕获给Top2并L2归一路由权重，四专家始终按冻结布局送同输入的加权副本，之后复用同一路由权重。每层原始CCD为非负线性478场，中间计算为relative mean/clamp12/log1p/pool224/rowLayerNorm/ReLU/RMS.25后同一有界幅度编码。最终固定20个等大64×64窗口能量归一化分类，窗口与类顺序冻结。

rho只在expert/global：M=(1-rho)exp(i phi)+rho。**必须联合amplitude×abs(M)和angle(M)，只加载相位BMP不等价。** 模拟/验证/导出均8bit幅度round255、相位floor256，不允许逐BMP额外峰值拉亮。原生黑padding是需要的，应看active与illuminated支持域的mean/p99/max/zero/saturation，不用整张画布均值作为拉亮依据。

实际服务器run为`LightGenV2/tasks/t18_optical_residual_ablation/runs/simulation/caltech_selected20_s17_mainL2L3L4_ABO_bmp8_reg150_mem2_3gpu_20261013`。各获选PT在`<run>/<key>/best_checkpoint.pt`，静态部署资产在`<run>/final/<key>/deployment/`：

- `phase_<stage>_canonical.bmp`真实478×478；`phase_<stage>_native.bmp`真实1920×1200，标准场到1016再居中。
- `amplitude_multipliers.npz`各阶段abs(M)、`fixed_masks.npz`冻结router/20探测mask、`classes.json`固定类映射。
- `manifest.json`记录选中PT/config/7训练源/导出源、文件SHA和校准状态。全六包文件SHA、标准/设备尺寸、mask/类映射和PT身份已核验。
- 动态幅度：使用主线`export_caltech_deploy234.save_sample_bmps`，调用冻结`model.physical_fields`，得到478标准幅度与1920×1080设备幅度，并记录亮度统计。中间幅度来自实际raw CCD的同一非线性处理；不能预先用仿真CCD替代实拍，也不能额外峰值归一化。

[全部部署manifest及类/support/PT身份索引](deployment_index.json)。PT、BMP和NPZ保存在受保护服务器run，不放入Git；本报告只发布清单/SHA与源码引用。六模型离线BMP数学回放、10项Linux CPU独占claim/联合编码检查通过；已训练4层static导出也通过。**离线合同通过不代表已经实拍验收。** 未连接实验室。物理相位/幅度LUT、相机ROI/方向/曝光/gain、逻辑/设备非整数采样以及相干rho比例/相对相位需要新的校准会话；不能静默复用旧CCD或旧校准。当前精度验收失败，不能把这些权重描述为达到75%的部署模型。

固定数据train1372/val283/test2064，20类3719图全保留；cacheSHA a81e78bbff06ea65e7ceba3771b688178276464e33fdcdedf3796fa2930fa6e1，manifestSHA91f7aa36a97713e300699367904c8866551996de199b9630c0545bb24dc5f60e。原数据/许可/作者与筛选记录见[固定数据报告](../../caltech_selected20_20261012/README.md)，不重新在线宣称API已恢复可用。

五份最终PDF已用Poppler实际渲染并逐页视觉检查，曲线、75%目标线、全部20类CM标签和色条清晰，无裁切或重叠；[视检记录](visual_inspection.json)。[交付文件SHA清单](artifact_manifest.json)。
