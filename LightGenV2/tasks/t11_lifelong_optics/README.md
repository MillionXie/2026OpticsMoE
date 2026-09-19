# T11：固定几何的光学终身学习

本工程替换此前的线性占位后端。此前 100%、62.5%、25% 和 0% 的试验不构成本任务结果：存在单类取样、训练集自评、未训练前向、错误任务切分等问题。

## 固定的物理计算图

RGB uint8 → 固定三通道平铺成 224×224 单位功率振幅 → router 相位及角谱传播 → 12 个 CCD 窗口能量 → 在允许的专家上归一化 → 同一输入乘 sqrt(q) 后加载到专家槽位 → 专家相位 → 全场相干传播 → global 相位 → 全场相干传播 → 八类 CCD 能量归一化。

只训练相位，无电子分类头，无中间 OEO。数值仿真不等于硬件验证。`optics.py` 是已有 archived optical_reference/optics.py 的 AngularSpectrumPropagator 原样提取，保留 float64 传播核计算；`model.py` 沿用 pure_optical 的编码、相位参数化和相干计算图。

从 A 开始预留 3×4 共 12 个槽位。专家 224×224，间隔 30，边框 20，固定画布 772×1026，global 732×986。坐标、router 探测器、八类探测器在所有阶段保持不变。第一组激活四角槽位，下一组激活上下行中间四槽。不能直接加载旧 518×518 四专家模型的 global 权重；需从头训练，不能与不同几何/OEO 高分作等价比较。

## 数据合同

仅使用原始 Kather2016 八类任务，不使用 MNIST。来源：https://zenodo.org/records/53169 ，作者 Kather et al.，DOI 10.5281/zenodo.53169。作者原文的 Data usage statement 明确指定 CC BY 4.0：https://www.nature.com/articles/srep27988#Sec4 。服务器现有数据来自镜像重打包，尚未核验与原始 ZIP 的逐像素等价性；具体来源保留在 data_manifest.json，加载时验证其 CC BY 4.0 声明和 NPZ SHA256。使用现有 train/val/test 图像身份划分，禁止跨集合身份重叠；这是图像级划分，不能声称患者独立。

从训练集每类随机划出互不重叠 A/B：A 每类 218（1744 总计），B 每类 219（1752 总计）。A 原始域；B 固定 RGB 增益 [1.12,.90,1.04] 与偏移 [3,-3,1]，裁剪至 uint8。这是合成色彩/照明域，不声称生理染色模型。验证用同一 752 张独立图像的两个域，每域均包含全部八类。开发和选模不读取测试图像/标签；曾经被旧脚本取过 8 张的旧测试集不能称完全未触碰。

## 三段训练

1. A：E1–E4、router、global 可训练，其他专家禁用。
2. warmup：只让 E5–E8 各接收 1/4 功率，只更新其相位，旧专家/router/global 全部冻结。
3. B：E1–E4 冻结，E5–E8/router/global 更新；每批 16 张 B + 4 张 replay。replay 固定 256 张 A 训练图，每类 32，不读取 A 的其余数据参与更新。最后不足一批时比例有取整偏差。

每位专家是独立 Parameter；阶段切换新建 optimizer。学习率 expert/router/global = .01/.002/.002。正式配置 A 20 轮、warmup 3 轮、B 20 轮；轮数不是收敛或成功保证。A 按 A 验证准确率选模；B 按 A/B 平均验证准确率选模；warmup 固定轮数。保存各阶段 best/last checkpoint、逐轮指标、配置、命令、commit、来源 hash、划分 ID 和冻结审计。

## 运行

从仓库根目录运行，Python >=3.10，安装 requirements.txt。先运行：

```text
python -m unittest discover -s LightGenV2/tasks/t11_lifelong_optics/tests -v
python -m LightGenV2.tasks.t11_lifelong_optics --data /DATA/DATA1/guest3/demo_reproduction_data/kather2016/kather2016_fixed_split.npz --manifest /DATA/DATA1/guest3/demo_reproduction_data/kather2016/data_manifest.json --out LightGenV2/tasks/t11_lifelong_optics/runs/smoke/kather_contract --pilot
```

正式训练移除 --pilot，输出改为 runs/simulation/<唯一run_id>。默认 cuda:0；通过 CUDA_VISIBLE_DEVICES 只选择一张空闲 GPU。--pilot 使用真实全训练/验证数据，每阶段一轮，仅验证完整流程，不能作为正式性能结论。输出目录不可覆盖。

## 评价与解释

报告 A-before、A-after、B-after 的混淆矩阵、准确率、NLL，BWT = A-after − A-before。B 后对两个域分别做 all / old-only / new-only；严格清零被屏蔽专家并重新归一化功率。保存逐样本概率、路由以及专家均值/方差/首选计数。全场相干干涉意味着屏蔽消融不等价于可加的知识贡献，不把消融差值当成标准 forward transfer。

复现状态唯一入口：reports/reproduction/README.md。没有验证证据前，不承诺高准确率或正向后向迁移。

## 跨数据集协议：Kather2016 → LC25000 lung

跨数据集实验统一为二分类：Kather2016 的原始 `tumor` 为肿瘤，其余七类为非肿瘤；
LC25000 使用 `lung_aca`、`lung_scc` 作为肿瘤，`lung_n` 作为非肿瘤。两项数据都是
H&E RGB 组织病理图，任务语义一致但器官和来源不同。LC25000 使用 Zenodo 记录
https://zenodo.org/records/14998042 ，许可证为 CC BY 4.0，归档 MD5 必须为
`1b1325f690bc51fd76bb8c4958c03b06`。

该 Zenodo 归档实际只含肺组织子集，采用发布方给出的 train/val 目录并且不读取 test；原始数据
由较小图像集合增强而来，但发布文件没有患者或增强家族 ID，因此该划分不能称患者独立。
主要指标使用平衡准确率，避免 Kather 二分类验证集的类别比例影响普通准确率。

训练仍为 A、warmup、B 三段。B 阶段每批混入固定 256 张 A 训练图组成的 replay；E1–E4
冻结，E5–E8、router 和 global phase 更新。评估明确区分 `A_all`、`A_old_only` 和
`A_new_only`。warmup 期间只把 B 数据用于反向传播；A 仅以 old-only 条件诊断，禁止把
启用专家数量变化造成的数值变化解释为 A 学习提升。

```text
python -m LightGenV2.tasks.t11_lifelong_optics.prepare_kather_lc25000 \
  --kather <kather2016_fixed_split.npz> --kather-manifest <data_manifest.json> \
  --lc25000-zip <LC25000.zip> --out <prepared_dir>

python -m LightGenV2.tasks.t11_lifelong_optics.cross_dataset \
  --config LightGenV2/tasks/t11_lifelong_optics/configs/kather_lc25000.json \
  --task-a <prepared_dir>/kather2016_binary.npz \
  --task-a-manifest <prepared_dir>/kather2016_binary_manifest.json \
  --task-b <prepared_dir>/lc25000_lung_binary.npz \
  --task-b-manifest <prepared_dir>/lc25000_lung_binary_manifest.json \
  --out LightGenV2/tasks/t11_lifelong_optics/runs/simulation/<run_id>
```

固定权重复评（在同一源码和数据环境运行）：

```text
python -m LightGenV2.tasks.t11_lifelong_optics.evaluate --run <run目录> --data <NPZ> --manifest <data_manifest.json>
```

该命令重载 A/B 最佳 checkpoint，复算全部七项验证指标，逐项核对准确率和混淆矩阵，并保存 checkpoint SHA256；不重新训练。

## 探测器几何对照

`configs/kather.json` 使用槽位中心作为 router CCD 探测中心；`configs/kather_ring.json` 将 12 个中心放在输入中心半径 179.2 像素的圆周上，按四个相隔 90° 的端口为一组依次激活。两配置的其他参数相同，且每次运行内部几何始终固定。环形布局意在控制探测距离偏差，不能预先保证均衡或更高准确率。窗口越界或相互重叠时构造模型直接报错。

结果图复现：

```text
python -m LightGenV2.tasks.t11_lifelong_optics.report --runs <run1目录> <run2目录> --out LightGenV2/tasks/t11_lifelong_optics/reports/reproduction
```

绘图另需 matplotlib。历史无效 `runs/smoke/rebuilt` 与 `runs/smoke/t11_smoke` 删除被环境策略拒绝，保留但作废，不得引用其准确率。
