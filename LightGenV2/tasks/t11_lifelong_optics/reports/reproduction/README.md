# 光学终身学习：初步结果与复现入口

日期：2026-09-19。目的：供课题组初步讨论，不作为论文最终性能结论。

## 可以向老师汇报的结果

已经完成真实相干光学 MoE 的固定几何 4→8 顺序学习。等半径 router 探测区版本在单种子验证集上观察到 A 的正向后向迁移：43.88% → 47.07%（+3.19 个百分点），新域 B 为 56.78%。冻结审计和重载 checkpoint 复评通过。尚不能说明多种子/独立测试集上稳定成立。

| Router 布局 | A 学完时 | B 学完后 A | B 学完后 B | BWT |
|---|---:|---:|---:|---:|
| 槽位中心探测区 | 45.74% | 44.02% | 45.88% | -1.73 pp |
| 等半径探测区 | 43.88% | 47.07% | 56.78% | +3.19 pp |

![训练曲线、阶段准确率和路由功率](initial_results.png)

## 专家屏蔽：等半径版本

| 验证域 | 全部 E1–E8 | 仅旧 E1–E4 | 仅新 E5–E8 |
|---|---:|---:|---:|
| A | 47.07% | 37.37% | 12.63% |
| B | 56.78% | 37.23% | 14.49% |

A 屏蔽新专家后下降 9.71 个百分点。最终旧专家功率占比：A 78.55%、B 78.05%；新专家分别为 21.45%、21.95%。旧专家相位在 B 阶段没有改变。

解释边界：屏蔽时重新归一化输入功率，改变了全场相干干涉；router/global 也在 adaptation 中更新。因此这是一项系统级消融，不能单独证明可加的“知识贡献”，更不能把 9.71 pp 当成 BWT。BWT 是 3.19 pp。第一版新专家占约 99% 功率；第二版只改 router 检测几何，支持继续检查几何偏置。两种子相同不是多种子验证。

## 实际架构及训练合同

- 固定 12 槽，224×224 独立专家相位；固定 772×1026 画布和 732×986 global 相位。
- RGB uint8 三通道平铺 → 单位功率振幅 → optical router → CCD 能量归一化 → sqrt(q) 专家输入 → 专家相位/相干传播 → global 相位/相干传播 → 八类 CCD。
- 没有线性替代模型、电子分类头或中间 OEO。使用数值角谱传播，不是实验台测量。
- A 20 轮；仅新专家等功率 warm-up 3 轮；B + replay 20 轮。expert/router/global 学习率 .01/.002/.002。
- replay 256 张 A 训练图，每类 32；B 批次 16 新域 + 4 replay，最后短批次有取整偏差。
- A 按 A 验证准确率选 checkpoint；B 按 A/B 平均验证准确率选 checkpoint；warm-up 使用最后一轮。没有使用测试准确率选模。

## 数据与限制

服务器 Kather2016：3496 train、752 validation、752 test，八类。逐类将训练集分为 A 1744 张（每类218）、B 1752 张（每类219），原图身份互不重叠。A 为原始域，B 为预先固定的 RGB 增益/偏移合成域；两个验证域使用同一组独立 752 张图、每类94张，便于配对比较。新训练/复评不读取测试图像或标签。

来源：[原始数据](https://zenodo.org/records/53169)，[作者 Data usage statement 明确 CC BY 4.0](https://www.nature.com/articles/srep27988#Sec4)。服务器缓存来自镜像重打包，未核验与原始 ZIP 逐像素等价；图像级划分未证实患者独立。数据 SHA256：`b6d887d8dfc890f5829639b47dd63e7fe22d1fee100119617a36756e32631ce7`。

这是验证集选模后的初步结果，不能当作独立测试性能。当前未做多种子、无 replay、固定容量或 frozen-global 对照，不能把提升全部归因于专家扩展。相较其他已有 Kather 高分，本实验的几何、光学层数/OEO设置和每阶段训练量可能不同，尚未建立可比基线。

## 运行身份与证据

服务器工作目录：`/DATA/DATA1/guest3/t11_optical_20260919`。Python 3.11.15，PyTorch 2.6.0+cu124，NumPy 1.26.4；单张 A100（CUDA_VISIBLE_DEVICES=6）。训练与复评进程已结束，显存已释放。

### kather_s17_f7c6fbba

训练 commit：`f7c6fbba91aa2041c1f41b0b5ccbf28fb65badfb`。

- A 最佳轮次：13；权重 SHA256：`a4678e32b64803796c31755e78ab382b7423eeeac2cba867e492e3584da8870a`。
- B 最佳轮次：18；权重 SHA256：`9524eae3f79ba81c20d089717822b704abe9d1e2e3c245b8505f9c4163a3e473`。

所有指标对应 `runs/simulation/kather_s17_f7c6fbba` 下的 metrics.json、history.json、audit.json、split.json、metadata.json、reevaluation.json；逐样本概率与路由为同目录 NPZ。A/B 最佳权重已下载到本地同名 run，并校验 SHA256；服务器保留各阶段 best/last。
### kather_ring_s17_7c1a07e9

训练 commit：`7c1a07e90f20b7e02d1836d2e90d03cae78eddf3`。

- A 最佳轮次：19；权重 SHA256：`be94f08a1a66407092125aaa7796273d9cad4fa10c4aedf57a080038d2b49495`。
- B 最佳轮次：12；权重 SHA256：`6d2558de79a17c3f0a4e3936fd7a6213d30779df48a1ea1b83d409fc4f86d90c`。

所有指标对应 `runs/simulation/kather_ring_s17_7c1a07e9` 下的 metrics.json、history.json、audit.json、split.json、metadata.json、reevaluation.json；逐样本概率与路由为同目录 NPZ。A/B 最佳权重已下载到本地同名 run，并校验 SHA256；服务器保留各阶段 best/last。

## 完整复现命令

在服务器仓库根目录执行，先 checkout 相应的训练 commit。输出名必须新建，不能覆盖已有 run。

```bash
CUDA_VISIBLE_DEVICES=6 /home/guest3/miniconda3/envs/xml/bin/python -m unittest discover -s LightGenV2/tasks/t11_lifelong_optics/tests -v

CUDA_VISIBLE_DEVICES=6 /home/guest3/miniconda3/envs/xml/bin/python -m LightGenV2.tasks.t11_lifelong_optics --config LightGenV2/tasks/t11_lifelong_optics/configs/kather_ring.json --data /DATA/DATA1/guest3/demo_reproduction_data/kather2016/kather2016_fixed_split.npz --manifest /DATA/DATA1/guest3/demo_reproduction_data/kather2016/data_manifest.json --out LightGenV2/tasks/t11_lifelong_optics/runs/simulation/kather_ring_reproduction

CUDA_VISIBLE_DEVICES=6 /home/guest3/miniconda3/envs/xml/bin/python -m LightGenV2.tasks.t11_lifelong_optics.evaluate --run LightGenV2/tasks/t11_lifelong_optics/runs/simulation/kather_ring_s17_7c1a07e9 --data /DATA/DATA1/guest3/demo_reproduction_data/kather2016/kather2016_fixed_split.npz --manifest /DATA/DATA1/guest3/demo_reproduction_data/kather2016/data_manifest.json
```

第一版使用 configs/kather.json，其余命令相同。第一版训练 commit 尚无 evaluate.py，可用 ba76afb2 的复评入口（核心计算图未变）复评。本轮六项合同测试通过；两组 A/B checkpoint 的七项验证准确率和混淆矩阵重算全部一致。

## 下一步优先级

1. 同协议多种子复验，确认 +3.19 pp 是否稳定。
2. 固定 global 对照，区分共享层适应与专家扩展的影响；补无 replay / 固定容量对照。
3. 对齐既有高准确率 Kather 的架构与训练预算后再优化性能，最终锁定配置后评价独立测试集。

旧线性 smoke 和错误切分的 100%、62.5%、25%、0% 结果均作废，不参与以上比较。
