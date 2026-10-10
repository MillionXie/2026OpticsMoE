# FishNet v1：新的八分类残差消融

2026-10-11用户要求使用空闲物理GPU1、更换未在工程用过且符合原许可/类别要求的数据集，
保留芒果最终结果，沿main同步、不新分支。2026-10-11仓库源码、配置与协议检索未发现
`FishNet`或DOI `10.17632/p3xh4fs7cp.1`已有实验。当前源码/实验只在T18增加适配入口，
原模型、历史训练器、芒果权重和全部结果保持原身份。

## 数据来源与任务

作者：Bishal Biswas、Rakibul Haque Rabbi、Md Mohtashim Masuk。
[作者Mendeley v1](https://data.mendeley.com/datasets/p3xh4fs7cp/1)，2025-05-27发布，
DOI `10.17632/p3xh4fs7cp.1`，**CC BY 4.0**。作者原文件
`Dataset.zip`为147,628,045字节，作者API SHA256
`dd4a6b47aa5a903bda05a0371060a526b82c487280e61b70f802cb505fc255fe`。

全部3,013张、原生八分类，包含鱼和虾，任务准确称为**鱼虾类别识别**，
不能把虾也描述成鱼类。按实际文件夹字典序固定标签，不合并或挑选类别：

| 标签 | 原文件夹 | 图像数 |
|---:|---|---:|
| 0 | Mola fish | 517 |
| 1 | Pabda fish | 435 |
| 2 | Prawns | 227 |
| 3 | Puti fish | 549 |
| 4 | Ruhi Fish | 53 |
| 5 | Shing mach | 598 |
| 6 | Shrimps | 122 |
| 7 | Tengra | 512 |

其中2,342张JPEG、671张HEIC。HEIC不能漏读：在数据目录独立安装
`pillow-heif==0.18.0`（不更新共享训练环境/Pillow），关闭缩略图解码，读取主图，
EXIF校正转RGB8，保存原图尺寸分布，再与JPEG一致双三次缩放到150×150。
原ZIP保留，不生成离线增强副本；全部主图实际为640×640，模型仍用固定100×100的[R,G;B,meanRGB]振幅输入。

固定split seed20261011，以原图精确像素和150×150缓存像素匹配的连通组，
逐类别按组70%/15%/15%划分，整数向下取整后余项进test。全部样本保留；
重复组不跨划分，跨标签的相同像素报错。实际train/val/test数量、类别支持、ID与SHA
以准备后的`data_manifest.json`为准。实际**train2,106／val448／test459**，
3,012个分组，1条精确重复与其原图处于同划分，全部3,013条保留。
图像级划分，作者未给个体/拍摄会话身份，不能声称个体独立泛化。
类别不平衡尤其Ruhi Fish少，必须同时报告accuracy、宏平均召回和每类支持数。

## 完全配对的模型与训练

仅MoE，无D2NN。2/4/6主干分别1/2/3层专家加同数global层，另有一个router。
九专家、原5cm传播/OEO LayerNorm-ReLU-Softsign、八个CCD32×32窗口、固定电子编码，
推理无CNN/Linear；初相位仍沿T18历史raw uniform[-1,1]后2πsigmoid，不改光路。
仅专家/global采用相干振幅混合 `[(1-rho)exp(i phi)+rho]U`，router无残差。
rho0与rho0.3两组都关闭专家vectorize，模型和参数量完全相同。

在看本数据训练结果前固定共同profile：seed17、每组完整100轮、batch16、
AdamW lr0.002余弦到0.0002，weight_decay0、EMA0.95、梯度裁剪1、
按train类别频率加权的label-smoothed NLL（0.02）＋收光0.2＋相位圆周平滑0.02；
增强旋转±10°、平移±3、尺度±5%。两组相同初始化、每轮顺序/增强和100轮预算，
不从芒果PT迁移，不因某组差而单独追加轮数。参数分别449,848／889,696／1,329,544。

沿用用户已授权的**每轮test开发选模**：每轮raw和EMA都测test，accuracy最高，
同分balanced NLL最低选PT；val仅诊断，训练梯度/标签仅来自train。
双方使用完全相同选模规则，固定100轮不提前停止。此为**test-selected DEVELOPMENT**，
不是独立泛化；不得根据预期残差收益或深度趋势挑选权重。
只保留best/last、全部history与更新best时逐样本预测。结束不再重推同权重test。

## 执行与同步

只用物理GPU1 UUID `GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d`，
本轮一张GPU、一个训练/推理子进程。首个独占启动预检发现GPU1被其它项目临时占用，
`fishnet_s17_e100_gpu1_20261011`已记录失败，未进入smoke/训练，数据已准备并保留。
沿用用户此前不要等待、允许指定GPU剩余显存的授权，新入口`--use-spare-memory`使用
Torch分配上限3GiB、启动时另留800MiB；复用既有`gpu_budget.py`，不升级其它环境、
不杀别项目。microbatch2按样本比例（含尾batch、收光与平滑项）积累到有效batch16，
每宏batch裁剪/AdamW/EMA一次，评估也batch2，OEO逐样本归一化。两个残差组都用
相同微批量执行，不改变有效batch或优化步数，浮点求和次序可能有微小差异。
CPU准备→六层smoke（rho0精确一致/初始化/梯度/router无残差）→
六层两组→四层两组→两层两组，全部串行；每阶段退出释放GPU，再启动下一项。
配对完成核验同初始化SHA、顺序/增强哈希、参数量、数据SHA和100轮数。
若剩余显存不足预设上限/余量，停止调度并记录，不杀其他项目。

入口：`fishnet_ablation.py --phase suite --data-root ... --out ... --gpu GPU-...`。
配置`fishnet_profile.json`；数据准备`prepare_fishnet.py`。
服务器数据根 `/DATA/DATA1/guest3/t18_fishnet_v1_20261011`。
新run `runs/simulation/fishnet_s17_e100_gpu1_spare_20261011`，原失败启动不覆盖。
每组配置、Git/源码/数据SHA、
command/PID/UUID、best/last/history/逐样本预测/梯度与配对收据完整保留。
GPU仅按UUID选择，源码先测试/commit/push main再服务器干净fetch/ff-only同步。
不上传原ZIP/缓存/PT/CCD、不覆盖其它窗口未跟踪文件。单种子无误差条，负结果也保留。
