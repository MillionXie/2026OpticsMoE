# MoE / D2NN 输入覆盖修正版：源码交接

此包可脱离原工程与Git运行，包含全部光学算子依赖、数据准备和训练脚本。当前重训仍在进行，**不含最终权重、最终性能或验证选定配置**。`configs/long.json`与`configs/lower_capture.json`是正在比较的两个候选，不能称为最优参数。

## 模型与输入

两种架构共享RGB编码：原图150×150，归一化到[0,1]，抗混叠缩放100×100、训练仿射增强，再将R、G、B、RGB平均各缩放50×50，按`[R,G;B,mean]`拼成100×100单通道振幅图。拼接保留颜色信息的空间位置，并非先转普通灰度图。

- **MoE**：路由器输入100×100；四块分别50→73后拼接146×146，填满每个专家首层，九路密集激活。每轮包含局部专家层与全局层，L=2/4/6对应1/2/3轮，另有一个路由相位板。
- **D2NN**：四块分别插值后拼接为474/472/470，分别对应2/4/6层，覆盖自己的相位面。
- 两者使用相同源图细节，放大后恢复原图总入射功率。仿真单波长532nm。没有Qwen、CNN前端或电子分类残差。
- OEO开关对应**每个主衍射层后**的强度探测、全场LayerNorm、ReLU、Softsign和零相位振幅重编码，不是只在末端添加。关闭OEO保留复场。

| `--arch` | 模型 |
|---|---|
| `moe_nooeo` | MoE，不加层间OEO |
| `moe` | MoE，每层OEO |
| `d2nn_wide_nooeo` | D2NN，不加层间OEO |
| `d2nn_wide` | D2NN，每层OEO |

## 安装和数据

在解压后的`kather_coverage_code`目录运行。需要NVIDIA GPU，建议Linux、Python 3.10与PyTorch 2.6.0；先安装适合本机CUDA的PyTorch，再安装`requirements.txt`其余依赖。文件列出导出服务器实际依赖版本。

```bash
python verify_manifest.py
python reproduction/prepare_kather_handoff.py --mirror --data-root data/kather2016 --out runs/prepare
```

数据不包含在ZIP中。脚本下载固定版本镜像，核验SHA256，生成`data/kather2016/kather2016_fixed_split.npz`及同目录`data_manifest.json`。已有相同缓存可直接指定路径。缓存SHA256须等于包根`data_manifest.json`的`cache_sha256`。

Kather2016原始数据CC BY 4.0，引用Kather等，Scientific Reports 6, 27988 (2016)，DOI:10.1038/srep27988；原始数据Zenodo:10.5281/zenodo.53169。镜像版本和下载地址见数据manifest。八类各625张，固定图像级训练/验证/测试3496/752/752；不代表患者独立划分，镜像与原始ZIP逐像素一致性尚未验证。

## 训练和复评

```bash
# 新覆盖MoE（无OEO），从头训练
CUDA_VISIBLE_DEVICES=0 python reproduction/handoff_kather.py --phase train --data data/kather2016/kather2016_fixed_split.npz --config configs/long.json --arch moe_nooeo --depth 6 --seed 17 --out runs/moe_L6_s17
# 相同配置的D2NN（无OEO）
CUDA_VISIBLE_DEVICES=0 python reproduction/handoff_kather.py --phase train --data data/kather2016/kather2016_fixed_split.npz --config configs/long.json --arch d2nn_wide_nooeo --depth 6 --seed 17 --out runs/d2nn_L6_s17
# 复评自己训练的best；默认仅验证集
CUDA_VISIBLE_DEVICES=0 python reproduction/handoff_kather.py --phase evaluate --data data/kather2016/kather2016_fixed_split.npz --checkpoint runs/moe_L6_s17/moe_nooeo_L6_seed17/best_checkpoint.pt --out runs/replay_moe --split val
```

替换`--arch`可开启逐层OEO；`--depth`可选2、4、6；配对种子为17、27、37。每次`--out`必须是新目录。配置固定后才使用`--split test`，不要根据测试集调整模型或选择最优轮次。

全部相位联合训练，无电子预训练或冻结阶段。AdamW、batch16、lr0.002余弦降至0.0002、EMA0.95、最多60轮；验证balanced NLL选择best，早停patience12、最少30轮。两个配置只差捕获损失权重0.2/0.02，均无dropout；其余参数以JSON为准。保存best、last、逐轮日志、逐样本训练/验证预测与梯度记录。

预测取八个探测窗口归一化能量的最大值。准确率为分类正确比例；balanced accuracy为八类召回率平均；AUROC是类别排序指标，不等于准确率。检查过拟合应同时看训练/验证曲线，不强制深度准确率单调。

## 修改入口

- `reproduction/kather_coverage.py`：新MoE专家输入覆盖及九路功率分配。
- `reproduction/bloodmnist_multiseed.py`：D2NN全孔径输入及功率归一化；虽沿用Blood文件名，Kather复用同一实现。
- `reproduction/bloodmnist_experiment.py`：共享RGB编码、训练循环及八类指标。
- `configs/*.json`：交接入口实际读取的训练参数。
- `adrenal_softsign_code_export_20260915_145336/code/`：完整光学算子依赖，名称为历史路径；本任务通过`handoff_kather.py`运行Kather八分类，不运行该目录旧二分类实验。

`ARCHITECTURE_AND_PROTOCOL.md`提供详细架构与公平性边界；其中工程内报告链接属于来源记录。理想九端口中继、局部与全局传播差异仍需单独物理验证。`verification/`是原工程覆盖/梯度检查，ZIP旁的解压验证记录另行验证本包可运行。`PROVENANCE.json`记录源码commit、依赖版本、数据哈希；`MANIFEST.json`逐文件校验。
