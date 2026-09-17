# Kather2016 光学分类代码

四种模型：`moe_nooeo`、`moe`（逐层OEO）、`d2nn_wide_nooeo`、`d2nn_wide`（逐层OEO）。主网络均支持2/4/6层；包内提供四种六层seed17权重和全部36次正式实验数值。D2NN输入覆盖相位板并匹配入射总功率。

原图150×150 RGB，缩放后将四张50×50图按`[R,G;B,(R+G+B)/3]`拼成100×100振幅输入。MoE路由器使用100×100；专家输入四块分别50→73，拼为146×146填满首层；D2NN四块也分别放大以填满自己的相位面。两者都保持总入射功率，且没有获得额外原图细节。不是RGB三波长传播，没有CNN、Qwen、电子分类残差。MoE为九路密集光学路由，每个循环包含一个局部专家层和一个全局层；2/4/6层分别对应1/2/3个循环，另有一个路由相位板。OEO为每层传播后测强度、全场LayerNorm、ReLU、Softsign、零相位振幅重编码，无可训练电子权重。

`selected_config.json`是验证集选出的共同训练配置；所有主层和路由相位同时训练。新实验最多60轮，比较延长训练、减轻探测器捕获辅助损失两种方案；不使用dropout。旧输入下启动的dropout诊断因输入覆盖修正而中止，不参与选模。未改变旧baseline，未按测试集差距选配置。详细架构、损失及结果见`reports/reproduction/`。

数据：Kather2016组织纹理八分类（原始数据CC BY 4.0，DOI 10.1038/srep27988；Zenodo 53169）。本实验固定镜像版本和图像级3496/752/752划分，详见`data_manifest.json`；不代表患者独立验证。数据缓存不装入代码包。联网执行`python reproduction/prepare_kather_handoff.py --mirror --data-root data/kather2016 --out runs/prepare`重建固定镜像缓存，并核对生成缓存与manifest的SHA256。所需文件是`kather2016_fixed_split.npz`，同目录须有`data_manifest.json`。

在包根目录安装requirements；PyTorch选择适配本机CUDA的2.6.0版本。可脱离Git使用：

```bash
# 用现有权重复评验证集；out必须是尚不存在的目录
CUDA_VISIBLE_DEVICES=0 python reproduction/handoff_kather.py --phase evaluate --data /path/kather2016_fixed_split.npz --checkpoint reference_weights/moe_L6_seed17.pt --out runs/replay_moe --split val
# 重新训练，同一配置仅改变arch/depth/seed
CUDA_VISIBLE_DEVICES=0 python reproduction/handoff_kather.py --phase train --data /path/kather2016_fixed_split.npz --config selected_config.json --arch moe_nooeo --depth 6 --seed 17 --out runs/train_moe_nooeo
```

训练只读取训练/验证集；验证balanced NLL选择best，早停、EMA及完整参数在配置中。`--split test`用于配置固定后的测试。不能把之前已用于观察结果的测试集称为全新盲测。模型预测为八个探测窗口能量归一化后argmax；准确率是预测正确比例，balanced accuracy是八类召回率平均，AUROC衡量类别排序能力，不等于准确率。

`adrenal_softsign_code_export_.../code`是保留路径以复用的光学算子依赖，其历史文件名不表示本任务运行Adrenal二分类。实际入口是`reproduction/handoff_kather.py`。`reference_results`中服务器路径仅为来源记录；复评使用本包`reference_weights`。`MANIFEST.json`逐文件列出SHA256，ZIP外另有整体SHA256。
