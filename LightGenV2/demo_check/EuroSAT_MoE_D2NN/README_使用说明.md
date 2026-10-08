# EuroSAT 光学/SAR：MoE 与 D2NN 实验代码包

本包保存本次 seed42 实验的原始训练代码，以及后续绘图、相位可视化和数据打包工具。
2026-09-15 已连接 connect.bjb1.seetacloud.com:25382 核对：420个封存文件全部存在，逐文件SHA256均与本地归档相同。
原服务器主机名：autodl-container-b61844b5bd-030a9ea3。

## 文件内容

- code/：420个原始封存文件、SOURCE_MANIFEST.json，以及队列读取所需的data_progress.json。
- code/experiments/：391个复用依赖文件，包含MoE、D2NN、冻结视觉前端、电子支路和分类头实现。
- code/PLAN.json、SPLIT.json、SPLIT_AUDIT.json：训练计划、53,784张图片的完整划分和空间分组审计。
- code/IMPLEMENTATION.md：架构、训练流程、预算和公平对比条件。
- code/licenses/、ATTRIBUTION.txt：数据来源、署名与许可快照。
- analysis_tools/：性能图、非线性相位mask、A/B配对样例和图片归档工具。工具依赖原实验结果文件，部分路径使用原Windows工作目录。
- reference_results/：四模型性能、架构记录、检查点选择记录及报告，不含PT权重。
- provenance/、PACKAGE_VERIFICATION.json、FILES_SHA256.json：封存、原结果复算及本次打包校验信息。

## 主要代码入口（相对code/）

| 用途 | 文件 |
|---|---|
| 顺序运行四模型的原实验队列 | run_eurosat.py |
| 训练、专家合并、D2NN AB检查点选择 | train_eurosat.py |
| 同一固定检查点的A/B评估与路由诊断 | evaluate_eurosat.py |
| 模型创建、损失与评估接口 | eurosat_runtime.py |
| 数据读取、56→224双三次处理、固定采样和增强种子 | eurosat_data.py |
| MoE专家合并与路由模式 | experiments/expert_merge/core.py |
| MoE/D2NN切换、D2NN的224→478双线性加载 | experiments/vision_transfer/model.py |
| 两个光电混合模块和自建十分类头 | experiments/vision_transfer/vision.py |
| 冻结的Qwen图像patch与位置嵌入前端 | experiments/vision_transfer/backbone.py |
| SAR编码与同地块对齐 | prepare_data.py |

## 运行环境与外部资源

原运行环境为Linux/CUDA，Python 3.11.16，解释器为/root/miniconda3/envs/opticsmoe/bin/python。
代码包含PyTorch、torchvision、transformers、safetensors、NumPy、Pillow、Matplotlib、PyYAML等依赖；数据准备还使用rasterio、pyproj、scipy和scikit-learn。
部分上游requirements文件保留在相应experiments目录中；本ZIP是源码归档，不是可直接部署到任意机器的容器或完整环境镜像。

原服务器路径：
- 主代码：/root/autodl-tmp/review/eurosat_expert_merge_20260912
- D2NN输出：/root/autodl-tmp/review/eurosat_d2nn_baselines_20260912
- 预处理图片：/root/autodl-tmp/data/eurosat_optical_sar
- Qwen缓存：/root/autodl-tmp/huggingface-cache

原队列入口为code/run_eurosat.py，需要已准备好的图片、IMAGE_MANIFEST.json、Qwen本地缓存和对应环境；检查点评估还需要选定的PT文件。
本包保留封存源码原样，配置中的绝对路径未改写。迁移机器或更改路径需要同步处理配置与封存校验，不能直接把修改后的代码视为原实验同一版本。
code/USER_AUTHORIZATION.json保留的是原实验授权记录。

## 数据与模型

MoE：四个224×224专家、224×224路由相位、478×478共享相位；D2NN：两层478×478相位。两者均有冻结Qwen视觉前端和自建十分类头，没有完整Qwen语言模型或视觉Transformer。
实验图片包与相位mask可视化已单独交付；本代码ZIP不重复包含数据集图片、训练PT权重或Qwen预训练权重。
同目录旁的下载资源说明可在EXTERNAL_ARTIFACTS.json中查看。
