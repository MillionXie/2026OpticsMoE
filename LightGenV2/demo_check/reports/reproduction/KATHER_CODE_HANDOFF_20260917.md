# 新输入尺寸的MoE / D2NN源码交接

ZIP：`releases/kather_coverage_code_20260917.zip`，187342字节；导出源码commit：`0a7bff96`。

SHA256：`d09a9d8bd2d5dd2dfd8a73df671be86df9080504efd690c9e3975807cdba6115`。

本包为源码交接，包含70个受manifest校验的文件及MANIFEST自身。包含全部光学算子依赖、两种架构的OEO开/关、2/4/6层、训练配置、固定镜像数据准备、数据manifest、中文README、独立训练/复评入口。不需要原工程或Git目录。MoE首层专家输入146×146；D2NN为474/472/470。相同四块50×50源信息，放大后恢复入射总功率。

不含数据图像、最终权重或最终性能。两个候选配置尚在验证，ZIP没有将其标为最优配置；导出时训练状态保存在`provenance/training_status_at_export.json`。已有训练不因打包而中断。

生成方式（Git仓库根目录，已存在的输出不覆盖）：

```bash
python LightGenV2/demo_check/build_lab_package.py --variant kather_code --run LightGenV2/demo_check/runs/simulation/kather2016_coverage_pilot_20260917 --data-manifest /DATA/DATA1/guest3/demo_reproduction_data/kather2016/data_manifest.json --out LightGenV2/demo_check/releases/kather_coverage_code_20260917.zip
```

验证在服务器`/tmp/kather_handoff_verification_20260917/kather_coverage_code`解压副本进行，脱离Git和原工程目录。包内manifest逐文件复核；四组模型检查前后向，六组MoE检查各专家首层照明、输入功率、初始参数及EMA复制。另对四架构分别执行完整训练集的一轮两层训练和保存权重后的验证集复评。此短程训练只用于验证交接入口，不作为性能实验。验证记录与ZIP同目录：`kather_coverage_code_20260917.verification.json`。

ZIP之外还有`.manifest.json`、`.sha256`和本地传输校验记录`.transfer.json`。直接将ZIP发给接收者，按包内README准备数据、训练与复评即可。
