# CC BY核验清单论文插图导出

源码：`b75a56203`，入口 `export_licensed_examples.py`。只推理、不训练。
固定best SHA256：`036bc8caedcde4d6dabce276b1a2e4af15d960d88620098b19e1c198849b8bfe`。
模型使用已验证release的原runtime，完整5000图参考CC为0.8624925081777596。
版权结论沿用同学提供的核验包，不是新的法律认定。

|COCO ID|本项目划分|本次CC64|
|---|---|---|
|715|public-test（官方val2014）|0.9111731481|
|508|train|0.8686644825|
|450|train|0.8834260828|
|6730|train|0.8779526760|
|292271|train|0.9030175723|
|562382|train|0.9232745805|

**五张为训练样例，不可把六张的表现统称测试结果。**未依据分数删选，六张全部保留。
715与原逐图测试CC=0.9111730621吻合；指标使用原224×224GT协议。
GT和预测的绘图共享pair-max线性显示，放大版本不用于评分，无配准、裁剪或gamma优化。

服务器输出：
`/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t03_saliency/reports/licensed_examples_08625_20260929.zip`

ZIP SHA256：`6ed921460c0eae09f7652fbcefc055767f7d72ca62a7e2170611cd7a99449f66`。
114个文件逐项校验通过。包含原核验ZIP、署名说明、原图/GT/预测/NPY/热图/叠加/预览、
逐图JSON指标、汇总Markdown表、复现脚本、输入和权重SHA。GPU1临时推理完成后已释放。

复现（output必须为新目录）：

```bash
CUDA_VISIBLE_DEVICES=1 /home/guest3/miniconda3/envs/xml/bin/python \
  LightGenV2/tasks/t03_saliency/export_licensed_examples.py \
  --release /DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t03_saliency/releases/20260914_shs_cc08625 \
  --license-zip /DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t03_saliency/SALICON_CC_BY*.zip \
  --output /DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t03_saliency/reports/licensed_examples_08625_reproduction \
  --device cuda
```

依赖release内settings指定的冻结Qwen本地缓存及原prepared_maps；不联网取模型。
逐图指标包括CC/PCC、SIM、NSS、KLD、AUC-Judd及项目峰值归一化MAE。
