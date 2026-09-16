# EuroSAT无电子残差的相位训练试验

固定3通道图像振幅编码→两层相位和全画布相干传播→固定10区域CCD读出。
无Qwen、电子残差、可训练电子投影/分类头、alpha或中间OEO。动态MoE仍含
路由探测、归一化和振幅SLM控制，不称为完全被动的全光系统。

三组分别为动态四支路、相同四支路固定等功率分光、整孔径两层D2NN。前两组初始
主光路相位逐位相同，三组全局相位相同；动态组额外训练路由相位。三组共享输入编码、
主路输入总功率1、固定读出、NLL损失、Adam与cosine调度、20轮、batch32、数据顺序和增强。
不使用域辅助损失，不预设四专家对应哪个域。四支路不按窗口独立传播后拼接；在518²画布联合传播。

输入56²三通道各放大112²，固定排列为[channel0,channel1;channel2,zero]，构成224²。
四路专家224²、间距254、有效区478²；整孔径D2NN为两个478²相位，输入双线性放大并保持总功率。
λ532nm、像素17μm、主传播每层10cm。CCD十个32²区域，能量归一化作为类别分数。
动态路由另有一个224²相位、一次518²传播和四个60²区域，能量占比直接解释为功率份额。
参数数目：动态479364、固定四路429188、整孔径456968；并非严格等参数。
额外路由探测/控制成本需独立计入，不能将主光路功率相等写成整机功耗相等。

数据沿用原53,784图像清单的空间划分；从训练/验证每类按pair_id固定哈希顺序分别取
300/100对，两个域一起保留，得到6000/2000张。原始RGB、MS、SAR压缩包校验原记录摘要。
RGB裁56²，SAR重投影和VV/VH固定编码沿用原规则。仅训练/验证，不读取测试样本；
这不是原完整数据量与原混合模型的数值复现，不能直接比较旧表准确率。

```bash
python LightGenV2/demo_check/pure_optical/prepare.py --root /absolute/path/eurosat
CUDA_VISIBLE_DEVICES=0 python LightGenV2/demo_check/pure_optical/run.py --phase smoke --out /absolute/path/demo_check/runs/smoke/pure_optical_20260916
CUDA_VISIBLE_DEVICES=0 python LightGenV2/demo_check/pure_optical/run.py --phase train --data /absolute/path/eurosat/phase_only_v1/data.npz --out /absolute/path/demo_check/runs/simulation/pure_optical_20260916
```

运行保留配置、Git/source/data摘要、环境、初始验证、训练曲线、best/last权重、
验证逐样本预测、路由功率分布与探测器捕获比例。最佳模型只按验证准确率、NLL选取。
后续若研究电子融合，先对支路尺度和共同冻结权重做独立设计；本轮不混入该变量。
