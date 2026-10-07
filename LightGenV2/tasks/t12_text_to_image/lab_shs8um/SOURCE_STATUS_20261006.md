# T12 正式硬件入口源码登记

七份采集/审计入口按 Git 提交 `f15e46a6660731d8d73d4438ceb227545566f220` 收录，与实验室 `T12_Small_Lab_SHS_8um/candidate_robust17m_20260927/source` 逐文件一致。没有启动设备或改变光学合同。

正式模型为 17M，旧 README 针对 9.96M 试采，不能沿用其模型身份或测速。原本地两份旧入口保存在提交 `ff0c7a2e1`；实验室原工程、数据、权重与全部测速未动。

正式报告位于实验室 `runs/robust17m_full_test_20260927/report.json`：2304 TEST、六层 13824 CCD，原权重 SHA256 `5b4f9a37f19ce95cf23e4b874badc8e88955559553d47929fb80cd3cbf527cac`。报告总耗时包括采集和末端推理，不是单样本延迟。

后续四份适配辅助源码也已从原 Git 提交收录：decoder 三份来自 `c5282545c35f47e41f9e4a91615e6e86d1c731c2`，比较导出来自 `6c8cfab4e442488f3e99aa8b0728c5ff91a643b9`。现在本目录 17 份 Python 源码均与实际正式候选逐文件一致，完整身份见 [源码身份清单](SOURCE_IDENTITY_20261006.json)。

CPU 检查覆盖源码 SHA/语法、TRAIN 适配器与固定捕获入口的替换合同、末端 decoder 冻结和上游哈希守卫；不运行正式训练、TEST、GPU 或 SDK。原科学协议保持 TRAIN 拟合、VAL 选模、封存后 TEST，不能套用其他任务的 TEST 开发选模授权。

尚待完整核验运行环境、外部 ABO 设备依赖及全部数据/PT 闭包。因此源码收录完成不代表硬件迁移完成；不要直接替换实验室运行目录。旧 README 为 9.96M 历史试采说明，正式模型身份以上述清单及任务复现页为准。

## 主线显式设备适配入口（2026-10-07）

新增`main_layerwise.py`不修改上述17份原源码。它先核验原`run_layerwise.py`的LF SHA，
仅去掉旧ABO项目参数/外部导入，绑定已审计主线geometry及SHSBench的显式构造器；
原模型CUDA执行、阶段顺序、相位/BMP、400us/GainX4/wait240、旧CCD复用与解码逻辑不改。
三个实际使用的geometry函数及Bench采集/释放逻辑与旧实现逐AST一致。

```text
python -m LightGenV2.tasks.t12_text_to_image.lab_shs8um.main_layerwise --mode inspect --project <原17M候选资产目录> --output <不存在的新输出目录> --reuse <原同权重CCD目录> --machine-config <私有SHS配置JSON> --phase-sdk <含Blink_C_wrapper.dll的原SDK目录> --phase-lut <原LUT文件> --amplitude-sdk <振幅SDK目录>
```

默认inspect仅查源合同、原5b4f权重SHA、路径存在并编译适配代码，不加载模型/SDK、不写输出。
明确授权的新采集才使用`--mode capture`，本次治理没有执行capture，也不替换原计划任务。
支持原17M TEST/VAL逐层入口；TRAIN显式加`--split train --selection <原TRAIN选择JSON>`，
由原`run_train_capture.py`的固定替换字典生成同一采集逻辑，另核该原适配器SHA。
保持20736原TRAIN大小、唯一索引、test_product_overlap/test_source_hash_overlap均为0，
TRAIN不允许max-samples；只读inspect不写原selection。eeec电子读出仍是单独入口。
目录存在不证明SDK二进制身份/可用设备/环境完整；原CUDA与显示边界尚未现场回归。
八项AST/合成文件测试验证绑定、TRAIN重叠/索引拒绝条件及导入边界，不代表正式数据、精度或光路验收。

实验室实际main随后通过该入口的只读inspect：原5b4f PT、合同与机器配置前后SHA
一致，原正式CCD目录及SDK/LUT路径存在，无模型加载/设备打开/输出创建。
见[实际资产绑定收据](../../../../maintenance/storage/T12_ACTUAL_MAIN_BINDING_20261007.json)。
只检查目录存在，不据此证明全部CCD完整或设备可用；原现场运行项目仍保留。
