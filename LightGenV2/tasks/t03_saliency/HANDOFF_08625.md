# SALICON 0.8625：师姐/AI先读这里

这是固定候选交接，不是最近退步的交替训练末轮。
checkpoint SHA256：036bc8caedcde4d6dabce276b1a2e4af15d960d88620098b19e1c198849b8bfe。
完整5000图CC64=0.8624925081777596；同权重去光=0.8422947020969439。
alpha=0.43068659/0.44104558，Top2使用率23.46/26.29/23.12/27.13%。
标准测试关闭随机光噪声；这些是仿真数值，不是师姐平台的实测数值。

## 先做离线核验，不需要访问Hugging Face

解压到短路径，例如Windows `E:\lab\salicon8625` 或Linux新的任务目录。
使用已有可运行Qwen实验的Python环境；不要覆盖原环境，尤其不要盲目更换PyTorch。
参考依赖在 `requirements-reference.txt`（不包含GPU PyTorch安装命令，按本机显卡选择）。

```bash
python handoff.py verify
python handoff.py simulate --fields 4 --device cpu
# 0代表全5000图；不同batch/设备可能有浮点微差
python handoff.py simulate --fields 0 --device cuda --output simulation_full
```

4图分数不能冒充全量分数。全量缓存包含标签和原sample_id，可直接复现，无需原图或大模型下载。
缓存是冻结patch embedding和位置编码之后、第一个替换块之前的张量，不是缓存预测。
实时仍执行输入适配、电子残差、光router/专家/全局、融合及显著性头。原生TF/attention不执行。

## 光路与BMP

只有3次传播：vision_router → vision_expert → vision_global。
没有语言阶段，不是视频，也不是6次传播。输出224×224显著性概率图，主指标CC而非SRCC。
光学合同：532nm、17µm模型采样、478×478有效区、10cm距离；不能任意缩小成另一物理口径。
4专家Top2，专家224×224；共6张相位参数合成为3张面板相位BMP。

```bash
# 离线参考，17µm/1024×1024振幅 + 8µm/1920×1200相位
python handoff.py export-reference-bmp --profile meadowlark17 --fields 4 --output reference17
# 离线参考，8µm/1920×1080振幅 + 8µm/1920×1200相位
python handoff.py export-reference-bmp --profile shs8 --fields 4 --output reference8
```

8µm面板的模型有效区转换为约1016×1016物理等价区。
**这两种参考导出均居中、不翻转、不反灰度；不是对你平台的方向/LUT/中心标定。**
参考BMP的后两层来自仿真上游，不能用于宣称真实闭环结果。正式过程必须先捕获Router CCD，
从实测四区域能量计算Top2，再生成Expert输入；随后用实测Expert CCD生成Global输入。
原始CCD只做已标定几何校正与固定单位换算，网络保留原mean-only归一化；不对每图做log/gamma/拉对比。
PNG预览按99.5百分位做显示拉伸，数值处理使用NPY/CCD DN，不使用预览PNG。

## 实验设备入口与移植边界（重要）

`run.py` / `COMMAND_SHS.md` 是已经在SHS-202-M相机＋Holoeye振幅8µm平台实采验证的入口。
它依赖独立的 `ABO_Lab_SHS_8um` 控制环境/厂商SDK，不是17µm Meadowlark/TUCam驱动。
包内**没有借用别人ROI、400µs曝光、相位翻转或反灰度当作你的配置**。
若设备正是SHS，请使用本机已验证控制工程，通过 `--bench-root` 指定目录，建立本机LAB JSON；
每次 `init/prepare/capture/evaluate` 都显式传 `--config 本机LAB.json`，详细顺序见COMMAND_SHS。
**本次确认设备为17µm Meadowlark＋8µm手动相位＋TUCam，请只按 `COMMAND_MEADOWLARK.md` 操作。**
新增 `meadowlark.py` 复用原生hardware_sdk采集，已接入3阶段实测上游合同与SHA审计。
先复用已有LGVQ硬件包做LUT、曝光、四角ROI和时序标定，再绑定本机formal_hardware.yaml。
它通过离线回放及模拟采集日志回归；尚未在收件方实际硬件上验证，不冒充实采测试。
不要用现有run.py默认SHS capture去打开另一种设备，也不要让服务器远程控制未知相位/相机。

## 微调边界，不要把测试缓存当训练数据

本包的 `inputs/` 是5000张官方val2014，作为项目public-test；不是10000张train2014。
本包当前是固定权重推理/3阶段实测交接，**没有现成的一键实测微调CLI**。
`runtime/`包含完整相关Python实现，`source_configs/`和`source_notes/`补全对应源码版本的配置/说明，
供后续AI实现任务专属微调。不能直接用推理函数 `replay()` 反向传播，它在inference_mode中。
仿真训练入口为 `LightGenV2.tasks.t03_saliency.run`；原始训练需要另备SALICON train2014、
密度/注视点标注、匹配的冻结前端/教师缓存，并按新路径修改配置；不假定它们在本压缩包里。
若微调固定mask：采集train子集，冻结相位/router，利用真实CCD读出适配电子尾部；
若改了上游电子输入映射，旧CCD不再对应当前入射场，必须重新采集或使用明确披露的代理梯度方法。
更新checkpoint要用新run和新身份，不覆盖这里的固定best，也不能删除load_model的SHA保护冒充原候选。
原训练没有validation，按public-test选best，存在选模偏差；后续实验请披露，不称独立盲测。

## 文件地图

- `weights/best_checkpoint.pt`：唯一采用的原候选，不携带多轮/末轮mask。
- `inputs/`：5000张精确冻结前端缓存＋标签＋sample_id。
- `phases/`：三张标准相位场NPY；硬件BMP需依设备几何、方向编码生成。
- `runtime/`：原部署包固定源码；包含独立cached模型和真实3CCD替代/重放实现。
- `release.json`：原模型源码commit、完整仿真指标、样本清单和SHA。
- `source_configs/`、`source_notes/`：对应原源码commit的复现配置与规则，不是已经本机化的执行配置。
- `handoff.json`：交接源码commit、原包身份与限制；`SHA256.json`为全部原件和新增文件校验。
- `COMMAND_SHS.md`：SHS平台历史实测步骤；平台相关路径必须修改，不直接复制另一台机器参数。

主包不包含原始视频、其他任务run、训练大模型权重、SSH凭据、他人的采集数据或厂商安装包。
