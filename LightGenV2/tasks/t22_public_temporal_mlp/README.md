# LightGen2 temporal video quality reference

本目录是作者侧研究、核验与打包入口。独立推理工程位于仓库根目录
[`LightGen2`](../../../LightGen2/README.md)。详细用户说明为该目录的
`README.docx`。公开发行包只包含推理、结果绘图、选定权重和归一化RGB输入。

## 唯一推理流程

在独立工程中运行 `python infer.py`。默认读取 `data/test_rgb.npz`，
对558条不同视频的归一化RGB帧推理，按原顺序组成35幅16视频相干光场。
末组14条真实视频和2条循环填充。558条真实预测及人工标签参与SRCC统计。
同一次推理捕获其中一幅光场，默认第1幅，供绘图导出。`visualize.py`是辅助模块。

输入已完成四帧采样、65%中心裁剪、448×448缩放及RGB归一化。
patch投影、位置编码、固定汇聚和其余网络仍由推理源码执行。
NPZ采用float16容器保存原冻结前端bfloat16精度的归一化RGB。
逐场加载控制内存占用。正式发行目录中的`data`只含该NPZ。

## 当前核验值

| 项目 | 记录 |
| --- | --- |
| 当前NPZ入口完整558条SRCC | 0.7988976461907575 |
| 原缓存入口558条SRCC | 0.7988976461907575 |
| 前一原视频CUDA入口SRCC | 0.7988921552113833 |
| 当前完整程序时间 | 23.2158772秒 |
| 计时设备 | NVIDIA GeForce RTX 4070 Laptop GPU |
| 计时范围 | 模块导入、加载权重、完整NPZ推理及结果绘图导出 |

本次只调整输入边界、统一入口和交付结构，权重未重训。
离线CPU预处理与此前CUDA归一化在一个像素等级上的精度舍入不同。
与前一原视频入口相比，当前558条MOS最大差约0.0036812，平均绝对差约0.0002337。
原始TEST用于开发选模，评估口径继承正式记录。

## 作者侧文件

| 路径 | 用途 |
| --- | --- |
| `model.py`, `infer.py` | 原研究实现及缓存推理入口 |
| `runs/simulation/current/LightGen.pt` | 选定研究权重 |
| `configs/selected.json` | 作者侧选定版本身份 |
| `assets/manifest.csv` | 2250 TRAIN和558 TEST原始身份、顺序和标签 |
| `tools/prepare_public_inputs.py` | 将原视频按正式预处理生成归一化RGB NPZ |
| `tools/build_public_guide.py` | 生成独立工程中的可编辑说明 |
| `tools/build_release.py` | 核验权重映射、数据身份、指标和光场数组后打包 |
| `tests/` | 研究实现与独立推理的回归检查 |
| `docs/DEVELOPMENT.md` | 研究过程及历史实验说明 |
| `runs/simulation/current/validation/public_inference/` | 私有核验记录及发行SHA |

根目录`LightGen2`是源码维护与用户审查入口。`releases/current`是自动生成的镜像。
发行包为`releases/LightGen2.zip`，旁边保留压缩包SHA。

## 打包顺序

1. 准备归一化RGB NPZ并核对558条身份及原视频SHA。
2. 在独立工程中完成一次全量推理，生成编号结果及完整程序时间。
3. 用`tools/build_public_guide.py`生成说明并完成逐页渲染检查。
4. 测试通过、提交并发布源码后运行`tools/build_release.py`。
5. 源码通过Git同步。权重、NPZ、Word及结果通过逐文件SHA核验同步。

公开结果依编号阅读。`01`为全部预测，`02`为指标及总运行时间，
`03–04`为全量评价图，`05–06`对应选定光场输入，`07`为558条专家份额，
`08–13`为完整vision和language相位及输入输出光场的NPZ/PNG。
完整输入文件没有保存循环填充视频。填充只在推理末组中构建，统计时排除。
