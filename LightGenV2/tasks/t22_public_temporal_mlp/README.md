# LightGen-II temporal video quality reference

本目录保存作者侧研究、核验与打包工具。独立推理工程位于仓库根目录
[`LightGen-II`](../../../LightGen-II/README.pdf)，可编辑说明为该目录的`README.docx`。
正式推理权重采用每支global共享一块相位的结构。

## 唯一推理流程

在独立工程中运行`python infer.py`。默认读取`data/input_for_review.npz`，将558条不同视频
按固定顺序组成35幅16视频相干光场。最后一幅含14条真实视频和2条循环填充。
558条真实预测与`data/reference_mos.csv`中的人工temporal MOS标签参与SRCC统计。
`data/input_for_review.npz`保存视频帧及其身份信息，标签独立保存并仅供指标计算。
模型前向只接收视频帧。一次完整推理同时捕获一幅光场，
默认第1幅，导出完整相位、输入输出光场及相应视频帧。

输入已完成四帧采样、65%中心裁剪、448×448缩放和RGB归一化。
NPZ采用float16容器保存冻结前端bfloat16精度的归一化RGB，逐场读取。
patch投影、位置编码、固定汇聚及后续网络均由推理代码执行。
循环填充在内存中构建，预测、质量指标和专家份额均按558条实际视频统计。

## 共享mask约束与修正

上一轮SRCC为0.796289的版本，每支global保存了16块独立可训练相位，优化后产生槽位差异。
本轮vision和language各保存一个`1×111×111`相位参数，通过广播应用到16个视频槽位。
每支全部槽位的梯度汇入同一个参数。初始化使用父模型16块物理相位的逐像素圆均值。
两支各有自己的相位，参数在所属分支的全部视频和光场中共享。

| 分支 | 可训练相位形状 | 导出槽位数 | 不同mask数 | 最大槽位相位差 |
| --- | --- | --- | --- | --- |
| vision | 1×111×111 | 16 | 1 | 0 rad |
| language | 1×111×111 | 16 | 1 | 0 rad |

研究检查点schema为9，架构标记以`_shared_phase`结尾。
公开检查点架构为`lightgen2_temporal_inference_v3_shared_phase`。
加载器同时校验架构、共享配置和参数形状。回归检查覆盖前向、梯度、保存加载及导出。
打包时额外核验相位数组中16块mask逐元素相等。
历史检查点按其原架构和配置读取。

## 当前选定权重与推理结果

训练run为`runs/simulation/shared_global_phase_20261114/noise_refinement`，共完成120轮，
按原噪声目标余量和物理门槛选定第60轮best。正式冻结副本位于该run同级`selected`。
开发阶段第35轮的核验记录保存在`validated_epoch35`，其原始文件按SHA归档。
当前唯一发布入口由`configs/selected.json`指向`runs/simulation/current/LightGen.pt`。

研究PT的SHA256为`df3ad58a1cb22e277bc792902a5af50ed7b0f0c3f427174784e1e2341f3a8a7c`。
独立推理PT的SHA256为`c6593a2b801d74698e4065cd63dcc3ba42fd24c8f571d4f8f3b19bdd8e41701e`。
两者保留的模型张量已逐一核验相等。

| 项目 | 核验值 |
| --- | --- |
| 完整558条归一化RGB推理SRCC | 0.7970645224412813 |
| 完整558条缓存前端推理SRCC | 0.7970645224412813 |
| 两入口最大MOS浮点差 | 0.00004196166992 |
| 16次槽位轮换平均SRCC | 0.7970111926 |
| 16次槽位轮换最低SRCC | 0.7961938741 |

公开推理入口导出558条预测、完整精度的SRCC和一组16视频的输入、相位及输出场。
公开说明中的SRCC按两位小数显示为0.80。
推理入口及结果JSON仅保留预测评价信息。既有计时原始记录按原身份归档。

## 传播核与前端核验

传播核对照正式时序版本`experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54/modeling.py`
中的`AngularSpectrum`和`configs/release/temporal.yaml`。518×518画布、532 nm波长、
17 μm像素间距、0.1 m距离和0.5°截止角一致。传递函数逐元素相等，
随机复光场、常数场及归档真实调制输入的传播输出最大绝对差均为0。
正式版本另有输入形状校验。当前发布传播类的实现文本保持原样。
核验记录位于`runs/smoke/lightgenii_release_20261116/propagation_verification.json`。

冻结前端按时间切片配对、patch投影、patch分组、位置编码与两次空间汇聚展开。
保留的张量名称、形状及前向计算顺序一致，逐元素回归覆盖展开前后的结果。
原Qwen3-VL的Transformers 4.57.3模型文件为1561行，patch嵌入类为18行。
本项目的前端提取范围及定制汇聚由当前代码直接表达。

## 相位收益与逐视频泄漏

逐视频审计使用整幅518×518相干传播，每次只照明一个视频，统计进入其他视频探测区的
能量占全探测面总能量的比例。完整558条及最后两个填充光源分别核验。

| 分支 | 跨视频能量均值 | 单视频最大值 | 10%读噪输入下的最大值 |
| --- | --- | --- | --- |
| vision | 0.3059364027% | 0.9861648083% | 0.9861648083% |
| language | 0.2237852863% | 0.4880309105% | 0.7039785385% |

全部已审计真实视频的比例严格小于1%。填充光源最大值为vision
0.2731561661%、language 0.2520322800%。
这里统计传播能量。CCD随机读出电噪声的注入与能量核算分别记录。

以下对照固定其余权重，单独将指定分支的学习相位替换为常相位。
圆均值取照明区域的相位圆均值，π和0采用统一常相位。

| 指定分支 | 圆均值常相位SRCC | π常相位SRCC | 0常相位SRCC | 学习相位相对最好常相位的增量 |
| --- | --- | --- | --- | --- |
| vision | 0.6930961154 | 0.6930961154 | 0.6931027513 | 0.1039617711 |
| language | 0.7925189930 | 0.7925189930 | 0.7925189930 | 0.0045455294 |

均匀随机相位使用每支一块随机mask并广播到16个槽位，10个种子的平均SRCC为
vision 0.6605516528、language 0.7651392148。
仅将传播替换为复光场恒等算子时，SRCC为vision
-0.0946765229、language 0.7862378997。
这些数值均来自固定其余权重的推理对照。

抑制跨视频复光场后的SRCC为0.7943924667，
相对联合传播变化-0.0026720557。
相位平移1像素的四方向最低SRCC为0.7893804666，
CCD平移1像素最低为0.7816406722。

## 专家份额与mask纹理

| 分支 | E1 | E2 | E3 | E4 | 逐槽位最大偏离25%的幅度 |
| --- | --- | --- | --- | --- | --- |
| vision | 26.7921% | 25.0896% | 23.3423% | 24.7760% | 7.1429个百分点 |
| language | 23.5663% | 24.7312% | 26.4337% | 25.2688% | 9.2857个百分点 |

两支均使用全部6种Top2组合，变化比例定义和逐槽位统计保存在评价JSON。
共享相位保留父模型的空间纹理，完整16槽位和输入输出光场由同次推理导出。
`phase_change_RMS_rad`比较新共享mask与父模型16块相位，包含共享初始化和训练两部分差异。
其值为vision 0.267993 rad、language
0.125789 rad。

## CCD噪声训练和复测

配置为`configs/shared_global_phase_20261114.json`，使用受保护父模型的干净TRAIN输出作教师。
梯度仅使用2250条TRAIN，原始TEST的558条用于开发选模。
优化现有共享相位、振幅编码器、32单元读出头和language路由末层偏置。
损失包括干净监督、带噪监督、教师一致性、相位正则、泄漏与专家均衡约束。
15%概率的±1像素配准扰动沿用配置记录，梯度相位平滑尺度为1像素。

两支CCD强度噪声均加在平方律探测之后、区域均值归一化之前。
读噪为`max(I + r * I_reference * Normal(0,1), 0)`。
散粒噪声为`Poisson(I * N / I_reference) * I_reference / N`，参考强度处相对标准差为
`1/sqrt(N)`。5%和10%分别对应N=400和N=100。
混合条件让两种噪声在参考强度处各贡献总方差的一半。
Poisson前向实际采样，反向采用恒等直通梯度估计。该反向为近似梯度。

开发选模使用40301–40303三个种子，最终检查使用29301–29310十个种子。
两支CCD在每种条件下同时加噪，每次完整推理558条。
下表自身参考采用候选TRAIN平均探测强度，固定参考采用父模型TRAIN统计。

| 噪声条件 | 父模型平均SRCC | 共享版自身参考平均 | 共享版自身参考最低 | 共享版固定参考最低 |
| --- | --- | --- | --- | --- |
| 高斯读噪5% | 0.794084 | 0.795638 | 0.793267 | 0.792824 |
| 高斯读噪10% | 0.789296 | 0.792143 | 0.786501 | 0.785699 |
| Poisson散粒噪声5% | 0.795534 | 0.796550 | 0.793057 | 0.793301 |
| Poisson散粒噪声10% | 0.794997 | 0.796201 | 0.793641 | 0.793563 |
| 混合噪声5% | 0.793917 | 0.795771 | 0.793810 | 0.793752 |
| 混合噪声10% | 0.790555 | 0.794936 | 0.793612 | 0.793153 |

所有已测种子均达到5%不低于0.790、10%不低于0.780的门槛。
原始记录位于`selected/review/controls/detector_noise_fixed`和`detector_noise_relative`。
父模型统一A100复测位于`comparison/previous_unshared_detector_noise`，上述路径均相对
`runs/simulation/shared_global_phase_20261114`。
193份预测CSV已逐表复算SRCC并核验样本ID、顺序、标签和558条计数。
当前百分比是数值模型中的定义，真实CCD曝光、增益、暗场和平场标定仍待建立。

## 作者文件与发布

| 路径 | 用途 |
| --- | --- |
| `configs/selected.json` | 当前权重和共享结构 |
| `runs/simulation/current/LightGen.pt` | 当前研究PT |
| `runs/simulation/current/validation/shared_phase` | 完整噪声、物理、路由和mask一致性核验 |
| `runs/simulation/current/validation/public_inference` | 独立发行包核验及SHA |
| `assets/manifest.csv` | 2250 TRAIN和558 TEST身份与标签 |
| `tools/build_public_guide.py` | 在作者当前Word上应用发布格式，保留手工修改 |
| `tools/build_release.py` | 张量、身份、指标、mask一致性和光场复算后打包 |
| `tests` | 共享梯度、检查点兼容、前向和导出回归检查 |

根目录`LightGen-II`维护独立推理源码，`releases/current`为生成镜像，
`releases/LightGen-II.zip`为发行包。源码通过Git同步，PT、数据、Word和结果按SHA核验同步。
正式包使用`README.pdf`，作者编辑用的`README.docx`单独保留在根目录工程中。
主项目代码沿用ACCEL官方README声明的GPL v3.0，Qwen上游Apache-2.0许可与署名文件随包附带。
本轮修复Python 3.8不支持`Path.write_text(newline=...)`的兼容问题，指标经临时文件原子替换保存。
已在RFL的Python 3.8.0、PyTorch 2.4.1环境核验完整558条推理的兼容性。
新增主入口回归覆盖旧版Pathlib及重复写入已有指标文件。
兼容回归记录位于`runs/smoke/python38_final_20261115`。
根目录`LICENSE`为本项目GPL v3.0文本，`licenses/NOTICE.txt`与`licenses/QWEN_LICENSE.txt`
集中保存第三方署名、数据出处和上游Apache-2.0许可。
历史ZIP、PT、配置、作者快照和临时材料统一保存在本任务`runs/archives`中。
2026-11-16迁移逐文件核验SHA，清单为
`runs/smoke/storage_relocation_20261116/migration_manifest.json`。
发布暂存和镜像备份分别在`releases`和`runs/archives/release_mirrors`中建立。

随包提供的参考结果保存在`reference_results/`；运行`infer.py`后在`results/`生成本次输出。
推理结束后按视频ID核对预测和SRCC，在终端显示`Reference comparison: PASS`或报错。
MOS绝对误差容差为0.001，SRCC绝对误差容差为0.0001；输出路径不得覆盖或链接到参考目录。
ZIP仅收集参考结果，不包含`results/`中的本次输出。参考文件及模型、数据保持原样。

公开结果按文件前缀顺序阅读，01为全部预测，02为完整精度SRCC与视频数量，
04为所选光场视频帧，05–07为vision输入、mask和输出，08–10为language输入、mask和输出。
2026-11-24按作者要求取消散点图输出，兼容清理结果目录中遗留的03文件。
当前Word沿用作者增大的正文字号和格式，六张光场图按每支一页重新分组，
每页上方为输入与mask，下方为输出。文档字体、样式、表格、页边距及页脚通过XML逐项核验。
公开CSV保留视频ID、人工MOS和预测MOS。光场分组及有效位仍保存在对应光学NPZ中，
打包工具据此核对样本身份和尾组填充。专家份额的作者侧核验记录保留在研究run中。

## 说明文档的论文结果引用

说明文档末段的实验数据核对自投稿文件
`E:/2026LightGen2Data/投稿文件/supplement information.pdf`。
论文实验系统为0.611 ms/frame与0.113 J/frame，Qwen3-VL对照为71.326 ms/frame与27.987 J/frame。
Qwen3-VL完整558条temporal SRCC为0.7663428037，文档显示0.77。
本工程完整推理SRCC显示0.80。文档将系统速度和能耗归属于论文实验系统。

## 公开传播参考与物理采样

`LightGen2/model.py`的`Propagation`采用
`opticalmoe/notebooks/github_D2NN_mnist4.ipynb`的中心频谱角谱传播写法。
保留选定模型的532 nm波长、17 µm采样、518画布、478有效区域、0.1 m传播距离与0.5°频域孔径。
Notebook的默认16 µm采样和0.2 m距离在实例化时由本工程参数替换。
二维FFT平移显式作用于最后两个空间维度。传播核以双精度构建，运行时沿用complex64。
记录用的`transfer`保留未平移频序，供原始NPZ的NumPy复算使用。
本轮修改前的独立数值对照显示，传播核最大绝对差7.45×10⁻⁹，
既有保存场的复输出相对L2差1.86×10⁻⁹。

17 µm是本工程输入与mask的共同计算网格。原版硬件导出脚本
`experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54/export_hardware_masks.py`
将相位按物理坐标从17 µm重采样到8 µm相位SLM，比例为17/8。
输入振幅SLM像元为17 µm。硬件导出仍需执行该物理坐标映射。

本轮数值复测记录位于`runs/smoke/public_propagation_20261116`。
Windows Python 3.8、3.11和服务器Python 3.11均通过97项回归测试。
完整558条预测CSV与修改前逐字节一致，SRCC为0.7970645224，因此保留现有权重。
逐视频最大泄漏为vision 0.9861648%、language 0.4880309%。
两支相位相对最佳常相位的SRCC收益为0.1039618和0.0045455。
固定与自身参考、三类CCD噪声、两个强度、每条件10个种子的120次完整复测均通过原门槛。
5%与10%条件的最低SRCC分别为0.7931133、0.7896774。
独立ZIP解压后完成558条推理，预测CSV与根目录工程一致，尾组光场数组通过复算。

## Qwen前端的位置编码准备

前端由`QwenPatchProjection`、`QwenPositionInterpolation`与`VisionSpatialPooling`组成。
原始Qwen位置表为2304×1024，当前448×448输入对应28×28个patch位置。
提取冻结前端时，原始位置表经双线性插值和位置向量重排生成784×1024缓存。
图像patch特征的分组重排在每次前向中执行。

独立PT采用schema 3，保存`frontend.position_table`原始位置表。
导出时逐元素核对原始表生成的位置向量与训练缓存。
加载时在CPU上构造插值坐标、四邻点索引与权重，按原bfloat16乘加顺序生成位置向量，
供全部视频帧复用。生成的784×1024位置缓存由加载器重建。
已有模型状态字典中的键、形状和值经核验相等，旧格式的缓存前端加载路径继续支持。
来源为Qwen修订`89644892e4d85e24eaac8bacfd4f463576704203`。
原始位置表和现有patch权重均已与该修订核对。

本轮备份与核验记录位于`runs/smoke/qwen_frontend_20261116`和
`runs/archives/author_archives/t22_qwen_frontend_20261116/before`。

独立推理工程仅使用`weights/LightGen2.pt`一份权重，Qwen原始位置表同样位于这份PT中。
加载时生成的位置向量只保存在模型内存中，后续所有前向复用，无需额外的前端文件。
帧批次恢复、图像patch与位置向量的共同排列、两次空间汇聚及冻结前端权重校验已分别展开。
这些改动未添加可训练层，router、expert、global及读出头的实现保持不变。
本轮核验记录位于`runs/smoke/qwen_frontend_layout_20261116`，修改前源码与发行包保存在
`runs/archives/author_archives/t22_qwen_frontend_layout_20261116/before`。

## Global输入固定尺度候选

`tools/router_power_candidate.py`在独立候选中采用每支一个固定输入振幅尺度，
按2250条TRAIN的均方强度标定。Top2专家混合经现有振幅编码器后除以这个固定尺度，
各样本的入射能量差异可传入global传播。候选检查点采用独立架构身份
`t22_fixed_train_input_scale_candidate_v1`，由同一工具的`load_candidate`加载。
对照记录位于`runs/simulation/router_power_fixed_scale_20261117`，
包含558条逐样本预测、相位贡献、全量逐视频泄漏和CCD噪声核验。
该候选独立于`configs/selected.json`指向的发布权重。
