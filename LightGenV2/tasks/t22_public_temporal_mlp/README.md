# T22 时间一致性公开替代模型试验

2026-11-11 用户批准独立试跑。本目录由本对话负责。参考仿真SRCC为0.8043868643，试验目标按绝对差不超过0.02记录，对应0.7843868643至0.8243868643。

## 当前架构

用户确认删除四条电子旁路、RMS融合和quality14，将router与expert换为MLP，将原Softplus位置改为ReLU，末端使用两层全连接评分。用户随后澄清，Qwen前端、merger等实际使用的模块要保持原运行结构。

修正版恢复原空间mean/max merger、输入prompt条件化、视觉LayerNorm和64位置的序列位置参数。模型架构身份为 `t22_serial_mlp_top2_global2_relu_original_merger_fc2`，schema为2。

逐层运行顺序如下。

1. 每视频4帧，原65%短边中心裁剪及448输入。冻结Qwen patch projection与位置编码，按原固定空间降采样得到每帧49×1024。固定Temporal prompt含38个2048维embedding。
2. 视觉适配为LN1024和Linear1024→192，文本适配为LN2048和Linear2048→192。原prompt条件化对文本token求均值，经LN192和Linear192→384形成scale/shift，执行原0.10系数调制和视觉LN192。
3. 帧router为LN192、Flatten9408、Linear9408→128、GELU、Linear128→4、softmax、Top2及选中权重L1归一化。
4. 四个帧专家分别为LN192、Linear192→384、GELU、Linear384→192，逐token执行后按Top2权重相加。四专家在帧和视频槽位之间共享，视频级使用独立的一组专家参数。
5. 帧global通过Linear192→27/ReLU和token轴Linear49→27/ReLU编码。原幅值归一化后装入518整场，执行原相位调制、传播、强度探测、区域读取、强度除均值、clip8和log1p。pool49×96、LN96、ReLU和Linear96→192得到帧特征。
6. 原Frame Merger对每帧49个token作空间mean与max，拼成384维，经过LN384、Linear384→192和GELU，再加原帧位置编码。
7. 四个图像token与38个prompt token拼接，加原序列位置编码前42项。视频router只读四个图像token，经LN192、Flatten768、Linear768→128、GELU、Linear128→4和Top2。
8. 四个视频专家逐token处理42个图文token并按Top2加权汇合，每个专家均为LN192、Linear192→384、GELU、Linear384→192。
9. 视频global通过Linear192→56/ReLU编码，补零至56行，执行原幅值归一化、整场传播和探测。pool42×96、LN96、ReLU与Linear96→192输出42×192。
10. 末端Flatten8064、Linear8064→1024、GELU和Linear1024→1得到评分，再还原MOS尺度。

空间mean/max属于原merger。末端时序卷积和均值、方差、差分统计已按用户要求删除。每场16视频，每视频对应一个输出。

两级global保持原17μm采样、532nm波长、10cm传播、518画布、478有效区、0.5°截止和111×111每视频相位。评测相干漏光系数为0.20，相位为2π·sigmoid(raw)。零ReLU场在平方均值上先clamp再开方，保证反传有限。

## 初始化和独立PT

`front.py` 定向抽取Qwen patch与position参数，保存固定prompt embedding，结果约4.9MB。独立推理加载 `LightGen.pt` 中的配置、state_dict、前端buffer、固定prompt和MOS尺度。

`--paper-initialize` 使用显式白名单导入原模型保留的适配层、输入条件化、merger、位置参数和两级global。每个导入张量记录原名、目标名，原PT的SHA写入run。新MLP router、expert和末端评分层单独初始化。

提取的前端参数与原缓存逐张量SHA一致。随机图像和实际视频上，独立前端与官方小组件的CPU输出均逐位相同。该视频CPU输出与历史CUDA缓存最大绝对差0.03125、平均绝对差约0.000201，原始核查记录已保留。

## 数据和评价

沿用2250 TRAIN与558 TEST，周期TEST评价用于选模，记录为开发评价。TRAIN标签用于反传。既有缓存仅向模型提供Qwen视觉特征。padding参与整场传播，从损失与指标中剔除。

原T06训练还使用了权重3.0的TRAIN教师软标签SmoothL1监督。`trial_original_merger_relu_trainsoft_s163.json` 复用此设置，严格校验教师sample_ids等于2250个TRAIN ID集合，将软标签列按名称绑定到temporal。教师文件的SHA与来源写入run。该设置仅影响训练损失。

每个run保留配置、源提交、文件SHA、环境、样本ID、逐样本预测、指标和路由统计，保存best与last。完整训练后执行16次循环换位审计并生成独立 `LightGen.pt`。

## 已有试验

早期Flatten merger原型按对应源提交封存。

| run | merger | 结果 |
|---|---|---|
| relu_s163_local_20261111 | Flatten | 80epoch完成，SRCC 0.6028183747 |
| relu_clean_s163_local_20261111 | Flatten | epoch25最佳0.6731984861，epoch28保存last时I/O异常 |
| relu_clean_refine_s163_local_20261111 | Flatten | 120epoch完成，最佳0.7354285207 |
| relu_clean_refine_s163_server_20261111 | Flatten | 120epoch完成，最佳0.7330343156 |

保存已改为临时文件写完并flush/fsync后原子替换。中断run保留，后续在新run记录父PT和新优化器身份。

修正版使用 `configs/trial_original_merger_relu_s163.json`，训练结果在完成后补入。

## 运行

`python -m unittest LightGenV2.tasks.t22_public_temporal_mlp.tests.test_contract -v` 检查拓扑、原merger结构、Top2、零场梯度、两级global梯度、PT重载和原子保存。

`paths.local.json` 指定vision_cache、language_cache、manifest和提取后的front。执行 `python run.py --paths paths.local.json --config configs/trial_original_merger_relu_s163.json --paper-initialize <原T06 PT> --out runs/simulation/<run_id> --device cuda` 训练。

`wait_for_gpu.py` 接受任意空闲CUDA卡，要求至少6GiB空闲显存、无其他计算进程、利用率不超过5%，连续三次检查后启动。

独立视频推理使用 `python infer.py --weights <LightGen.pt> --videos <video1.mp4> <video2.mp4> --output result.json --device cuda`。

本轮内部试验源代码通过私有Git bundle同步服务器，只取出本任务路径。公开GitHub远端的push留待交付确认。
