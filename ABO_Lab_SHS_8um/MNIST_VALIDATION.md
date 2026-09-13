# 2026-09-13 新相机位置：MNIST 验证入口

## 结论：流程已实测，识别性能尚未验收通过

本次不是 ABO 六层结果，而是原有 `d2nn_mnist4_single_layer_17um_10cm_v2`
单层光学 MNIST-4（数字0、1、2、3）。固定 post_robust_best epoch12，未训练。

|项目|结果|
|---|---|
|独立小样本|官方 test 每类第3～12张，共40张；不按预测筛选|
|同样40张仿真|38/40 = 95%|
|实际光路|24/40 = 60%|
|四类分别正确|6/10、6/10、5/10、7/10|
|实测/仿真平均全幅 PCC|0.3045628906545447|
|首张在末尾重复一次的 PCC|0.9958555028140884（重复帧不计准确率）|
|不完整帧|0|
|最大 canonical 饱和比例|0.000013130022233504315|
|曝光/增益/相机流|150μs / Gain_X4 / 100fps|
|振幅Visible后的等待|200ms内持续读帧丢弃，再丢6帧取下一帧|
|单次输入BMP检查至原始帧就绪|平均285.21ms；不含PNG保存/跨机传输/初始化|

历史 validation 88.1212%、本次小样本仿真95%、实测60%是三个不同口径。
不能把小样本95%写成全测试集结果。没有删错例，也没有为了提高分数调整读出框。

## 三者方向与物理尺寸

用户文件：师弟电脑 `E:\code\guest\20260913.txt`，原件已复制到本机
`results/mnist_markers_20260913_154939/20260913.txt`。

原文件标签按**相机画面的四角**理解：TL(552,99)、TR(1378,101)、
BL(556,916)、BR(1374,917)。四个独立振幅方块及不对称F图确认相机左右镜像。
因此模型逻辑四角在相机上的坐标为：

```json
{
  "top_left": [1378, 101],
  "top_right": [552, 99],
  "bottom_right": [556, 916],
  "bottom_left": [1374, 917]
}
```

一次 homography 直接映射到478×478逻辑CCD；镜像已经包含在四角对应关系中，
**不要再额外 flip 一次**。独立方块质心残差1.6～4.1个逻辑像素，故这是正确的
整体方向，不是精确像素共面配准的证明。证据：
`results/mnist_markers_20260913_154939/geometry.json`、`canonical_F.png`。

振幅保持原方向，1920×1080，中心(960,540)。相位1920×1200，中心(960,600)。
保持训练478×478、17μm的8.126mm物理宽度，以8μm原生像素最近邻栅格化为1016×1016。
不把训练ROI直接缩为478个8μm像素，也不在相位wrap处做双线性插值。

相位候选比较用了各类前两张中固定的第一张0和第一张1（不进入40张测量集）：
原方向/左右/上下/上下左右 × 正灰度/反灰度共8种。按两张的平均仿真PCC选择
`xy_inverse`，也与此前物理方向判断一致。它的平均PCC仅0.3157，领先幅度不大；
该结论是**当前最佳候选，不是精密相位/电压LUT标定完成**。

实际加载文件：`generated/mnist_heldout_20260913/phase_xy_inverse.bmp`。
它从逻辑训练相位重新编码，先上下+左右翻转，再物理栅格化，最后整幅255−g。
旧 `generated/phase_inverted/mnist_v2` 中中心(980,590)、仅垂直翻转的文件未覆盖，
不要与本次候选混用。师姐正常LUT的面板也不能直接使用这个反灰度候选。

## 实测数据与图

主结果目录：`results/mnist_heldout_20260913_batch/`。

- `summary.json`：完整40张指标、混淆矩阵、计时与相机参数。
- `comparison.json`：每张标签、预测、四区域原始能量、PCC、饱和比例。
- `simulation_vs_measured.png`：每类第一张测量样本，不按效果挑图；仿真/实测/四区域能量。
- `measured_contact_sheet.png`：全部40张及1张重复帧，均带探测框和预测。
- `confusion_matrix.png`：所有错误均保留。
- `<样本>.png`：原始1920×1080 Mono8 CCD；`<样本>.json`：帧状态和相机回读。
- `<样本>_canonical.npy`：一次几何映射后的float32线性强度。
- `report.json`：manifest、配置快照、phase SDK receipt、显示坐标、源码版本。

分类只对固定59×59四框求和取argmax，无逐图归一化、log、背景扣除或去噪。
框为[162,162,221,221]、[257,162,316,221]、[162,257,221,316]、[257,257,316,316]。
图中能量比例仅用于比较；不回写推理。仿真图采用一个共享显示上限；实测图固定0～255。

原始CCD及capture.json也留在师弟电脑同名工程的
`results/mnist_heldout_20260913_batch/batch/`。诊断只有41帧，保留raw用于排障；
不是要求今后几万张正式实验都保存全幅或TIFF。本轮无数据删除。

## 操作命令（本地电脑执行，远端自动显示振幅/取图）

相位SDK在本地；师弟电脑有Holoeye+SHS。两边GUI先关闭，电脑桌面保持登录。
相位SDK采用持久RGBA缓冲和同线程消息泵。用户已将Windows相位扩展屏Y设为0，
本轮无需改变显示位置。仍保持1920×1200、60Hz。

```powershell
Set-Location C:\Users\Xml12\OneDrive\2026OpticsMoE
$py = 'C:\ProgramData\anaconda3\python.exe'
# SSH密码通过SHS_SSH_PASSWORD环境变量或程序的隐藏密码提示提供，不写进文档。

# 使用固定候选采集新会话；必须换一个未存在的输出目录。
& $py ABO_Lab_SHS_8um/mnist_joint_capture.py `
  --manifest ABO_Lab_SHS_8um/generated/mnist_heldout_20260913/manifest.json `
  --out ABO_Lab_SHS_8um/results/mnist_heldout_repeat01 --batch

& $py ABO_Lab_SHS_8um/mnist_diagnostic.py evaluate `
  --run ABO_Lab_SHS_8um/results/mnist_heldout_repeat01
& $py ABO_Lab_SHS_8um/mnist_report.py `
  --run ABO_Lab_SHS_8um/results/mnist_heldout_repeat01
```

`--batch`要求全部输入使用同一个相位；一次开设备完成，不要用于不同相位混合的pilot。
镜像、ROI、曝光、相位中心更改后必须新建生成manifest及采集目录。这里不覆盖
`LAB.local.json` 的旧ABO会话，也不自动批准其 `geometry_confirmed`。
采集从师弟电脑读取该文件的曝光、增益、等待；实际回读在每帧JSON中，不保证永远是150μs。

参考输入和仿真CCD位于本机 `assets/mnist_reference_20260913/reference.npz`，
配套JSON包含来源/数据索引/SHA。重新生成BMP见：

```powershell
& $py ABO_Lab_SHS_8um/mnist_diagnostic.py prepare `
  --reference ABO_Lab_SHS_8um/assets/mnist_reference_20260913 `
  --geometry ABO_Lab_SHS_8um/results/mnist_markers_20260913_154939/geometry.json `
  --out ABO_Lab_SHS_8um/generated/mnist_repeat01 --variant xy_inverse
```

checkpoint SHA256：`e297b9baa4c028b49695daed24bb291bb71cd93a23f43e2b01ee1d87b1607887`。
参考数组SHA256：`95e148bb8a698b0afa059165e9ebe585e15a364706a6489e644040cfe8906878`。
服务器通过已推送的 `3bf31e3e` 导出脚本运行，原modeling.py与本地SHA一致；
没有更新服务器工作树或重训。实际批量控制远端源码 `edfc3c75`，本地协调源码
`d735545b`，最终说明/汇总代码版本见交付的CODE_MANIFEST。

## 尚未解决的差距

实测仍明显保留输入数字轮廓，四探测区的区分度小于仿真。当前证据说明通信/取帧
与粗方向可工作，不说明整个光学传递函数正确。

优先后续检查：相位与振幅有效区域的精细共面配准；相位灰度—相位响应及是否覆盖
模型要求的范围；振幅灰度—光场幅度响应；实际未调制分量。当前使用的是
`19x12_8bit_linearVoltage`而非实测线性相位LUT，振幅配置也未加载任务专用灰度LUT。
这些是待排查项，不能仅从本次图像断言是某一个原因。不要通过改分类框、删错例、
把相似度最高的测试子集当全集，或只增强显示图来掩盖差距。
