# 封存 rank72：六层离线回放核心

本目录的 `replay.replay_batch` 是从已验证 Windows 最终流程提取的计算核心，
不导入相机、DVP、SLM 或旧独立工程，不启动设备，也不默认评估800查询。
封存权重仍是 SHA256
`25f23260864a00a0f32b5de209c27f5876166018c8d8ab0e880de81e0d2c2c22`。
它保留视觉和语言的 router→expert→global 六次注入、原块内alpha和
rank72 RGB patch旁路的末端图像token .5/.5混合。

调用方必须提供严格加载并置为CPU/eval的模型、原processor产生的batch、
对应唯一sample IDs，以及 `capture_stage(stage, bounded_active_amplitude, ids)`。
回调返回 `(CPU CCD tensor, receipt, amplitude_scale)`，CCD必须保持原线性强度，
不得按帧归一化；实际振幅的保零有界编码使用原 `standalone.bounded_export`。
回调不由本模块创建，设备独占、收据/相位/曝光/增益/ROI/身份校验仍由上层负责。

## 已核验与未完成边界

- 原Windows计算主体逐句AST核对：只移除路径、processor及设备全局包装，
  capture调用改为显式回调；六层计算和旁路混合逻辑不变。
- 封存PT在训练服务器从候选Git树严格CPU加载；一个合成输入的六层理想CCD桥接
  描述子最大误差为0，两次回放逐位相同。没有重评原图或查询集。
- 原 `abo_full_query_flow_snapshot_20260928.py` 的旧 `main()` 未采用：其CUDA、
  可选模拟图库和默认曝光不能被当作最终实拍入口。
- 离线核心提取时尚无完整runner；后续逐层编排已收拢，见本页末节。仍不能仅凭
  离线测试删旧工程或宣称main已现场验收。原14400有效CCD、收据、权重和结果仍原样保留。

源码身份审计（不需要私有恢复对象，不打开设备）：

```bash
python maintenance/git_safety/check_t07_replay_source.py --commit main
```

维护机持有原恢复记录时加 `--audit-archive`，另核对原计算图。
合成桥接需已核验的私有封存PT和兼容PyTorch环境；不需要原图/CCD：

```bash
python maintenance/git_safety/check_t07_hardware_replay.py --repository /path/to/2026OpticsMoE --commit main --checkpoint /path/to/rank72_epoch13_best_snapshot.pt
```

来源及边界见 `maintenance/storage/T07_OFFLINE_REPLAY_SOURCE_20261004.json`。
这不是训练或正式检索命令，也不改变最终版指标。

## 只读 CCD 收据和断点检查

`ccd_store.CCDStore` 读取已存在的正式会话。默认独立绑定封存会话的
contract SHA `bdc0d96745c144bd5dbe2865534ff40aceb9d4d3764e3d7b506f2ddfaed07df9`，
核对六层相位文件、原几何、400µs/Gain_X4/wait240、hv_inverse/flip_v、
样本身份、Mono8尺寸，以及图片与收据的亮度统计。成对缺失标为待补，
单边缺失、额外身份、暗帧或合同不一致直接报错，不擅自修复、移动或重拍。

```bash
python -m LightGenV2.tasks.t07_abo_image_retrieval.hardware.ccd_store --run /path/to/layerwise_selected2400_20260929
```

仅审计存储，不加载权重、不执行推理或计算检索指标。
20项合成测试和一张正式TRAIN样本的六层PNG/收据抽查通过。
2026-10-04追加对封存Windows会话的完整只读核验：六层各2400张、14400张PNG
全部解码并与各自收据的mean/p99/maximum/饱和比例一致；2400个样本身份和六层
相位、400µs/Gain_X4/wait240合同一致，六层最低p99为130/81/82/38/36/36。
28807份PNG／收据／相位／合同的当前内容哈希已记录，摘要见仓库
`maintenance/storage/T07_FULL_PHYSICAL_CONTENT_IDENTITY_20261004.json`。
未重评检索精度、加载模型或打开设备；这不等于新硬件回归或runner迁移完成。
原收据没有PNG哈希，当前计算的SHA只标识当前文件，不能倒称采集当时已有哈希。
临时振幅BMP当时已清除，因此只检查其收据SHA格式，不冒称可从收据重新验证原BMP。
`read(stage, ids)` 返回未做归一化的uint8数组，供明确选择的上层回放；
它本身不会调用模型或设备。

## 显式配置的原SHS控制层

`bench.SHSBench` 与 `geometry` 已收拢最终版实际使用的控制层，显式传入
`machine_config`（原Controller JSON）、`phase_sdk` 和 `phase_lut`，不再自动寻找
LGVQ视频项目或旧DVP工程。JSON中振幅SDK及binary路径按其所在目录解析；
相机sdk_root仍按原Camera行为相对进程工作目录解析，建议显式绝对路径。
SDK/LUT/数据不进Git。
若现有机器JSON的相对振幅SDK实际位于另一工程，须显式传入
`--amplitude-sdk /absolute/path/to/holoeye_python`。它只覆盖内存中的
`amplitude_slm.sdk_path`，不修改机器JSON；默认仍按JSON所在目录解析，不猜测旧工程。
2026-10-05此路径修复与原编排合计28项无设备测试通过，未启动真实SDK或采集。
七个原几何函数和Bench的进入／退出／捕获方法AST一致；SHS ROI、1016像素原生映射、
400µs/Gain_X4/wait240、相位更新后fresh、持续清帧与1%饱和守卫保持原含义。
九项纯CPU模拟设备测试通过，未打开SDK，未替换实验室代码；记录见仓库
`maintenance/storage/T07_BENCH_SOURCE_IMPORT_20261005.json`。
它仍不是完整逐层接续CLI，阶段编排、前端／protocol路径和运行目录切换继续审计。

## 逐层编排源码已收拢，尚未替换正式设备入口

`layerwise` 现已收拢实际最终runner：显式指定封存PT及SHA、processor目录、
protocol、图片根目录、geometry、运行目录和设备配置／SDK／LUT，不再引用外部
ABO工程。默认 `--mode inspect` 只核验路径、2400身份及固定几何，不加载Torch、
processor或SDK，不创建输出目录。已完成会话只能inspect，其他模式拒绝重新启动。

原六层外循环、四样本CPU回放、仅缺失帧打开阶段SDK、每层两暖帧及原检索评价函数
保持原逻辑；接续帧改用已审计CCDStore严格核对收据与像素统计。
16项新CPU前检／模拟编排测试通过，与控制层合计25项；**没有打开设备、
重评800查询或替换实验室正式源码**。不能据此删除旧工程。正式设备回归／Git切换
仍须单独安排，并且当前用户封存状态不允许启动selftest/pilot/full。

源码SHA与三项原函数AST检查：

```bash
python maintenance/git_safety/check_t07_layerwise_source.py --commit main
```

持有私有恢复记录的维护机可加 `--audit-archive`。来源记录见
`maintenance/storage/T07_LAYERWISE_SOURCE_IMPORT_20261005.json`。
新入口的 `full` 是完整真图库和查询的历史计算逻辑，不是清理工作的测试命令；
不允许为了整理重算封存指标。原800查询仍属于开发期指标，非独立泛化。

## 只读机器路径前检

从唯一主仓库运行`maintenance/git_safety/check_t07_machine_paths.py`，显式指定
`--machine-config`、`--phase-sdk`、`--phase-lut`，必要时指定`--amplitude-sdk`。
它按真实运行代码区分路径基准：JSON振幅路径相对配置目录，相机sdk_root、显式
振幅覆盖及相位SDK/LUT相对进程工作目录。检查相机DLL/CTI、振幅Python wrapper/原生DLL、相位wrapper
与LUT，并记录当前文件SHA。缺失返回2；只读，不打开SDK或创建采集目录。
报告区分原JSON参数与rank72 Bench实际覆盖的400µs/Gain_X4/wait240，不修改原配置。
七项模拟路径测试通过，包括相机相对路径及显式振幅覆盖的不同工作目录；
只修正前检、不改变原采集实现或封存配置。空SDK目录不算通过；存在性不证明
SDK授权、ABI、交互桌面、有效光信号或现场回归。
实验室已直接从发布Git对象执行只读核验：原LGVQ机器JSON的相对振幅SDK缺失，
显式绑定原ABO安装的SDK后八项路径全部存在，六项文件SHA已记录，原JSON未改变。
见 [`机器路径收据`](../../../../maintenance/storage/T07_MACHINE_PATH_GATE_20261006.json)。
实际旧工程仍保留；这不授权重采、不证明设备回归，也不等于实验室运行代码已切main。

## 主仓库的封存资产绑定检查

实验室原路径集中在 `../configs/lab/rank72_windows_20261007.json`。
该配置不替换权重、机器JSON或旧会话，且只允许inspect；从仓库根目录运行：

```powershell
python -m LightGenV2.tasks.t07_abo_image_retrieval.hardware.inspect_binding --binding LightGenV2/tasks/t07_abo_image_retrieval/configs/lab/rank72_windows_20261007.json --load-cpu
```

检查原PT、protocol及geometry SHA、2400图片身份和SDK文件路径，随后可选严格CPU加载
封存模型与本地processor。不运行图片推理、检索评价或SDK，不创建输出、不修改机器配置。
缺少原资产或依赖即失败，不能把存在性检查当成硬件现场验收。
