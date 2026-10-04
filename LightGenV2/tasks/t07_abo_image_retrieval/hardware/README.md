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
- **尚未完整迁移逐层采集/断点恢复runner。**不能仅有此模块就删旧工程或宣称
  main已经可替换实验室正式入口。原14400有效CCD、收据、权重和结果仍原样保留。

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
LGVQ视频项目或旧DVP工程。JSON相对设备路径按其所在目录解析，SDK/LUT/数据不进Git。
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
