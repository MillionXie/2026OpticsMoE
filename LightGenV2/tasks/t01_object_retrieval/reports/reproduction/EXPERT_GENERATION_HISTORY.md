# 固定专家相位生成：主入口与历史边界

此页是T01下的历史研究索引，不替换正式DC20三系统结果，不启动任何训练。
核心源码已归同一个main；兼容模块仍在仓库
`TransferFromElectricity/tasks/t01_object_retrieval`，不是新的独立工程副本。

## 保留的两项后期研究

- 第五轮`spatial_v4`：Caltech十类、CIFAR固定十类、Imagenette-160；direct、CLIP
  视觉空间生成、Qwen视觉池化／空间生成、Qwen生成expert与global。原训练源码
  `c024b9280433f6e7fe31fc0122a1b8aadf342b38`；固定0.6融合、raw零初始化，
  4/8/18轮×120步，按validation选live权重。15组及450轮时间原位保留。
- 第六轮`unseen_v1`：CIFAR原十类之外两组新十类，5/20-shot、三次支持抽样，
  direct与Qwen视觉空间生成。原训练源码
  `0d854376cca29e16143ad1eb7762860770e59459`；仅expert/生成器更新，原电子、
  router、global冻结，固定5轮×20步末轮，不按新query选模。24组及120轮时间保留。

前面各轮文本生成、强扰动、30类扩展、同卡短计时及不利对照仍保留原run身份。
这些不是同一模型或统一选模协议，不能将其最佳精度／时间拼成一行。
原第五轮／第六轮报告、逐样本预测和best/last仍位于既有服务器runs和本机报告
目录；目前未把全部历史报告／资产纳入发布源码树，不声称从零复现已完成。

## 本次源码收敛

来源：`maintenance/storage/STATIC_EXPERT_SOURCE_IMPORT_20261006.json`，路径均相对
仓库根。64份源码／配置纳入main，63份服务器已有文件原字节不变，另补入一份
本地汇总工具；本地和实际服务器均通过29项CPU合同测试。没有模型训练、数据集
重评或GPU/设备调用。三份formal配置差异仅换行格式，明确记录SHA归一化。

两份连接工具`collect_results.py`、`collect_unseen_audits.py`和依赖它们的连接
测试未公开纳入Git，原位私有保留。CPU环境没有Paramiko，因此不把连接测试算通过。
用户原始`d2nn_pack`保留，不作为当前trainer的同等实现，也不因它未入Git而删除。

后续仍需收拢历史报告、完整基座／数据／PT依赖和运行目录用途；当前主入口只
确定源码归属和证据边界。不新建branch/worktree，不覆盖原run，不清理任何测速。
