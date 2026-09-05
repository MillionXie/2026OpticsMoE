# T06 dataset contract

当前正式结果使用 LGVQ：训练 2250 个视频、测试 558 个视频，不单独划分验证集；
每 5 epoch 在 test 上评估并按最高 Temporal SRCC 选择候选。论文使用其他数据集时，
在本目录新增 manifest/config，不要把原视频复制进仓库。

36 帧缓存必须由每个原视频均匀采样 36 个真实帧后，经冻结 Qwen3-VL-2B-Instruct
前端生成；不能从 16 帧缓存插值。Temporal prompt 为：

> Please evaluate the temporal quality of this video and rate it using one of the following five levels: Excellent, Good, Fair, Poor, or Bad.

大文件仍位于旧正式实验的 `artifacts/`，新 profile 会冻结为绝对路径再启动任务。
