# T12 正式硬件入口源码登记

七份采集/审计入口按 Git 提交 `f15e46a6660731d8d73d4438ceb227545566f220` 收录，与实验室 `T12_Small_Lab_SHS_8um/candidate_robust17m_20260927/source` 逐文件一致。没有启动设备或改变光学合同。

正式模型为 17M，旧 README 针对 9.96M 试采，不能沿用其模型身份或测速。原本地两份旧入口保存在提交 `ff0c7a2e1`；实验室原工程、数据、权重与全部测速未动。

正式报告位于实验室 `runs/robust17m_full_test_20260927/report.json`：2304 TEST、六层 13824 CCD，原权重 SHA256 `5b4f9a37f19ce95cf23e4b874badc8e88955559553d47929fb80cd3cbf527cac`。报告总耗时包括采集和末端推理，不是单样本延迟。

仍待 Git 收录四份后续适配辅助源码：`compare_physical_decoder.py`、`physical_decoder_boundary.py`、`run_train_capture.py`、`tune_physical_decoder.py`。本次是源码收敛，不代表硬件环境完整迁移；不要直接替换实验室运行目录。
