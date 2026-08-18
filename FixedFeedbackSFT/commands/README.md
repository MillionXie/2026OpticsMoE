# FixedFeedbackSFT commands

所有命令从仓库根目录执行。服务器仓库根目录为：

```text
/DATA/DATA1/guest3/2026OpticsMoE
```

验证 V2 核心测试、重新聚合已有 checkpoint，并输出四个结果文件的 SHA-256：

```bash
bash FixedFeedbackSFT/commands/01_verify_v2_results.sh
```

服务器默认回退到 `xml` conda 环境。其他机器可先激活项目环境，或显式指定：

```bash
PYTHON_BIN=/path/to/python bash FixedFeedbackSFT/commands/01_verify_v2_results.sh
```

脚本不会重新训练或删除 checkpoint。正式训练和单组恢复命令仍位于各实验自己的
`commands/` 目录。SSH 密码不得写入本目录或任何 Git 跟踪文件。

性能优先的新 backbone CLI 与命令目录约定见：

```text
FixedFeedbackSFT/commands/02_performance_first_runbook.md
```

该 runbook 当前是接口设计，不代表性能实验代码已经实现。
