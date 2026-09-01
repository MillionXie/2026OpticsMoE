# FixedFeedbackSFT commands

所有命令从仓库根目录执行。新布局在服务器使用独立的 clean worktree；其实际路径以
[`../RUN_REGISTRY.md`](../RUN_REGISTRY.md) 为准，不要默认在旧的脏主目录运行。

先验证目录、旧模块名兼容层与配置解析：

```bash
bash FixedFeedbackSFT/commands/00_validate_layout.sh
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

只读审计旧 checkpoint 的格式、关键字段和 tensor 统计：

```bash
python FixedFeedbackSFT/commands/03_audit_checkpoint.py /absolute/path/to/checkpoint.pt
```

将 pinned 旧 worktree 的 run 以符号链接登记到新布局（不复制、不覆盖原产物）：

```bash
bash FixedFeedbackSFT/commands/04_link_legacy_run.sh \
  /absolute/path/to/legacy/run <project-name> <run-name>
```

性能优先的新 backbone CLI 与命令目录约定见：

```text
FixedFeedbackSFT/commands/02_performance_first_runbook.md
```

该 runbook 当前是接口设计，不代表性能实验代码已经实现。
