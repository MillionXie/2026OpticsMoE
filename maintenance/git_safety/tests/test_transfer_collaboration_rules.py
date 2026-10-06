"""Guard the published collaboration correction, not scientific model behavior."""
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[3]

class TransferCollaborationRules(unittest.TestCase):
    def test_no_automatic_isolation_instructions(self):
        for relative in ('TransferFromElectricity/README.md','TransferFromElectricity/AI_RULES.md'):
            content=(ROOT/relative).read_text(encoding='utf8')
            self.assertIn('AGENTS.md',content)
            self.assertIn('未经用户明确许可',content)
            self.assertNotIn('使用独立分支/必要时隔离 worktree',content)
            self.assertNotIn('训练从独立、固定 GitHub commit 的 worktree 启动',content)

    def test_existing_runs_and_assets_protected(self):
        rules=(ROOT/'TransferFromElectricity/AI_RULES.md').read_text(encoding='utf8')
        self.assertIn('禁止删除用户已有 run、缓存或未提交改动',rules)
        self.assertIn('不修改原始 d2nn_pack',rules)
        self.assertIn('Git事务串行',rules)

if __name__=='__main__':unittest.main()
