import hashlib
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[3]
TASK = ROOT / 'LightGenV2/tasks/t07_abo_image_retrieval'


class NavigationTests(unittest.TestCase):
    def test_historical_reproduction_body_preserved(self):
        text = (TASK / 'reports/reproduction/README.md').read_text(encoding='utf8')
        head, historical = text.split('<!-- preserved-historical-body-begins -->', 1)
        original = '# T07 复现说明入口\n\n' + historical.lstrip('\n')
        self.assertEqual(hashlib.sha256(original.encode()).hexdigest(),
                         '8c9ff592d46f2f4af327ee131e3e5c9082816bf45f4e7efe4f9b3737086804da')
        self.assertIn('800查询／1600真实图库', head)
        self.assertIn('只在任务主页维护', head)

    def test_current_navigation_links_resolve(self):
        for relative in ['README.md', 'reports/reproduction/MAIN_ENTRY_DRAFT_20261004.md',
                         'reports/reproduction/README.md']:
            path = TASK / relative
            text = path.read_text(encoding='utf8').split('<!-- preserved-historical-body-begins -->')[0]
            for link in re.findall(r'\[[^\]]+\]\(([^)]+)\)', text):
                if '://' not in link:
                    self.assertTrue((path.parent / link.split('#')[0]).is_file(), (relative, link))

    def test_sealed_identity_and_metrics_unchanged(self):
        identity = json.loads((TASK / 'reports/reproduction/FINAL_IDENTITY_20261002.json').read_text())
        self.assertEqual(identity['checkpoint']['sha256'], '25f23260864a00a0f32b5de209c27f5876166018c8d8ab0e880de81e0d2c2c22')
        self.assertEqual(identity['reports'][0]['sha256'], '94947127262ead3fc440b5e67b513f0038949b25795c5be0b916305d251e6eef')
        page = (TASK / 'README.md').read_text(encoding='utf8')
        self.assertIn('| 未微调实拍 R@1 / R@5 | .81125 / .94125 |', page)
        self.assertIn('专属测速', page)


if __name__ == '__main__':
    unittest.main()
