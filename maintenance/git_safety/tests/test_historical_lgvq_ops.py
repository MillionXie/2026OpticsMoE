"""Small CPU fixtures only; never load Qwen, datasets or historical checkpoints."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
PATH = ROOT / 'LightGenV2/scripts/calculate_lgvq_topsw_ops.py'
spec = importlib.util.spec_from_file_location('historical_lgvq_ops', PATH)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


class OperationCountTests(unittest.TestCase):
    def test_server_source_identity(self):
        record = json.loads((ROOT / 'maintenance/storage/HISTORICAL_LGVQ_OPS_SOURCE_20261006.json').read_text())
        normalized = PATH.read_bytes().replace(b'\r\n', b'\n')
        self.assertEqual(hashlib.sha256(normalized).hexdigest(), record['server_original_sha256'])

    def test_linear_occurrences_and_hook_removal(self):
        model = nn.Linear(3, 2)
        row, modules = tool.count_component('small', model, lambda: model(torch.zeros(4, 3)), 2)
        self.assertEqual(row['macs_per_occurrence'], 24)
        self.assertEqual(row['macs_total'], 48)
        self.assertEqual(modules[0]['output_shape'], [4, 2])
        self.assertFalse(model._forward_hooks)

    def test_grouped_convolution(self):
        model = nn.Conv2d(4, 4, kernel_size=3, groups=2, bias=False)
        row, _ = tool.count_component('grouped', model, lambda: model(torch.zeros(1, 4, 5, 5)), 1)
        self.assertEqual(row['macs_total'], 4 * 3 * 3 * 2 * 3 * 3)

    def test_counter_activation_and_manual_attention(self):
        counter = tool.ModuleMacCounter()
        model = nn.Linear(3, 2)
        counter.install(model, 'qwen.lm_head')
        counter.active = False
        model(torch.zeros(1, 3))
        self.assertEqual(counter.rows, [])
        counter.active = True
        model(torch.zeros(1, 3))
        counter.add_manual(name='qwen.model.visual.attn', kind='attention_dense_QK_AV', macs=12)
        self.assertEqual(sum(row['macs'] for row in counter.rows), 18)
        self.assertEqual(tool.group_qwen('qwen.lm_head', 'Linear'), 'native_vocabulary_projection')
        self.assertEqual(tool.group_qwen('qwen.model.visual.attn', 'attention_dense_QK_AV'), 'vision_attention_matmul')
        counter.close()
        self.assertFalse(model._forward_hooks)

    def test_only_historical_temporal_prompt(self):
        tree = ast.parse(PATH.read_text(encoding='utf8'))
        prompts = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Attribute) and node.func.attr == 'render_prompt']
        self.assertEqual(len(prompts), 1)
        self.assertEqual(prompts[0].args[1].value, 'temporal')


if __name__ == '__main__':
    unittest.main()
