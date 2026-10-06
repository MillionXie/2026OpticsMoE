"""AST contracts only; do not import pyplot or open archived data."""
import ast
import hashlib
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[3] / 'LightGenV2/tasks/t12_text_to_image/perceptual/build_annotated_figures.py'


class FigureTests(unittest.TestCase):
    def test_scientific_functions_unchanged_except_exclusive_output(self):
        text = SOURCE.read_text(encoding='utf8').replace(
            '    with output.open("xb") as handle:\n        fig.savefig(handle, format="png", dpi=120)',
            '    fig.savefig(output, dpi=120)')
        expected = {
            'load_csv': '5dd659958c359ec231d46ae05cd3f3ec1cc0bf61a17fd30e4b3d52dee2689dff',
            'png': 'a465162e15f8e469bc966f7ca46d953d0ff244c95768c25b337da83b10600a4e',
            'score_text': '8f326dea4f0ed44243f42abe32f9b6e4dfd4add537e44f59cc05c52baa8477e8',
            'title_text': '91358a61befce45befeac0d36f0c08d494593702f948559278046302471f497b',
            'make': 'f4cd6563c42baf5884801864b5f5e525cb4a6ea027bd6a8bd859fac32ab7857b',
        }
        functions = {n.name: n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}
        for name, sha in expected.items():
            self.assertEqual(hashlib.sha256(ast.dump(functions[name], include_attributes=False).encode()).hexdigest(), sha)

    def test_no_top_level_asset_reads(self):
        tree = ast.parse(SOURCE.read_text(encoding='utf8'))
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                self.assertFalse(any(isinstance(n, ast.Call) for n in ast.walk(node)))

    def test_new_output_and_resource_release(self):
        text = SOURCE.read_text(encoding='utf8')
        for contract in ('if OUT.exists():', 'exist_ok=False', 'output.open("xb")', 'with ExitStack() as stack:'):
            self.assertIn(contract, text)


if __name__ == '__main__':
    unittest.main()
