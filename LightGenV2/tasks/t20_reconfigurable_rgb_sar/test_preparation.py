import unittest
from pathlib import Path
import ast


class TestSourcePreparation(unittest.TestCase):
    def test_vendor_sources_parse(self):
        for p in Path(__file__).parent.glob('reference/*.py'):
            ast.parse(p.read_text(encoding='utf-8'))

    def test_native_sources_parse(self):
        ast.parse((Path(__file__).parent/'hardware_aware.py').read_text())

    def test_aperture(self):
        self.assertEqual(2*476+64,1016)
        self.assertEqual(4*230+3*32,1016)
        self.assertAlmostEqual(1016*8e-6,0.008128)


if __name__=='__main__':unittest.main()
