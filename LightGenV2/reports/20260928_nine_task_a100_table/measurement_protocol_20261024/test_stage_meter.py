from pathlib import Path
import importlib.util
import time
import unittest

spec = importlib.util.spec_from_file_location("stage_meter", Path(__file__).with_name("stage_meter.py"))
meter_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(meter_module)


class Toy:
    def forward(self, value, *, scale=2):
        before = value + 1
        layout = [before, before]
        result = sum(layout) * scale
        return result


class Contracts(unittest.TestCase):
    def test_ast_keeps_math_and_default(self):
        toy = Toy()
        expected = toy.forward(3)
        meter = meter_module.StageMeter()
        meter.patch_statements(toy, "forward", [("neural", "neural", True, 2, 3)])
        self.assertEqual(toy.forward(3), expected)
        self.assertEqual(toy.forward(3, scale=3), 24)
        self.assertEqual(len(meter.rows), 2)
        self.assertEqual(len(meter.ledger[0]["original_statements"]), 2)
        self.assertNotIn("layout =", "\n".join(meter.ledger[0]["original_statements"]))
        meter.close()
        self.assertEqual(toy.forward(3), expected)

    def test_no_nested_double_count(self):
        meter = meter_module.StageMeter()
        with meter.scope("parent", gpu=False):
            with meter.scope("child", gpu=False):
                _ = sum(range(20))
        self.assertEqual([r["stage"] for r in meter.rows], ["parent"])

    def test_exception_is_retained(self):
        meter = meter_module.StageMeter()
        with self.assertRaises(ValueError):
            with meter.scope("failure", gpu=False):
                raise ValueError("keep the failed interval")
        self.assertEqual(len(meter.rows), 1)
        self.assertEqual(meter._depth, 0)

    def test_overlap_rejected(self):
        meter = meter_module.StageMeter()
        with self.assertRaises(ValueError):
            meter.patch_statements(Toy(), "forward", [("a", "neural", True, 0, 2), ("b", "neural", True, 1, 3)])

    def test_tail_resources_not_dropped(self):
        optical_seconds = 13 * 6 * .0010447
        energy = meter_module.optical_energy_j(optical_seconds, .02)
        c = meter_module.equivalent_cost(optical_seconds + .02, energy, 200, 4)
        self.assertAlmostEqual(c["equivalent_frame_ms"], c["sample_ms"] / 4)
        self.assertGreater(c["sample_ms"], (12.5 * 6 * .0010447 + .02) * 1000 / 200)

    def test_phase_energy_no_idle_subtraction(self):
        self.assertAlmostEqual(meter_module.optical_energy_j(.0062682, .002), .9715882772)

    def test_zero_denominator_rejected(self):
        with self.assertRaises(ValueError):
            meter_module.equivalent_cost(1, 1, 0)


if __name__ == "__main__":
    unittest.main()
