"""CPU-only checks for the frontend/token report derivation."""
import importlib.util
import math
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("frontend_token_audit", HERE / "audit_frontend_tokens.py")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class FrontendTokenAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = AUDIT.build()
        cls.rows = {r["key"]: r for r in cls.audit["rows"]}

    def test_ratio_of_sums_not_mean_of_rates(self):
        self.assertEqual(AUDIT.throughput([1, 9], [1, 3]), 2500)
        self.assertNotEqual(AUDIT.throughput([1, 9], [1, 3]), (1000 + 3000) / 2)

    def test_invalid_pairs_fail(self):
        for counts, times in [([], []), ([1], [1, 2]), ([1], [0]), ([-1], [1])]:
            with self.assertRaises(ValueError):
                AUDIT.throughput(counts, times)

    def test_temporal_matching_workload(self):
        row = self.rows["lgvq_temporal"]
        self.assertEqual(row["workload"], 16)
        self.assertEqual(row["qwen_tokens"]["language_total"], 89200)
        self.assertEqual(row["ours_input_count_reference"], 3136)
        self.assertAlmostEqual(row["qwen_tokens"]["native_rate"], 6004.782217850558)
        self.assertAlmostEqual(446 * 16 / (row["baseline_workload_ms"] / 1000), row["qwen_tokens"]["native_rate"], places=6)

    def test_varying_tokens_are_read_from_each_row(self):
        row = self.rows["abo_text_to_image"]
        self.assertEqual(row["qwen_tokens"]["language_total"], 9576)
        self.assertEqual(row["qwen_tokens"]["language_mean"], 47.88)
        self.assertEqual(self.rows["salicon"]["qwen_tokens"]["visual_mean"], 192)

    def test_missing_is_not_zero(self):
        for row in self.rows.values():
            self.assertIsNone(row["frontend_increment_ms"])
            self.assertIsNone(row["additional_nn_increment_ms"])
            self.assertFalse(row["full_gpu_frontend_remeasured"])
        self.assertIsNone(self.rows["text_to_image_generation"]["qwen_tokens"]["native_mean"])
        self.assertIsNone(self.rows["abo_text_to_image"]["ours_input_count_reference"])

    def test_bridge_absence_and_omission_are_separate(self):
        self.assertIn("未注册", self.rows["abo_image_to_image"]["bridge_status"])
        self.assertIn("不适用", self.rows["lsp"]["bridge_status"])
        self.assertGreater(self.rows["lgvq_temporal"]["bridge_ms"], 0)

    def test_text_query_three_pass_scope(self):
        row = self.rows["abo_text_to_image"]
        self.assertEqual(row["passes"], 6)  # Do not rewrite historical evidence.
        self.assertEqual(row["expanded_passes"], 3)
        self.assertAlmostEqual(row["expanded_passes"] * self.audit["per_pass_ms"], 3.1341)

    def test_paired_same_run_differences_reconcile(self):
        for row in self.audit["paired_lgvq"]:
            self.assertTrue(math.isclose(row["prepared_input_wall_ms"], row["core_wall_ms"] + row["preblock_host_difference_ms"], abs_tol=1e-6))
        combined = self.audit["paired_lgvq"][-1]
        self.assertEqual(combined["calls"], 600)
        self.assertAlmostEqual(combined["preblock_host_difference_ms"], 6.086297370493412)

    def test_exported_workbook_does_not_turn_missing_into_zero(self):
        import openpyxl
        book_path = HERE / "frontend_tokens_20261023/九任务_核心与含前端_Tokens_20261023.xlsx"
        book = openpyxl.load_workbook(book_path, data_only=True)
        old, expanded = book.worksheets[:2]
        for i, row in enumerate(self.audit["rows"]):
            o, b = 6 + 2 * i, 7 + 2 * i
            self.assertAlmostEqual(old[f"H{o}"].value, row["total_reference_ms"])
            self.assertAlmostEqual(old[f"I{o}"].value, row["ours_reference_j"])
            self.assertAlmostEqual(old[f"I{b}"].value, row["baseline_reference_j"])
            self.assertEqual(old[f"A{o}"].value.strip(), row["id"])
            for n in [o, b]:
                for col in ["F", "H", "I", "K", "L", "O", "P"]:
                    self.assertIsInstance(expanded[f"{col}{n}"].value, str)
                self.assertIsNone(book["输入依据与Bridge"][f"D{n}"].value)
        self.assertEqual(old["C31"].value, "范围不匹配")
        self.assertEqual(expanded["G14"].value, 3.1341)


if __name__ == "__main__":
    unittest.main()
