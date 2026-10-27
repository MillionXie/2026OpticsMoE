from pathlib import Path
import sys
import unittest
import run_lgvq_a100_layout_20261027 as candidate


class A100LayoutTests(unittest.TestCase):
    def test_import_no_torch_or_gpu1_adaptation(self):
        self.assertNotIn('torch',sys.modules)
        source=Path(candidate.__file__).read_text(encoding='utf8')
        self.assertNotIn('development_adapter',source)
        self.assertNotIn('GPU-e8837b85',source)
        self.assertNotIn('--parallel-text',source)

    def test_closure_retains_original_entry_and_all_original_sources(self):
        baseline=candidate.ORIGINAL_SOURCES()
        with candidate.configured():
            frozen=candidate.source_map()
            self.assertTrue(set(baseline)<=set(frozen))
            self.assertEqual({k:frozen[k] for k in baseline},baseline)
            self.assertEqual(len(frozen),len(baseline)+2)
            self.assertIn(candidate.ORIGINAL_ENTRY.relative_to(candidate.original.REPO).as_posix(),frozen)

    def test_runtime_patch_restores_after_failure(self):
        before=candidate.original.ENTRY
        with self.assertRaises(RuntimeError):
            with candidate.configured():
                self.assertEqual(candidate.original.ENTRY,candidate.ENTRY)
                raise RuntimeError('test')
        self.assertEqual(candidate.original.ENTRY,before)

    def test_hardware_and_numeric_gate_are_original_not_rewritten(self):
        source=candidate.ORIGINAL_ENTRY.read_text(encoding='utf8')
        self.assertIn('Guard(args.deadline,100)',source)
        self.assertIn('candidate.worker(args,guard,2048)',source)
        self.assertIn('if not accepted_report(report)',source)
        self.assertNotIn('allclose',Path(candidate.__file__).read_text(encoding='utf8'))


if __name__=='__main__':unittest.main()
