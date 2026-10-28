"""CPU-only evidence: no overlapping check, no swallowed failure, missing15."""
import concurrent.futures
import threading
import time
import sys
import unittest

from serialized_mps_guard_20261029 import serialized_guard_class


class Checks(unittest.TestCase):
    def test_foreground_background_do_not_overlap(self):
        class Original:
            def __init__(self):
                self.active = self.maximum = self.calls = 0
                self.audit = []

            def check(self):
                self.active += 1
                self.maximum = max(self.maximum, self.active)
                time.sleep(.02)
                self.calls += 1
                self.audit.append(self.calls)
                self.active -= 1
                return 'original snapshot'

        guard = serialized_guard_class(Original)()
        start = threading.Barrier(3)
        def run():
            start.wait()
            return guard.check()
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            a, b = pool.submit(run), pool.submit(run)
            start.wait()
            self.assertEqual([a.result(), b.result()], ['original snapshot'] * 2)
        self.assertEqual((guard.maximum, guard.calls, guard.audit), (1, 2, [1, 2]))

    def test_failure_propagates_without_retry(self):
        class Original:
            def __init__(self): self.calls = 0
            def check(self):
                self.calls += 1
                raise RuntimeError('original MPS or foreign rejection')
        guard = serialized_guard_class(Original)()
        with self.assertRaisesRegex(RuntimeError, 'original MPS or foreign rejection'):
            guard.check()
        self.assertEqual(guard.calls, 1)

    def test_remaining_plan_and_no_GPU_import(self):
        import run_baselines_serial_mps_20261029 as entry
        plan = entry.missing_plan('lab')
        self.assertEqual(plan, [('glm', 100, r) for r in (1, 2, 3)] +
            [(m, c, r) for m in ('clip', 'yolo') for c in (50, 100) for r in (1, 2, 3)])
        self.assertEqual(len(entry.RETAINED_NAMES), 6)
        with self.assertRaises(ValueError): entry.missing_plan('rental')
        self.assertNotIn('torch', sys.modules)


if __name__ == '__main__': unittest.main()
