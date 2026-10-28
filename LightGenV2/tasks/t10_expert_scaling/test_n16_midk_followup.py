import unittest
from .n16_midk_followup import available,exceeds_previous


class QueuePolicies(unittest.TestCase):
    def test_sharing_but_not_own_overlap(self):
        self.assertTrue(available(6000,['foreign job'],['/our/run']))
        self.assertFalse(available(5999,['foreign job'],['/our/run']))
        self.assertFalse(available(20000,['python --out /our/run'],['/our/run']))

    def test_strict_integer_threshold(self):
        self.assertFalse(exceeds_previous([522,518,540]))
        self.assertFalse(exceeds_previous([530,518,540]))
        self.assertTrue(exceeds_previous([531,518,540]))

    def test_no_rounded_accuracy_input(self):
        with self.assertRaises(AssertionError):exceeds_previous([.71,.72,.7])


if __name__=='__main__':unittest.main()
