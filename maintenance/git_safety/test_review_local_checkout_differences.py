import unittest
from maintenance.git_safety.review_local_checkout_differences import compare

class StructuralReviewTests(unittest.TestCase):
    def test_comments_and_line_numbers_not_changes(self):
        report=compare('# note\n\ndef run():\n    return 1\n','def run():\n    return 1\n')
        self.assertTrue(report['module_ast_identical'])
        self.assertEqual(report['different'],[])

    def test_actual_function_change(self):
        report=compare('def run():\n    return 2\n','def run():\n    return 1\n')
        self.assertFalse(report['module_ast_identical'])
        self.assertEqual(report['different'],['run'])

    def test_main_guard_stable_after_line_shift(self):
        report=compare('\n\nif __name__ == "__main__":\n    run()\n','if __name__ == "__main__":\n    run()\n')
        self.assertTrue(report['module_ast_identical'])
        self.assertEqual(report['local_only'],[])

    def test_import_order_not_runtime_equivalence(self):
        report=compare('import first\nimport second\n','import second\nimport first\n')
        self.assertEqual(report['different'],[])
        self.assertFalse(report['module_ast_identical'])

if __name__=='__main__':unittest.main()
