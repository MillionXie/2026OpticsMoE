"""Pure scheduling / original-AST checks; no model or GPU import/execution."""
import ast
from pathlib import Path
import subprocess
import unittest

HERE=Path(__file__).resolve().parent
ENTRY=HERE/'lab_corrected_energy_triples_20261026.py'
REPO=HERE.parents[3]
PUBLISHED='95bcdacc9062f1ec716c3694dd31a84590419a5e'


def function(tree,name):
    return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)


tree=ast.parse(ENTRY.read_text(encoding='utf-8-sig'))
namespace={}
exec(compile(ast.fix_missing_locations(ast.Module(body=[function(tree,'point_plan')],type_ignores=[])),str(ENTRY),'exec'),namespace)
plan=namespace['point_plan']


class OrderTests(unittest.TestCase):
    def test_default_same_original_24_order(self):
        self.assertEqual(plan(['qwen','glm','clip','yolo']),[(m,c,r) for m in ('qwen','glm','clip','yolo') for c in (100,50) for r in (1,2,3)])

    def test_reorder_keeps_exact_missing_fifteen(self):
        retained={f'qwen_cap{c}_repeat{r:02d}' for c in (100,50) for r in (1,2,3)}
        retained|={f'glm_cap100_repeat{r:02d}' for r in (1,2,3)}
        result=plan(['clip','yolo','glm','qwen'],retained)
        self.assertEqual(len(result),15)
        self.assertEqual(result[:6],[(m,c,r) for m in ('clip',) for c in (100,50) for r in (1,2,3)])
        self.assertEqual(result[-3:],[('glm',50,r) for r in (1,2,3)])
        self.assertEqual(set(result),set(plan(['qwen','glm','clip','yolo'],retained)))

    def test_invalid_order_or_retained_identity_rejected(self):
        for order in (['clip','yolo','glm'],['clip','yolo','glm','glm'],['clip','yolo','glm','bad']):
            with self.assertRaises(ValueError):plan(order)
        with self.assertRaises(ValueError):plan(['qwen','glm','clip','yolo'],['qwen_cap50_repeat04'])

    def test_original_worker_guard_admission_and_sources_AST_unchanged(self):
        relative=ENTRY.relative_to(REPO).as_posix()
        old=ast.parse(subprocess.check_output(['git','show',PUBLISHED+':'+relative],cwd=REPO))
        for name in ('worker','diagnostic_guard_check','admission_snapshot','wrap_tokenizers','sources'):
            self.assertEqual(ast.dump(function(old,name),include_attributes=False),
                             ast.dump(function(tree,name),include_attributes=False),name)

    def test_original_suite_body_only_changes_plan_metadata(self):
        relative=ENTRY.relative_to(REPO).as_posix()
        old=ast.parse(subprocess.check_output(['git','show',PUBLISHED+':'+relative],cwd=REPO))
        original=function(old,'suite');current=function(tree,'suite')
        original_protected=[n for n in original.body if isinstance(n,ast.Try)]
        current_protected=[n for n in current.body if isinstance(n,ast.Try)]
        self.assertEqual(ast.dump(ast.Module(body=original_protected,type_ignores=[]),include_attributes=False),
                         ast.dump(ast.Module(body=current_protected,type_ignores=[]),include_attributes=False))


if __name__=='__main__':unittest.main()
