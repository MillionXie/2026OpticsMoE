import unittest
from rental_energy_missing_20261027 import missing_plan,EXPECTED


class MissingTests(unittest.TestCase):
    def test_exact_seven_and_no_repeat_of_saved_five(self):
        saved=[dict(model=m,cap=50,repeat=r) for m,rounds in (('kimi',(1,2,3)),('deepseek',(1,2))) for r in rounds]
        plan=missing_plan(saved)
        self.assertEqual(plan[0],('deepseek',50,3));self.assertEqual(len(plan),7)
        self.assertEqual(set(plan)|{(r['model'],r['cap'],r['repeat']) for r in saved},EXPECTED)
    def test_all_complete_is_empty(self):
        self.assertEqual(missing_plan([dict(model=m,cap=c,repeat=r) for m,c,r in EXPECTED]),[])
    def test_duplicate_and_unknown_rejected(self):
        for rows in ([dict(model='kimi',cap=50,repeat=1)]*2,[dict(model='qwen',cap=100,repeat=1)]):
            with self.assertRaises(ValueError):missing_plan(rows)


if __name__=='__main__':unittest.main()
