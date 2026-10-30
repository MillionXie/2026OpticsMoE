import unittest
from ..scene_compact_candidates import selected_edges


class CompactPathProtocol(unittest.TestCase):
    def test_full_original_teacher_grid_not_regrouped_twelve_grid(self):
        self.assertEqual(selected_edges(30,6),[0,5,10,15,20,25,30])
        self.assertEqual(selected_edges(30,8),[0,3,7,11,15,18,22,26,30])
        for steps in (6,8):
            edges=selected_edges(30,steps)
            self.assertEqual(len(edges),steps+1)
            self.assertTrue(all(x<y for x,y in zip(edges,edges[1:])))

    def test_unknown_or_unvalidated_teacher_budget_rejected(self):
        for teacher,steps in [(12,6),(30,4),(30,7),(8,8)]:
            with self.assertRaises(ValueError):selected_edges(teacher,steps)


if __name__=='__main__':unittest.main()
