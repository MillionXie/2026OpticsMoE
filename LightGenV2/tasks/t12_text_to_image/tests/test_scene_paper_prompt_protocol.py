import unittest
from ..scene_paper_prompt_protocol import condition,PROFILE


class PaperPrompts(unittest.TestCase):
    def test_each_second_sentence_adds_a_distinct_scene_entity(self):
        anchors={'lamp':['bookcase','bed','window'],'pillow':['plant','bookcase','table']}
        for category,words in anchors.items():
            for i,word in enumerate(words):
                q=condition(category,i);first,second,_=q['prompt'].split('.')
                self.assertIn(word,second.lower());self.assertNotIn(word,first.lower().split())
                self.assertEqual(q['prompt_profile'],PROFILE);self.assertEqual(len(q['required_visible_facts']),3)
    def test_no_reference_or_repeated_show_together_instruction(self):
        for category in ['lamp','pillow']:
            for i in range(3):
                q=condition(category,i);self.assertEqual(q['prompt'].count('.'),2)
                self.assertNotIn('together',q['prompt']);self.assertNotIn('reference',q['prompt']);self.assertLessEqual(len(q['prompt'].split()),30)


if __name__=='__main__':unittest.main()
