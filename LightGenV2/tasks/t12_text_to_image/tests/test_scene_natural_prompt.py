import unittest
from LightGenV2.tasks.t12_text_to_image.appearance_scene_protocol import conditions,SHORT_PROFILE
from LightGenV2.tasks.t12_text_to_image.scene_natural_prompt import revise,PROFILE


class NaturalPrompt(unittest.TestCase):
    def test_every_scene_preserves_attributes_and_reference_input_metadata(self):
        for category in ['lamp','pillow']:
            for scene in conditions(category,SHORT_PROFILE):
                old=dict(scene,reference='source.png',reference_sha256='rgb',seed=42,reference_id='item')
                new=revise(old)
                self.assertEqual({k:v for k,v in old.items() if k not in ['prompt','native_caption','prompt_profile']},
                    {k:v for k,v in new.items() if k not in ['prompt','native_caption','prompt_profile']})
                self.assertNotIn('reference',new['prompt'].lower())
                self.assertNotIn('shape',new['prompt'].lower())
                self.assertEqual(new['prompt'].count('.'),2)
                self.assertEqual(new['prompt_profile'],PROFILE)
                base=old['prompt'].split('. Keep')[0].replace('the reference ','a ',1)
                place,details=base.rsplit(' with ',1)
                self.assertEqual(new['prompt'],place+'. Include '+details+' in the scene.')
                self.assertEqual(new['native_caption'],new['prompt'])
    def test_unknown_source_wording_is_not_silently_rewritten(self):
        scene=conditions('lamp',SHORT_PROFILE)[0]
        with self.assertRaises(ValueError):revise(dict(scene,prompt='arbitrary changed source'))


if __name__=='__main__':unittest.main()
