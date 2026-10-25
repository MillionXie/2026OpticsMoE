import json,unittest
from pathlib import Path
from .refine_LG_D_speech_development import validate


class SpeechDevelopmentTests(unittest.TestCase):
    def test_one_step_scope_and_disclosure_are_immutable(self):
        cfg=json.loads(Path(__file__).with_name('config_LG_D_speech_development_20261026.json').read_text());validate(cfg)
        for key,value in [('maximum_steps',2),('SDK',True),('GPU',True),('valid_for_formal_test_table',True)]:
            with self.assertRaises(ValueError):validate(dict(cfg,**{key:value}))


if __name__=='__main__':unittest.main()
