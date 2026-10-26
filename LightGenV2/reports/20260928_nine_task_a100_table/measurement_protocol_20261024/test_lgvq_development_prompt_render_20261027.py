import ast
import unittest
from pathlib import Path
import lgvq_development_prompt_render_20261027 as prompt
import run_lgvq_development_20261027 as dev


class PromptTests(unittest.TestCase):
    def test_only_text_not_tokenizer_output_reused(self):
        calls=[];render=prompt.RenderOnce()
        for i in range(128):self.assertEqual(render(lambda:calls.append(i) or 'fixed text'),'fixed text')
        self.assertEqual((render.requests,render.renders,len(calls)),(128,1,1))
        self.assertLessEqual(*render.first_interval)
        with self.assertRaises(ValueError):prompt.RenderOnce()(lambda:[1,2,3])

    def test_only_fixed_render_statement_changes(self):
        source=dev.WORKER.read_text(encoding='utf8')
        tree=prompt.prompt_render_AST(ast.parse(source))
        line="rendered = text.render_prompt(front.tokenizer, target_name='temporal', prompt=text.TARGET_PROMPTS['temporal'])"
        self.assertEqual(source.count(line),1)
        expected=source.replace(line,"rendered = development_render_once(lambda: text.render_prompt(front.tokenizer, target_name='temporal', prompt=text.TARGET_PROMPTS['temporal']))")
        self.assertEqual(ast.dump(tree),ast.dump(ast.parse(expected)))

    def test_template_change_fails_closed(self):
        source=dev.WORKER.read_text(encoding='utf8').replace("target_name='temporal', prompt=text.TARGET_PROMPTS['temporal']","target_name='spatial', prompt=text.TARGET_PROMPTS['temporal']")
        with self.assertRaises(ValueError):prompt.prompt_render_AST(ast.parse(source))


if __name__=='__main__':unittest.main()
