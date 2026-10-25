import ast
from pathlib import Path
import unittest
from lgvq_hoisted_pipeline_20261026 import partition_forward,PREFIX_NAMES


class Partition(unittest.TestCase):
    def test_original_tail_unchanged(self):
        repo=Path(__file__).resolve().parents[4]
        model=repo/'LightGenV2/tasks/t06_video_quality_assessment/models/multivideo9x4.py'
        tree=ast.parse(model.read_text(encoding='utf8'))
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MultiVideo9x4OpticalVQA')
        fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
        module,receipt=partition_forward(ast.unparse(fn))
        self.assertTrue(receipt['original_tail_AST_unchanged'])
        self.assertEqual(receipt['prefix_values'],list(PREFIX_NAMES))
        tail=module.body[1]
        self.assertNotIn('language_adapter',ast.unparse(tail))
        self.assertIn('frame_merger',ast.unparse(tail))
        self.assertIn('parallel_optics.expert',ast.unparse(tail))
        self.assertIn('serial_optics.global_path',ast.unparse(tail))
    def test_partition_executes_same_arithmetic(self):
        source='''def forward(self,vision_tokens,quality_tokens,language_tokens,language_mask,*,optical_enabled=True):
    batch=1
    videos=16
    vision=vision_tokens+quality_tokens
    prompt_tokens=language_tokens*2
    prompt_mask=language_mask
    routing: dict={}
    result=vision+prompt_tokens if optical_enabled else vision
    return dict(result=result,prompt_mask=prompt_mask,batch=batch,videos=videos,routing=routing)
'''
        namespace={};exec(source,namespace);module,_=partition_forward(source);exec(compile(module,'<unit>','exec'),namespace)
        kwargs=dict(vision_tokens=3,quality_tokens=5,language_tokens=7,language_mask=1)
        for enabled in (True,False):
            prefix=namespace['prepare_native_prefix'](None,**kwargs,optical_enabled=enabled)
            self.assertEqual(namespace['forward'](None,**kwargs,optical_enabled=enabled),namespace['forward_from_prefix'](None,prefix,optical_enabled=enabled))
    def test_unrecognized_boundary_refused(self):
        with self.assertRaises(AssertionError):partition_forward('def forward(self):\n    return 3')
    def test_only_LG_and_real_online_tokenizer(self):
        source=Path(__file__).with_name('lgvq_hoisted_pipeline_20261026.py').read_text(encoding='utf8')
        ast.parse(source)
        self.assertIn("assert args.system=='lg'",source)
        self.assertIn("tok=front.tokenizer([rendered]",source)
        self.assertIn('len(token_records)==128',source)
        self.assertIn('online_CPU_tokenizer_in_continuous_window=True',source)
        self.assertIn('equal_tree(original,altered)',source)
        self.assertIn('minimal.timed(model,front,position,raw,guard',source)
    def test_full_field_constants_not_shortened(self):
        source=Path(__file__).with_name('lgvq_hoisted_pipeline_20261026.py').read_text(encoding='utf8')
        self.assertIn('video_calls=2048,unique_videos=512,fields=128,frames_per_field=64',source)
        self.assertIn("context.get('passes')==6",source)
        self.assertNotIn('torch.compile',source)


if __name__=='__main__':unittest.main()
