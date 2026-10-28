"""Small CPU-only boundary, attribution and additive-schedule checks."""
import ast
from contextlib import nullcontext
import inspect
import unittest
import lgvq_compute_scope_20261029 as scope

def preprocess(self,items):
    outputs=[]
    for item in items:
        resized_height=resized_width=1
        temporal_patch_size=patch_size=merge_size=1
        patches=self.rescale_and_normalize(item)
        patches=patches+3
        flatten_patches=patches*2
        outputs.append(flatten_patches)
    return outputs

def forward_from_prefix(self,prefix,*,optical_enabled=True):
    v=prefix['value']
    image_tokens=v+1
    sequence=image_tokens*2
    fields3=sequence+3
    other=fields3*4
    normalized=other+5
    return normalized

class Checks(unittest.TestCase):
    def test_patch_native_AST(self):
        captured=[]
        observed,patch,receipt=scope.processor_adapter(preprocess,lambda *x:captured.append(x))
        class Processor:
            def rescale_and_normalize(self,x):return x*10
        self.assertEqual(observed(Processor(),[2,4]),[46,86])
        self.assertEqual(patch(20,1,1,1,1,1),46)
        self.assertEqual(len(captured),2)
        self.assertTrue(receipt['native_patch_statements_preserved'])

    def test_tail_math_preserved(self):
        class Meter:
            def scope(self,name):return nullcontext()
        annotated,_=scope.annotate_tail(forward_from_prefix,Meter(),inspect.getsource(forward_from_prefix))
        for v in range(10):self.assertEqual(annotated(None,dict(value=v)),forward_from_prefix(None,dict(value=v)))

    def test_kernel_attribution(self):
        events=[dict(name='LG|0|adapter_visual',ph='X',cat='user_annotation',ts=0,dur=100,pid=1,tid=1),
                dict(name='LG|0|emb_conv3d',ph='X',cat='user_annotation',ts=10,dur=40,pid=1,tid=1),
                dict(name='aten::conv',ph='X',cat='cpu_op',ts=20,dur=5,pid=1,tid=1,args={'External id':9}),
                dict(name='conv_kernel',ph='X',cat='kernel',ts=30,dur=10,pid=0,tid=7,args={'External id':9,'stream':7})]
        self.assertEqual(scope.traced_kernels(dict(traceEvents=events))[0]['phase'],'emb_conv3d')
        events[-1]['args']['External id']=99
        with self.assertRaises(ValueError):scope.traced_kernels(dict(traceEvents=events))

    def test_union_not_span_or_sum(self):
        self.assertEqual(scope.union_seconds([(0,3),(1,4),(8,10)]),6e-6)

    def test_shared_resource_and_additive_closure(self):
        def node(name,resource,seconds,parents=()):
            return dict(id=name,field=0,sequence=0,resource=resource,duration_s=seconds,parents=list(parents))
        r=scope.compute_schedule([node('a','gpu',2),node('b','optic',3,['a']),
                                  node('c','gpu',4,['a']),node('d','gpu',1,['b','c'])])
        self.assertEqual(r['total_s'],7)
        by={n['id']:n for n in r['nodes']}
        self.assertEqual(sum(by[k]['duration_s'] for k in r['additive_path']),7)
        self.assertGreaterEqual(by['d']['start_s'],by['b']['end_s'])
        with self.assertRaises(ValueError):scope.compute_schedule([node('x','gpu',1,['x'])])

if __name__=='__main__':unittest.main()
