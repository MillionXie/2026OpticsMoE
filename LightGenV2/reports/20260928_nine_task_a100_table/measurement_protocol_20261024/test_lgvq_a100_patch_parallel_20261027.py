import sys
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
import lgvq_a100_patch_parallel_20261027 as patch
import run_lgvq_a100_patch_parallel_20261027 as entry
import run_lgvq_a100_patch_fourway_20261027 as four
from audit_lgvq_a100_patch_parallel_20261027 import validate


class PatchParallelTests(unittest.TestCase):
    def test_import_no_gpu(self):
        self.assertNotIn('torch',sys.modules)

    def test_complete_patch_partition_order(self):
        for n in (2,32,50176):
            k=patch.partition_bounds(n)
            self.assertEqual(list(range(n))[:k]+list(range(n))[k:],list(range(n)))
        for n in (0,1,3,True,2.0):
            with self.assertRaises(ValueError):patch.partition_bounds(n)
        for n in (4,32,50176):
            k=patch.partition_bounds(n,4)
            self.assertEqual([j for i in range(4) for j in range(n)[i*k:(i+1)*k]],list(range(n)))
        for n in (2,6):
            with self.assertRaises(ValueError):patch.partition_bounds(n,4)

    def test_actual_fork_join_and_output_order(self):
        log=[]
        class Stream:
            def __init__(self,name):self.name=name
            def wait_stream(self,other):log.append(('wait',self.name,other.name))
        current=Stream('front');aux=Stream('aux')
        class Tensor:
            ndim=5;device='cuda'
            def __init__(self,values):self.values=values;self.shape=(len(values),3,2,16,16)
            def __getitem__(self,key):return Tensor(self.values[key])
            def record_stream(self,stream):log.append(('record',self.values,stream.name))
        @contextmanager
        def context(stream):
            log.append(('enter',stream.name));yield;log.append(('exit',stream.name))
        def conv(value):
            log.append(('conv',value.values));return Tensor([x*2 for x in value.values])
        def cat(values,dim):
            self.assertEqual(log[-2][0:2],('wait','front'))
            self.assertEqual(dim,0);return Tensor([j for value in values for j in value.values])
        fake=SimpleNamespace(cuda=SimpleNamespace(current_stream=lambda device:current,stream=context),cat=cat)
        result=patch.parallel_convolution(fake,conv,Tensor([0,1,2,3]),aux)
        self.assertEqual(result.values,[0,2,4,6])
        self.assertEqual(log[0],('wait','aux','front'))
        self.assertEqual([v for tag,*rest in log if tag=='conv' for v in rest],[[0,1],[2,3]])
        with self.assertRaises(ValueError):patch.parallel_convolution(fake,conv,Tensor([0,1]),current)
        log.clear()
        branches=[Stream('aux'+str(i)) for i in range(3)]
        result=patch.parallel_convolution(fake,conv,Tensor(list(range(8))),branches)
        self.assertEqual(result.values,list(range(0,16,2)))
        self.assertEqual(log[:3],[('wait',b.name,'front') for b in branches])
        self.assertEqual([row[1] for row in log if row[0]=='conv'],[[0,1],[2,3],[4,5],[6,7]])
        self.assertEqual([row[2] for row in log if row[:2]==('wait','front')],[b.name for b in branches])
        with self.assertRaises(ValueError):patch.parallel_convolution(fake,conv,Tensor(list(range(8))),[aux]*3)

    def test_original_source_closure_and_runtime_restoration(self):
        baseline=entry.ORIGINAL_SOURCES();before=entry.original.ENTRY
        with self.assertRaisesRegex(RuntimeError,'test'):
            with entry.configured():
                frozen=entry.source_map()
                self.assertEqual({k:frozen[k] for k in baseline},baseline)
                self.assertEqual(len(frozen),len(baseline)+2)
                raise RuntimeError('test')
        self.assertEqual(entry.original.ENTRY,before)

    def test_fourway_closure_and_restore(self):
        baseline=entry.ORIGINAL_SOURCES();before=entry.ENTRY
        with four.configured():
            frozen=four.source_map()
            self.assertEqual({k:frozen[k] for k in baseline},baseline)
            self.assertEqual(len(frozen),len(baseline)+3)
            self.assertIn(four.TWO_WAY_ENTRY.relative_to(entry.original.REPO).as_posix(),frozen)
        self.assertEqual(entry.ENTRY,before)

    def test_original_gate_not_rewritten(self):
        source=entry.ORIGINAL_ENTRY.read_text(encoding='utf8')
        self.assertIn('candidate.worker(args,guard,2048)',source)
        self.assertIn('Guard(args.deadline,100)',source)
        self.assertIn('if not accepted_report(report)',source)

    def test_audit_requires_real_cold_forward_counts_and_original_scope(self):
        import ast
        source=entry.ENTRY.with_name('lgvq_a100_patch_parallel_20261027.py').read_text(encoding='utf8')
        call=next(n.value for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Assign)
                  and any(isinstance(t,ast.Name) and t.id=='info' for t in n.targets))
        info={k.arg:ast.literal_eval(k.value) for k in call.keywords
              if k.arg not in ('profile','fixed_equal_partitions','class_source_sha256','kernel','stride','weight_shape','weight_stride','weight_dtype','bias_dtype')}
        info.update(profile='native_Conv3d_two_stream_complete_patch_batches',fixed_equal_partitions=2,
            class_source_sha256=patch.CLASS_SHA,kernel=[2,16,16],stride=[2,16,16],
            weight_shape=[1024,3,2,16,16],weight_stride=[1536,512,256,16,1],
            weight_dtype='torch.bfloat16',bias_dtype='torch.bfloat16',
            hardware_scope='ORIGINAL_A100_250W_108SM',original_A100_reference_retained=True,
            all_original_math_and_timing_gates_retained=True,python_forward_calls=5,native_branch_calls=10,
            input_patch_counts=[50176]*5)
        validate(info)
        four_info=dict(info,profile='native_Conv3d_four_stream_complete_patch_batches',fixed_equal_partitions=4,native_branch_calls=20)
        validate(four_info,4)
        with self.assertRaises(ValueError):validate(four_info,2)
        for key,value in [('python_forward_calls',6),('native_branch_calls',12),
                          ('explicit_extra_warmup_forwards',1),('backend_policy_unchanged',False),
                          ('input_patch_counts',[50176]*4),('join_before_concat',False)]:
            with self.assertRaises(ValueError):validate(dict(info,**{key:value}))


if __name__=='__main__':unittest.main()
