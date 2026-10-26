"""Pure scheduling tests. Fake events do not claim GPU timing or equivalence."""
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from lgvq_async_replay_20261026 import replay_fields,validate_slots

class Tensor:
    def __init__(self,value=0):self.value=value
    def data_ptr(self):return id(self)
    def untyped_storage(self):return self
    def pin_memory(self):return self
    def bool(self):return self
    def copy_(self,other,non_blocking=False):self.value=other.value
    def clone(self):return Tensor(self.value)

class Event:
    def __init__(self,**kw):self.waits=0;self.recorded=False
    def record(self):self.recorded=True
    def synchronize(self):self.waits+=1

class Stream:
    def __init__(self):self.dependencies=[]
    def wait_event(self,event):self.dependencies.append(event)

class Graph:
    def __init__(self,fn=lambda:None):self.calls=0;self.fn=fn
    def replay(self):self.calls+=1;self.fn()

def slot():
    s=dict(static={k:Tensor() for k in ('pixels','quality','ids','mask')},
           prefix_owned={'vision':Tensor(),'prompt':Tensor()},prediction=Tensor(),score_ready=Event())
    kinds=['frontend']+['optical_simulation']*6+['detector_ROI']*6+['electronic_residual']*4+['electronics']*17
    s['segments']=[dict(kind=k,label=str(i),graph=Graph()) for i,k in enumerate(kinds)]
    s['segments'][-1]['graph']=Graph(lambda:setattr(s['prediction'],'value',s['static']['pixels'].value))
    return s

class SchedulingTests(unittest.TestCase):
    def test_event_pool_snapshots_every_completed_field_before_overwrite(self):
        from lgvq_async_worker_20261026 import serialize_field
        class ClockEvent(Event):
            created=0;clock=0
            def __init__(self,**kw):super().__init__(**kw);ClockEvent.created+=1;self.timestamp=None
            def record(self):super().record();ClockEvent.clock+=1;self.timestamp=ClockEvent.clock
        origin=SimpleNamespace(elapsed_time=lambda e:e.timestamp)
        slots=[slot(),slot()]
        raw=[dict(pixels=Tensor(i),quality=Tensor(i)) for i in range(32)]
        torch=SimpleNamespace(inference_mode=nullcontext,cuda=SimpleNamespace(Event=ClockEvent,stream=lambda s:nullcontext()))
        snapshots=[]
        def snapshot(row):
            self.assertGreater(row['score_ready'].waits,0)
            value=serialize_field(origin,row);snapshots.append(value);return value
        result=replay_fields(torch,slots,raw,lambda f:(Tensor(f),Tensor(1),dict(field=f)),
            SimpleNamespace(fast=lambda:None),Stream(),Stream(),first_field=5,total_fields=20,
            reuse_events=True,snapshot_completed=snapshot)
        rows=[serialize_field(origin,r) for r in result['records']]
        self.assertEqual(len(snapshots),13)
        self.assertEqual(ClockEvent.created,2*72)
        self.assertEqual(result['event_reuse']['allocated_events'],ClockEvent.created)
        self.assertEqual([x.value for x in result['outputs']],list(range(5,20)))
        self.assertTrue(all(a['score_ready_cuda_ms']<b['score_ready_cuda_ms'] for a,b in zip(rows,rows[1:])))
        self.assertTrue(all('score_ready' not in r and 'original_H2D_events' not in r for r in rows))
        self.assertLess(rows[0]['events'][0]['cuda_start_ms'],rows[2]['events'][0]['cuda_start_ms'])

    def test_reuse_without_snapshot_callback_rejected(self):
        with self.assertRaisesRegex(ValueError,'snapshot'):
            replay_fields(None,[slot(),slot()],[None]*32,None,None,None,None,
                first_field=5,total_fields=10,reuse_events=True)
    def test_offset_storage_alias_is_not_independent(self):
        class View(Tensor):
            def __init__(self, root):self.root=root
            def untyped_storage(self):return self.root
        slots=[slot(),slot()]
        slots[1]['prefix_owned']['vision']=View(slots[0]['prefix_owned']['vision'])
        self.assertNotEqual(slots[0]['prefix_owned']['vision'].data_ptr(),slots[1]['prefix_owned']['vision'].data_ptr())
        with self.assertRaisesRegex(ValueError,'alias'):validate_slots(slots)

    def test_field_order_buffers_and_scores_retained(self):
        slots=[slot(),slot()];old_fences=[s['score_ready'] for s in slots]
        raw=[dict(pixels=Tensor(i),quality=Tensor(i)) for i in range(32)];calls=[]
        def tokenize(field):calls.append(field);return Tensor(field),Tensor(1),dict(field=field)
        torch=SimpleNamespace(inference_mode=nullcontext,cuda=SimpleNamespace(Event=Event,stream=lambda s:nullcontext()))
        fs,bs=Stream(),Stream();guard=SimpleNamespace(fast=lambda:None)
        result=replay_fields(torch,slots,raw,tokenize,guard,fs,bs,first_field=5,total_fields=10)
        self.assertEqual(calls,[5,6,7,8,9])
        self.assertEqual([x.value for x in result['outputs']],calls)
        self.assertEqual([r['slot'] for r in result['records']],[0,1,0,1,0])
        self.assertEqual(len(bs.dependencies),5)
        self.assertEqual([e.waits for e in old_fences],[1,1])
        self.assertEqual(result['records'][0]['score_ready'].waits,1)
        self.assertEqual(result['records'][-1]['score_ready'].waits,1)
        self.assertEqual(sum(x['kind']=='optical_simulation' for r in result['records'] for x in r['events']),30)
        self.assertTrue(all(s['static']['ids'].value in (8,9) for s in slots))
    def test_input_prefix_and_output_aliases_rejected(self):
        for key in ['static','prefix_owned','prediction']:
            slots=[slot(),slot()];slots[1][key]=slots[0][key]
            with self.assertRaisesRegex(ValueError,'alias'):validate_slots(slots)
    def test_missing_real_initial_fence_rejected(self):
        slots=[slot(),slot()];slots[0]['score_ready']=None
        with self.assertRaisesRegex(ValueError,'fence'):validate_slots(slots)
    def test_no_optical_pass_can_be_dropped(self):
        slots=[slot(),slot()];slots[0]['segments'][1]['kind']='electronics'
        with self.assertRaises(ValueError):validate_slots(slots)

if __name__=='__main__':unittest.main()
