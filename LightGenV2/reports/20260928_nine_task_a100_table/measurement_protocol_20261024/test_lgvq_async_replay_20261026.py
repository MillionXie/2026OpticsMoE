"""Pure scheduling tests. Fake events do not claim GPU timing or equivalence."""
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from lgvq_async_replay_20261026 import replay_fields,validate_slots

class Tensor:
    def __init__(self,value=0):self.value=value
    def data_ptr(self):return id(self)
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
