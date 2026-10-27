import ast
from contextlib import contextmanager
import copy
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import lgvq_a100_batch_replay_20261027 as candidate
import lgvq_a100_batch_raw_adapter_20261027 as adapter
import run_lgvq_a100_batch_replay_20261027 as entry


class BatchTests(unittest.TestCase):
    def test_worker_only_two_imports_change(self):
        source=inspect.getsource(entry.native.worker);tree,receipt=candidate.worker_AST(source)
        for n in ast.walk(tree):
            if isinstance(n,ast.Import) and n.names[0].name=='lgvq_a100_batch_replay_20261027':
                n.names[0].name='lgvq_async_replay_20261026'
            if isinstance(n,ast.ImportFrom) and n.module=='lgvq_a100_batch_raw_adapter_20261027':
                n.module='lgvq_async_raw_adapter_20261026'
        self.assertEqual(ast.dump(tree),ast.dump(ast.parse(source)))
        self.assertTrue(receipt['only_two_imports_redirected'])
        with self.assertRaises(ValueError):candidate.worker_AST(source.replace('import lgvq_async_replay_20261026 as replay','pass'))

    def test_adapter_exact_changes_reverse_to_original(self):
        original=ast.parse(inspect.getsource(adapter.original.prepare_records));changed=adapter.prepare_AST()
        class Reverse(ast.NodeTransformer):
            def visit_Expr(self,node):
                if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='_preparation_actions':return None
                return node
            def visit_Assign(self,node):
                if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='_child_action':
                    return ast.parse("host=action(prefix+f'.submit{i}','dispatch',e['host_submit_start'],e['host_submit_end'],field)").body[0]
                return node
        self.assertEqual(ast.dump(Reverse().visit(changed)),ast.dump(original))

    def test_shared_CPU_duration_charged_once_not_subtracted(self):
        rows=[]
        def action(name,kind,a,b,field):rows.append((name,a,b));return name
        batch=dict(start=1.,end=1.02);row=dict(body_batch_submission=batch)
        event=dict(label='original_phase',host_submit_start=1.,host_submit_end=1.02,
            host_submission_owner='one_complete_body_graph_launch',external_timing_record=True)
        for i in range(1,38):adapter.child_action(action,'f5',i,event,5,row,1)
        self.assertAlmostEqual(sum(b-a for _,a,b in rows),.02)
        self.assertEqual(sum(b>a for _,a,b in rows),1)
        with self.assertRaises(ValueError):adapter.child_action(action,'f5',2,dict(event,external_timing_record=False),5,row,1)

    def test_original_closure_and_restoration_no_torch(self):
        self.assertNotIn('torch',sys.modules)
        original=entry.core.ORIGINAL_SOURCES();before=entry.core.ENTRY
        with self.assertRaises(RuntimeError):
            with entry.configured():
                result=entry.source_map()
                self.assertEqual({k:result[k] for k in original},original)
                self.assertEqual(len(result),len(original)+5)
                raise RuntimeError('test')
        self.assertEqual(entry.core.ENTRY,before)

    def test_composition_keeps_all_37_children_and_74_external_events(self):
        calls=[]
        class Event:
            def record(self):calls.append('initial_event')
            def synchronize(self):calls.append('sync')
        class Graph:
            def capture_begin(self,**kw):calls.append(('begin',kw))
            def capture_end(self):calls.append('end')
        @contextmanager
        def stream(_):yield
        fake=SimpleNamespace(cuda=SimpleNamespace(Event=lambda **kw:Event(),stream=stream,CUDAGraph=Graph))
        segments=[dict(graph=None)]+[dict(graph=SimpleNamespace(replay=lambda i=i:calls.append(('child',i)))) for i in range(37)]
        pool=[];recorder=SimpleNamespace(record=lambda e,s:calls.append(('external',pool.index(e))))
        candidate.compose(fake,dict(segments=segments,phase_detail=True),object(),pool,recorder)
        self.assertEqual(len(pool),80)
        self.assertEqual([c[1] for c in calls if isinstance(c,tuple) and c[0]=='child'],list(range(37)))
        self.assertEqual([c[1] for c in calls if isinstance(c,tuple) and c[0]=='external'],list(range(4,78)))
        self.assertEqual(calls[-1],'end')
        self.assertLess(calls.index('sync'),calls.index(('begin',dict(capture_error_mode='global'))))

    def test_synthetic_shared_owner_replays_existing_clock_without_subtraction(self):
        # CPU-only schema fixture, NOT a new measured result or speed claim.
        import json
        path=entry.core.original.SIM/'a100_lgvq_async_20261026_10/lg_repeat01/async_raw_timing.json'
        if not path.exists():self.skipTest('Optional prior raw schema fixture unavailable')
        before=json.loads(path.read_text(encoding='utf8'));raw=copy.deepcopy(before)
        for row in raw['records'][5:]:
            children=[e for e in row['events'] if e['stream']=='body' and e['label']!='retain_current_score']
            a=children[0]['host_submit_start'];b=children[-1]['host_submit_end']
            row['body_batch_submission']=dict(start=a,end=b,ordered_child_graphs=37,
                external_device_event_pairs=37,one_host_call=True,all_shared_launch_cost_charged=True)
            for event in children:
                event.update(host_submit_start=a,host_submit_end=b,
                    host_submission_owner='one_complete_body_graph_launch',external_timing_record=True)
            if row['field']==5:
                point=row['tokenizer']['start']
                row['body_batch_builds']=[dict(sync_start=point,sync_end=point,start=point,end=point,
                    slot=i,child_graphs=37,timing_event_pairs=37,timed_initial_event_materializations=80,
                    extra_model_forwards=0,all_composition_cost_charged=True,event_record_flag='cudaEventRecordExternal') for i in (0,1)]
        trace,report=adapter.adapt(raw)
        self.assertAlmostEqual(report['actual_async_software_wall_ms'],before['actual_async_software_wall_ms'])
        self.assertTrue(report['every_observed_node_start_reproduced'])
        # Original fixture is never modified on disk or re-qualified.
        self.assertEqual(json.loads(path.read_text(encoding='utf8')),before)


if __name__=='__main__':unittest.main()
