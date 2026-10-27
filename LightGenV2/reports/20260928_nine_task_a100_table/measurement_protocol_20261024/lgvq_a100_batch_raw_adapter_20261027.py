"""Explicit shared CPU submission ownership, unchanged original CUDA event clock.

One measured launch duration is charged ONCE to the first electronics segment.
Other child nodes use zero-duration shared-launch-BEGIN dependency markers,
not invented per-kernel CPU measurements. Their original external CUDA start/end
timestamps remain untouched. All new graph composition is charged, conservatively.
"""
import ast
import inspect
import math
import lgvq_async_raw_adapter_20261026 as original


def child_action(action,prefix,index,event,field,row,first):
    if field<5:
        return action(prefix+f'.submit{index}','dispatch',event['host_submit_start'],event['host_submit_end'],field)
    batch=row['body_batch_submission'];a,b=batch['start'],batch['end']
    if event['label']=='retain_current_score':
        return action(prefix+f'.submit{index}','dispatch',event['host_submit_start'],event['host_submit_end'],field)
    if (event.get('host_submission_owner')!='one_complete_body_graph_launch'
            or event.get('external_timing_record') is not True
            or event['host_submit_start']!=a or event['host_submit_end']!=b):
        raise ValueError('Actual shared launch and external event evidence missing')
    return action(prefix+f'.submit{index}','dispatch',a,b if index==first else a,field)


def preparation_actions(actions,rows,scores):
    builds=[]
    for row in rows[5:]:
        field=row['field'];batch=row.get('body_batch_submission',{})
        if (batch.get('ordered_child_graphs')!=37 or batch.get('external_device_event_pairs')!=37
                or batch.get('one_host_call') is not True or batch.get('all_shared_launch_cost_charged') is not True
                or not math.isfinite(batch['start']+batch['end']) or batch['start']>batch['end']):
            raise ValueError('Exactly one charged launch for all 37 unchanged children required')
        for build in row.get('body_batch_builds',[]):
            builds.append(build['slot'])
            if (field!=5 or build['child_graphs']!=37
                    or build['timing_event_pairs']!=37 or build['timed_initial_event_materializations']!=80
                    or build['extra_model_forwards']!=0 or build['all_composition_cost_charged'] is not True
                    or build['event_record_flag']!='cudaEventRecordExternal'
                    or not rows[4]['host_end']<=build['sync_start']<=build['sync_end']<=build['start']<=build['end']<=row['tokenizer']['start']):
                raise ValueError('Timed, finite, no-extra-forward composition evidence required')
            actions.append(dict(id=f'f{field}.batchsync{build["slot"]}',kind='sync',a=build['sync_start'],b=build['sync_end'],
                field=field,wait_for=[scores[4]],witness='actual before-composition device synchronization'))
            actions.append(dict(id=f'f{field}.batchbuild{build["slot"]}',kind='capture_electronics',a=build['start'],b=build['end'],
                field=field,wait_for=[],witness='all actual composition/event-initialization cost charged'))
    if builds!=[0,1]:raise ValueError('Exactly two independent body compositions required')


def prepare_AST():
    tree=ast.parse(inspect.getsource(original.prepare_records));changes=0
    signature=ast.dump(ast.parse("host=action(prefix+f'.submit{i}','dispatch',e['host_submit_start'],e['host_submit_end'],field)").body[0])
    class SharedHost(ast.NodeTransformer):
        def visit_For(self,node):
            nonlocal changes
            if ast.unparse(node.target)=='(i, e)' and ast.unparse(node.iter)=='body_events':
                for index,statement in enumerate(node.body):
                    if ast.dump(statement)==signature:
                        changes+=1
                        node.body[index]=ast.parse('host=_child_action(action,prefix,i,e,field,row,body_events[0][0])').body[0]
                return node
            return self.generic_visit(node)
    tree=SharedHost().visit(tree)
    if changes!=1:raise ValueError('Pinned captured/replay body submit boundary required')
    fn=tree.body[0]
    insert=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='by_action' for t in n.targets))
    fn.body.insert(insert,ast.parse('_preparation_actions(actions,rows,scores,start)').body[0])
    return ast.fix_missing_locations(tree)


def _preparation_actions(actions,rows,scores,start):
    added=[];preparation_actions(added,rows,scores)
    for item in added:
        item['a']=(item['a']-start)*1000;item['b']=(item['b']-start)*1000
    actions.extend(added)


def adapt(raw):
    namespace=dict(original.__dict__,_child_action=child_action,_preparation_actions=_preparation_actions)
    exec(compile(prepare_AST(),__file__+'::shared_host','exec'),namespace)
    before=original.prepare_records
    original.prepare_records=namespace['prepare_records']
    try:trace,report=original.adapt(raw)
    finally:original.prepare_records=before
    description=dict(steady_fields=123,ordered_children_per_launch=37,shared_launch_charged_once=True,
        zero_duration_nodes_are_shared_begin_markers_not_measured_child_CPU_calls=True,
        all_original_CUDA_events_unchanged=True,all_composition_cost_charged=True,
        no_ideal_overlap_or_queue_subtraction=True)
    trace['body_batch_profile']=description;report['body_batch_profile']=description
    return trace,report
