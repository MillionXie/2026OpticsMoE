"""Explicit accepted middle8 native capture/replay adapter; no SDK in this entry.

No new PT is bundled or accepted by this source. Production modes require an
accepted SHA-bound handoff and an independently admitted camera session.
"""
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from .scene_lab_middle8 import F6,PAIRS,TEST,checked_binding,checked_state,check_data,require_training
from .scene_lab_protocol import Ledger,STAGES,NeedCapture,open_ledger,put,sha


def deployment_environment():
    from .scene_lab_middle8 import hex_sha
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d':
        raise ValueError('Only previously authorized deployment GPU1, recheck availability before launch')
    if not hex_sha(os.environ.get('T12_SOURCE_COMMIT'),40):
        raise ValueError('Explicit published execution-source commit required')


def load_head(checkpoint, report, binding):
    checked_binding(binding)
    if sha(checkpoint)!=binding['generator_sha256']:raise ValueError('Accepted PT SHA changed')
    import torch
    from .qwen3_scene_conditioning import QwenSDXLConditions
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    checked_state(state,report,binding)
    head=QwenSDXLConditions(joint_image_context=False)
    head.load_state_dict(state['head'],strict=True)
    head.eval().requires_grad_(False)
    return head,state


def initialize(ledger_root, session, binding, data, role):
    checked_binding(binding)
    rows=check_data(data)[role]
    session=dict(session,cohort_role=role,request_manifest_sha256=PAIRS if role=='train'else TEST,
                 TEST_gradients=False,evaluation_only=role=='test',TEST_used_for_model_selection=role=='test')
    models=dict(F6=F6,generator=binding['generator_sha256'],source_commit=binding['source_commit'])
    return Ledger(ledger_root,session,models,rows,create=True,middle8_binding=binding)


class ComposedObserver:
    """Install the source depth graph FIRST, then actual-CCD supply, never overwrite either."""
    def __init__(self,graph,measured=None):self.depth=graph;self.measured=measured

    def prepare(self,qwen,processor,head,pipe):
        processor=self.depth.prepare(qwen,processor,head,pipe)
        if self.depth.graph['backend']['actual_blocks']!=64:
            raise ValueError('Middle8 must really execute the64-block UNet')
        if self.measured is not None:
            processor=self.measured.prepare(qwen,processor,head,pipe)
        return processor

    def begin(self,row):
        self.depth.begin(row)
        if self.measured is not None:self.measured.begin(row)

    def end(self,row):
        if self.measured is not None:self.measured.end(row)
        self.depth.end(row)

    def close(self):
        if self.measured is not None:self.measured.close()


def measured_observer(ledger, binding, manifest_sha):
    # Reuse the original exact ticket/prefix/array/provenance callbacks, not
    # the original F9-only admission constructor or its128-request assumption.
    from .scene_lab_render import MeasuredObserver
    class MiddleMeasuredObserver(MeasuredObserver):
        def __init__(self):
            if (ledger.middle8_binding!=binding or ledger.session.get('scope')!='reviewed_physical'
                or ledger.session.get('cohort_role')!='test' or len(ledger.requests)!=500
                or manifest_sha!=TEST or ledger.session.get('request_manifest_sha256')!=TEST
                or ledger.session.get('evaluation_only')is not True):
                raise ValueError('Actual reviewed TEST500 middle8 ledger required')
            self.ledger=ledger;self.active=None;self.restore=None;self.order=[];self.prefix=[];self.completed=0;self.verified=None
            self.by_input={(r['reference_sha256'],r['prompt'],r['seed']):r for r in ledger.requests}
            if len(self.by_input)!=500:raise ValueError('Complete distinct actual requests required')
            completed_cohort(ledger)
    return MiddleMeasuredObserver()


def completed_cohort(ledger):
    for stage in STAGES:
        report=json.loads((ledger.root.parent/(stage+'_outputs_data_stream_complete.json')).read_text())
        if not report['complete']or report['count']!=len(ledger.requests)or report['ledger_SHA256']!=ledger.key:
            raise ValueError('Incomplete/wrong actual cohort; no ideal fallback')


def native(a,binding):
    import torch
    from .qwen3_scene_paired_train import new_qwen
    from .scene_paired_render import install_optical_frontend
    from .scene_depth_probe import prune_frontend
    from .scene_lab_native import NativeRunner
    head,_=load_head(a.checkpoint/'best_checkpoint.pt',json.loads((a.checkpoint/'report.json').read_text()),binding)
    qwen,processor=new_qwen(SimpleNamespace(qwen=a.qwen,qwen_side=256))
    optical=install_optical_frontend(qwen,a.frontend,a.frontend_features,a.cache)
    if optical['checkpoint_sha256']!=F6:raise ValueError('Wrong F6 phase/frontend assets')
    graph=prune_frontend(qwen,6,4)
    if graph['retained_original_TF']!={'vision':6,'language':4}or graph['optical_cycles']!=6:
        raise ValueError('Native optical boundary graph changed')
    return NativeRunner(qwen,processor,head.cuda(),a.references,frontend_sha256=F6,
                        head_sha256=binding['generator_sha256'],export_features=True,middle8_binding=binding)


def prepare_or_features(a,binding):
    deployment_environment()
    import torch
    from .scene_lab_pilot import mark_input_ready
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.25,0)
    ledger=open_ledger(a.ledger)
    if ledger.middle8_binding!=binding:raise ValueError('Different physical input lineage')
    runner=native(a,binding)
    a.out.mkdir(parents=True,exist_ok=False)
    put(a.out/'execution.json',dict(source_commit=os.environ['T12_SOURCE_COMMIT'],generator_source_commit=binding['source_commit'],adapter_source_SHA256=sha(__file__),
        generator_SHA256=binding['generator_sha256'],mode=a.mode,SDK=False,gradients=False,cohort_role=ledger.session['cohort_role']))
    if a.mode=='prepare':
        if a.stage not in STAGES:raise ValueError('Explicit single whole-cohort optical stage required')
        admission=ledger.admit_stage(a.stage)
        def stream(request,provider):
            try:return runner(request,provider)
            except NeedCapture:
                mark_input_ready(ledger.folder(a.stage,request));raise
        pending=ledger.prepare(a.stage,stream,admission=admission)
        put(a.out/'report.json',dict(complete=True,stage=a.stage,pending_count=len(pending),
            ledger_SHA256=ledger.key,SDK=False,gradients=False,cohort_role=ledger.session['cohort_role']))
    else:
        completed_cohort(ledger)
        values={}
        for request in ledger.requests:values[request['sample_id']]=ledger.infer(request,runner)['features']
        role=ledger.session['cohort_role'];filename=role.upper()+'_measured_features.pt'
        torch.save(dict(schema='scene_lab_middle8_measured_features_v1',features=values,records=ledger.requests,
            binding=binding,ledger_SHA256=ledger.key,all_six_stages_measured=True,
            evaluation_only=role=='test',TEST_gradients=False),a.out/filename)
        put(a.out/'report.json',dict(complete=True,contexts=len(values),feature_SHA256=sha(a.out/filename),
            all_six_stages_measured=True,SDK=False,gradients=False,cohort_role=role,quality_accepted=False))


def matched_training_inputs(measured,nominal):
    """A separate guard for future existing-head fitting; TEST features cannot enter it."""
    from .scene_lab_adapt import match_train
    if (measured.get('schema')!='scene_lab_middle8_measured_features_v1' or measured.get('evaluation_only')is not False
        or measured.get('all_six_stages_measured')is not True or measured.get('TEST_gradients')is not False):
        raise ValueError('Only actual complete TRAIN features can calibrate the existing head')
    checked_binding(measured['binding']);require_training(measured['records'])
    if (nominal.get('schema')!='scene_lab_middle8_nominal_conditions_v1'
        or nominal.get('binding')!=measured['binding'] or nominal.get('pair_sha256')!=PAIRS
        or nominal.get('TRAIN_only')is not True):
        raise ValueError('Same accepted new PT/F6 TRAIN SIM targets required, never old F9 targets')
    return match_train(measured['records'],nominal['records'])


def render_main(binding,ledger_path,remaining):
    deployment_environment()
    import torch
    from . import scene_paired_render as renderer
    from . import scene_natural_prompt as prompts
    from .scene_expanded_selection_render import checked_revision,public_inputs,public_report
    from .scene_compact_candidates import GraphObserver
    old_render,old_inputs,old_revision=renderer.render,renderer.inference_inputs,prompts.checked_revision
    def render(a):
        report=json.loads((a.checkpoint/'report.json').read_text())
        _,state=load_head(a.checkpoint/'best_checkpoint.pt',report,binding)
        if (a.steps!=8 or a.scope!='dev' or a.prompt_revision is None or not a.fuse_task_lora
            or a.sampler!='original' or a.lcm_adapter or a.measure_software_latency
            or a.ip_scale!=.25 or a.unet_precision!='bf16' or getattr(a,'probe_conditioning_side',None)is not None
            or getattr(a,'request_observer',None)is not None):
            raise ValueError('Exact expanded8-step quality renderer only; no timing/precision/conditioning overrides')
        check_data(a.data,a.data)
        measured=None
        if ledger_path is not None:
            measured=measured_observer(open_ledger(ledger_path),binding,sha(a.data/'inference_inputs.json'))
        observer=ComposedObserver(GraphObserver(binding['compression_profile']),measured)
        a._verified_depth_compression=state['depth_compression'];a.request_observer=observer
        try:
            old_render(a)
            if measured is not None and measured.completed!=500:raise ValueError('Incomplete actual500 replay')
            public_report(a.output)
            path=a.output/'report.json';out=json.loads(path.read_text())
            if out['generated_count']!=500:raise ValueError('All500 quality outputs required')
            out.update(lab_middle8_binding=binding,actual_CCD_replay=measured is not None,SDK_used=False,
                depth_compression=state['depth_compression'],experimental_compression=observer.depth.graph,
                compression_training_performed=True,physical_ledger_SHA256=measured.ledger.key if measured else None,
                physical_CCD_stage_calls=3000 if measured else 0,TEST_gradients=False,quality_accepted=False)
            path.write_text(json.dumps(out,indent=2))
        finally:observer.close()
    # Do not invoke renderer.main(): that simulation CLI is intentionally bound
    # to GPU6/A100 and an80% allocation. The bench adapter keeps its existing
    # GPU1 policy and calls the same unchanged render function directly.
    a=render_arguments(remaining)
    if a.output.exists():raise FileExistsError('Fresh physical/SIM run required')
    a.output.mkdir(parents=True)
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.5,0)
    put(a.output/'execution.json',dict(source_commit=os.environ['T12_SOURCE_COMMIT'],
        generator_source_commit=binding['source_commit'],adapter_source_SHA256=sha(__file__),
        source_renderer_SHA256=sha(Path(renderer.__file__)),SDK=False,actual_CCD_replay=ledger_path is not None))
    renderer.inference_inputs=public_inputs;prompts.checked_revision=checked_revision
    try:render(a)
    except BaseException as exc:
        import traceback
        put(a.output/'failure.json',dict(error_type=type(exc).__name__,traceback=traceback.format_exc()));raise
    finally:
        renderer.inference_inputs=old_inputs;prompts.checked_revision=old_revision


def render_arguments(remaining):
    """Parse the exact original generation contract without importing Torch."""
    p=argparse.ArgumentParser(description='Accepted middle8 SIM/actual quality replay, no SDK')
    for key in ('data','checkpoint','qwen','cache','model-path','adapter-path','output','lg-frontend','lg-features','prompt-revision'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--fuse-task-lora',action='store_true',required=True)
    p.add_argument('--steps',type=int,choices=[8],default=8)
    p.add_argument('--ip-scale',type=float,choices=[.25],default=.25)
    p.add_argument('--unet-precision',choices=['bf16'],default='bf16')
    p.add_argument('--sampler',choices=['original'],default='original')
    p.add_argument('--scope',choices=['dev'],default='dev',help='Internal revised-request mechanism; reports explicitly identify TEST selection')
    a=p.parse_args(remaining)
    vars(a).update(expanded_development=None,lcm_adapter=None,lcm_guidance=1.,measure_software_latency=False,
                   reference_run=None,frozen_evaluation=None,frozen_model_key=None,probe_conditioning_side=None)
    return a


def main():
    p=argparse.ArgumentParser(description=__doc__,add_help=False)
    p.add_argument('--mode',choices=['init','prepare','features','render-sim','render-exp'],required=True)
    p.add_argument('--binding',type=Path,required=True);p.add_argument('--ledger',type=Path)
    a,remaining=p.parse_known_args();binding=checked_binding(json.loads(a.binding.read_text()))
    if a.mode.startswith('render-'):
        if (a.mode=='render-exp')!=(a.ledger is not None):raise ValueError('Actual render requires its new ledger; SIM must not take one')
        render_main(binding,a.ledger,remaining);return
    q=argparse.ArgumentParser(description=__doc__)
    for k in ['data','session','qwen','checkpoint','frontend','frontend-features','cache','references','out']:
        q.add_argument('--'+k,type=Path)
    q.add_argument('--role',choices=['train','test']);q.add_argument('--stage',choices=STAGES)
    extra=q.parse_args(remaining);vars(a).update(vars(extra))
    if a.ledger is None:p.error('--ledger required')
    if a.mode=='init':
        if any(v is None for v in (a.data,a.session,a.role)):q.error('init requires --data --session --role')
        ledger=initialize(a.ledger,json.loads(a.session.read_text()),binding,a.data,a.role)
        print(json.dumps(dict(initialized=True,contexts=len(ledger.requests),SDK=False,physical_capture_started=False)))
    else:
        if any(getattr(a,k)is None for k in ('qwen','checkpoint','frontend','frontend_features','cache','references','out')):
            q.error('Native replay requires all explicit model/reference/cache/output paths')
        prepare_or_features(a,binding)


if __name__=='__main__':main()
