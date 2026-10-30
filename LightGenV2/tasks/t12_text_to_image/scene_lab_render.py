"""Original F9 renderer with a verified actual-CCD observer, no SDK/FFT fallback.

The observer only supplies the six already measured optical intensities. It
does not replace Qwen, the head, SDXL, sampling, precision or task LoRA.
"""
import argparse,json,os,sys,traceback
from pathlib import Path
from types import SimpleNamespace
from .scene_lab_protocol import Ledger,STAGES,identity,array_sha,encode,sha,put
from .scene_lab_native import F6,F9

GPU='GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d'


class MeasuredObserver:
    def __init__(self,ledger,manifest_sha256):
        if ledger.session.get('scope')!='reviewed_physical' or ledger.models['F6']!=F6 or ledger.models['F9']!=F9:
            raise ValueError('Only the original F6/F9 reviewed physical ledger is eligible')
        if ledger.session.get('cohort_role')!='dev' or ledger.session.get('evaluation_manifest_sha256')!=manifest_sha256:raise ValueError('Wrong actual DEV manifest')
        self.ledger=ledger;self.active=None;self.restore=None;self.order=[];self.prefix=[];self.completed=0;self.verified=None
        self.by_input={(r['reference_sha256'],r['prompt'],r['seed']):r for r in ledger.requests}
        if len(self.by_input)!=128:raise ValueError('Complete unique DEV128 required')
        # Admit only six complete DATA cohorts. Actual bytes/provenance are
        # still fully checked before EACH request's one complete Qwen forward;
        # do not sweep all large camera files once here and once again later.
        for stage in STAGES:
            path=ledger.root.parent/(stage+'_outputs_data_stream_complete.json');report=json.loads(path.read_text())
            if not report['complete']or report['count']!=128 or report['ledger_SHA256']!=ledger.key:raise ValueError('Incomplete/wrong actual DEV DATA cohort')

    def prepare(self,qwen,processor,head,pipe):
        from .scene_optical_boundary import attach
        self.restore=attach(qwen.model.hybrid,self.provider,simulation_diagnostic=False)
        return processor

    def begin(self,row):
        if self.active is not None:raise ValueError('Nested physical request')
        self.active=self.by_input[(row['reference_sha256'],row['prompt'],row['seed'])];self.order=[];self.prefix=[]
        self.verified={stage:self.ledger.receipt(stage,self.active)for stage in STAGES}
        if any(receipt['metadata']['frame_provenance']!='physical_linear_camera'for receipt,_ in self.verified.values()):raise ValueError('Synthetic frame is not physical evidence')

    def provider(self,stage,amplitude,phase,ideal):
        import torch
        if self.active is None or ideal is not None or stage!=STAGES[len(self.order)]:raise ValueError('Missing/wrong actual stage or forbidden ideal fallback')
        a,p,ag,pg=encode(amplitude.detach().cpu().float().numpy(),phase.detach().cpu().float().numpy());folder=self.ledger.folder(stage,self.active)
        expected=dict(ledger_sha256=self.ledger.key,request_sha256=identity(self.active),stage=stage,prefix_receipts=list(self.prefix),
            amplitude_sha256=array_sha(a),total_phase_sha256=array_sha(p),logical_amplitude_code_sha256=array_sha(ag),logical_phase_code_sha256=array_sha(pg))
        if json.loads((folder/'ticket.json').read_text())!=expected:raise ValueError('Physical input identity differs; never reuse these CCDs')
        receipt,value=self.verified[stage];self.order.append(stage);self.prefix.append(identity(receipt))
        return torch.from_numpy(value.copy()).to(amplitude.device)

    def end(self,row):
        if tuple(self.order)!=STAGES:raise ValueError('Incomplete six actual stages')
        self.completed+=1;self.active=None;self.verified=None

    def close(self):
        if self.restore is not None:self.restore();self.restore=None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','dev','data','qwen','cache','model-path','adapter-path','out'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--ledger',type=Path);p.add_argument('--source-commit',required=True);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU or len(a.source_commit)!=40:raise ValueError('Explicit shared GPU1 and published source required')
    import torch
    from .scene_paired_render import render
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.5,0)
    task=Path(__file__).resolve().parent;base=task/'runs/simulation';state_report=json.loads((a.checkpoint/'report.json').read_text())
    if not state_report['complete'] or sha(a.checkpoint/'best_checkpoint.pt')!=state_report['best_sha256']:raise ValueError('Selected checkpoint changed')
    state=torch.load(a.checkpoint/'best_checkpoint.pt',map_location='cpu',weights_only=False)
    if state_report['best_sha256']!=F9:
        adaptation=state.get('lab_adaptation',{})
        if adaptation.get('parent_F9_SHA256')!=F9 or adaptation.get('frontend_F6_SHA256')!=F6 or not adaptation.get('unchanged_frontend_alpha_optics_UNet_LoRA'):raise ValueError('Not a verified original-head F9 adaptation')
    if state['optical_frontend']['checkpoint_sha256']!=F6 or not state['joint_image_context'] or state.get('depth_compression'):raise ValueError('Preserve original frozen graph')
    observer=None
    if a.ledger:
        c=json.loads((a.ledger/'contract.json').read_text());observer=MeasuredObserver(Ledger(a.ledger,c['session'],c['models'],c['requests']),sha(a.dev/'inference_inputs.json'))
    args=SimpleNamespace(data=a.data,checkpoint=a.checkpoint,qwen=a.qwen,cache=a.cache,model_path=a.model_path,adapter_path=a.adapter_path,
        output=a.out,lg_frontend=base/'qwen3_scene_LG_frontend_20261029',lg_features=base/'qwen3_scene_LG_teacher_features_20261029',
        lcm_adapter=None,fuse_task_lora=True,unet_precision='bf16',lcm_guidance=1.,expanded_development=None,prompt_revision=a.dev,
        sampler='original',steps=12,ip_scale=.25,measure_software_latency=False,reference_run=None,scope='dev',frozen_evaluation=None,frozen_model_key=None,
        request_observer=observer)
    a.out.mkdir(parents=True,exist_ok=False);os.environ['T12_SOURCE_COMMIT']=a.source_commit
    put(a.out/'execution.json',dict(argv=sys.argv,source_commit=a.source_commit,source_SHA256=sha(__file__),original_renderer_SHA256=sha(Path(__file__).with_name('scene_paired_render.py')),
        actual_CCD_replay=observer is not None,SDK=False,TRAIN_or_TEST_gradients=False,GPU_memory_fraction=.5))
    try:
        render(args)
        if observer is not None and observer.completed!=128:raise ValueError('Incomplete actual DEV rendering')
        path=a.out/'report.json';report=json.loads(path.read_text());report.update(actual_CCD_replay=observer is not None,
            physical_ledger_SHA256=observer.ledger.key if observer is not None else None,physical_CCD_stage_calls=768 if observer is not None else 0,
            original_frozen_F9_SHA256=F9,adapted_existing_head=state_report['best_sha256']!=F9,SDK_used=False)
        path.write_text(json.dumps(report,indent=2))
    except BaseException as error:
        put(a.out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc()));raise
    finally:
        if observer is not None:observer.close()


if __name__=='__main__':main()
