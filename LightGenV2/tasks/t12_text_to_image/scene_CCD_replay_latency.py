"""SDK-free full-request CCD replay with fixed optical waits, not a bench result.

Only six detector calculations are replaced. Each request still executes native
RGB/text Qwen, codecs, routing, electronic residuals, fusion, head, UNet and VAE.
CPU SLM-payload conversion and host/device copies remain in the timed window.
An exact-field audit must precede formal replay rounds. Unknown actual driver,
SLM transport, DMA and calibration costs are not silently assumed zero.
"""
import argparse,json,math,os,sys,time
from pathlib import Path

F6='6f34f2d720060c9193cda0e19ffb8fb4d157bc03e3fd5e5c1433c60ad884471e'
PERIOD_SECONDS=1/1886+.000300+.0000307


def wait_optical_period(period=PERIOD_SECONDS,*,clock=time.perf_counter,sleep=time.sleep):
    if not math.isfinite(period) or period<=0:raise ValueError('Positive fixed optical period required')
    start=clock();remaining=period
    while remaining>0:
        sleep(remaining);remaining=period-(clock()-start)
    return (clock()-start)*1000


def request_matches(row,tape):
    return (row['sample_id'],row['prompt'],row['reference_sha256'])==(tape['sample_id'],tape['prompt'],tape['reference_sha256'])


class ReplayObserver:
    def __init__(self,folder,output,config,verify_fields,audit):
        from .white_reference_scene_probe import digest
        import numpy as np
        import torch
        self.folder=Path(folder);self.output=Path(output);self.config=config;self.verify_fields=verify_fields;self.audit=audit
        report=json.loads((self.folder/'report.json').read_text());proof=json.loads((self.folder/'boundary_check.json').read_text())
        assert report['complete'] and proof['complete'] and proof['simulation_and_replay_hidden_exact']
        assert report['optical_frontend']['checkpoint_sha256']==F6 and proof['n']==8
        assert proof['ideal_boundary_replay_is_not_physical_CCD'] and proof['output_PNG_identity_vs_completed_quality_run']
        from .scene_optical_boundary import STAGES
        self.stages=STAGES;self.tapes={};self.frames={};self.rows=[];self.restorer=None
        for row in proof['requests']:
            key=row['sample_id'];assert tuple(row['stages'])==STAGES and row['optical_simulation_calls_in_replay']==0
            self.tapes[key]=row;self.frames[key]={}
            for stage in STAGES:
                path=self.folder/'boundaries'/key/(stage+'.npz');assert digest(path)==row['files'][path.name]['sha256']
                with np.load(path,allow_pickle=False)as f:
                    values={name:torch.from_numpy(f[name].copy())for name in ['amplitude','total_phase_radians','ideal_CCD']}
                assert all(v.shape==(1,478,478) and torch.isfinite(v).all()for v in values.values())
                assert values['amplitude'].min()>=0 and values['amplitude'].max()<=1 and values['ideal_CCD'].min()>=0
                self.frames[key][stage]=values
        self.proof_sha=digest(self.folder/'boundary_check.json')
        if not verify_fields:
            checked=json.loads(Path(audit).read_text())
            assert checked['complete'] and checked['all48_fields_exact'] and checked['tape_boundary_check_sha256']==self.proof_sha
            assert checked['replay_module_source_sha256']==digest(Path(__file__))

    def prepare(self,qwen,processor,head,pipe):
        from .scene_compact_candidates import GraphObserver
        from .scene_optical_boundary import attach
        if self.config:
            graph=GraphObserver(self.config);processor=graph.prepare(qwen,processor,head,pipe)
        assert set(qwen.model.hybrid)=={'vision','language'} and all(m.pairs==1 for m in qwen.model.hybrid.values())
        self.restorer=attach(qwen.model.hybrid,self.provide,simulation_diagnostic=False)
        return processor

    def begin(self,row):
        if not request_matches(row,self.tapes[row['sample_id']]):raise ValueError('CCD tape is bound to different input RGB or full prompt')
        self.current=row;self.calls=[]

    def provide(self,stage,amplitude,phase,ideal):
        import numpy as np
        import torch
        assert ideal is None and stage==self.stages[len(self.calls)]
        frames=self.frames[self.current['sample_id']][stage]
        # Actual per-frame device-to-host copies and CPU pixel conversion are
        # timed, rather than treating an unspecified SLM transport as free.
        a=amplitude.detach().cpu().float();p=phase.detach().cpu().float()
        if self.verify_fields:
            assert torch.equal(a,frames['amplitude']) and torch.equal(p,frames['total_phase_radians']),stage
        amplitude_pixels=np.rint(a.numpy()*255).clip(0,255).astype(np.uint8)
        phase_pixels=np.rint(p.numpy()/(2*np.pi)*255).clip(0,255).astype(np.uint8)
        elapsed=wait_optical_period()
        self.calls.append(dict(stage=stage,fixed_wait_target_ms=PERIOD_SECONDS*1000,actual_wait_ms=elapsed,
            host_SLM_payload_bytes=amplitude_pixels.nbytes+phase_pixels.nbytes,
            exact_fields_checked=self.verify_fields,CCD_host_to_device_copy_performed_by_boundary=True))
        # This is a stored *ideal intensity frame*, not a cached Qwen prefix.
        # The native readout and subsequent layer compute still run normally.
        return frames['ideal_CCD']

    def end(self,row):
        assert tuple(v['stage']for v in self.calls)==self.stages
        self.rows.append(dict(sample_id=row['sample_id'],stages=self.calls))

    def finish(self):
        from .white_reference_scene_probe import digest,write
        assert len(self.rows)==8 and len({r['sample_id']for r in self.rows})==8
        write(self.output/'CCD_replay_latency.json',dict(complete=True,n=8,requests=self.rows,
            tape_boundary_check_sha256=self.proof_sha,replay_module_source_sha256=digest(Path(__file__)),
            all48_fields_exact=self.verify_fields,audit_run_not_formal_timing=self.verify_fields,
            audit_proof=None if self.verify_fields else str(self.audit),raw_RGB_text_Qwen_each_request=True,
            FFT_propagation_calls=0,physical_CCD_or_SDK_used=False,model_load_and_tape_loading_outside_window=True,
            electronic_residual_and_fusion_retained=True,CPU_payload_conversion_and_device_copies_retained=True,
            ideal_float_CCD_is_not_quantized_physical_quality=True,actual_SLM_LUT_or_transport_not_measured=True,
            unknown_additional_hardware_cost='H >= 0, unmeasured; not assumed zero',fixed_optical_ms=6*PERIOD_SECONDS*1000,
            unprofiled_complete_request_optical_wait_emulation=True,physical_speed_or_quality_accepted=False))


def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--CCD-tape',type=Path,required=True)
    p.add_argument('--compression-config',type=Path);p.add_argument('--verify-all-fields',action='store_true')
    p.add_argument('--field-audit',type=Path);custom,remaining=p.parse_known_args()
    if not custom.verify_all_fields and custom.field_audit is None:raise ValueError('A prior exact-field audit is required')
    config=None
    if custom.compression_config:
        from .scene_depth_probe import checked_config
        config=checked_config(json.loads(custom.compression_config.read_text()))
    import torch
    from . import scene_paired_render as renderer
    from .white_reference_scene_probe import digest,write
    original=renderer.render
    def render(a):
        assert a.measure_software_latency and a.reference_run and a.lg_frontend and a.fuse_task_lora
        assert a.scope=='dev' and a.prompt_revision and a.sampler=='original' and not a.lcm_adapter
        quality=json.loads((a.reference_run/'report.json').read_text())
        assert quality['optical_frontend']['checkpoint_sha256']==F6 and not quality['image_feedback']
        if config:
            state=torch.load(a.checkpoint/'best_checkpoint.pt',map_location='cpu',weights_only=False)
            report=json.loads((a.checkpoint/'report.json').read_text());description=state['depth_compression']
            assert description['schema']=='scene_compact_recovered_v1' and description['profile']==config
            assert report['compression_training_performed'] and digest(a.checkpoint/'best_checkpoint.pt')==report['best_sha256']
            assert a.steps==state['sampling_contract']['steps']==quality['sampling_steps']
            a._verified_depth_compression=description
        observer=ReplayObserver(custom.CCD_tape,a.output,config,custom.verify_all_fields,custom.field_audit);a.request_observer=observer
        try:original(a);observer.finish()
        finally:
            if observer.restorer:observer.restorer()
        path=a.output/'software_latency.json';value=json.loads(path.read_text())
        value.update(contains_GPU_FFT_optical_simulation=False,optical_wait_emulation=True,
            audit_run_not_formal_timing=custom.verify_all_fields,physical_hardware_estimate_or_measurement=False,
            fixed_optical_ms=6*PERIOD_SECONDS*1000,unknown_additional_driver_and_DMA_not_assumed_zero=True)
        write(path,value)
        report=json.loads((a.output/'report.json').read_text());report.update(CCD_replay_optical_wait_emulation=True,
            FFT_simulation_used=False,physical_measurement=False,audit_run_not_formal_timing=custom.verify_all_fields)
        write(a.output/'report.json',report)
    renderer.render=render;sys.argv=[sys.argv[0],*remaining]
    try:renderer.main()
    finally:renderer.render=original


if __name__=='__main__':main()
