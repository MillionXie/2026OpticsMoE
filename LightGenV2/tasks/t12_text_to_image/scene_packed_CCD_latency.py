"""Optional byte-exact packed SLM-payload timing, without any device SDK.

Quantization runs on the GPU and transfers one contiguous uint8 amplitude/phase
payload per optical stage, instead of transferring two full float fields and
quantizing on the CPU. Both routes must produce identical payload bytes in a
separate all48-field audit. This is an implementation optimization of necessary
electronic work, not removal of that work or a physical latency measurement.
"""
import json
from pathlib import Path
from . import scene_CCD_replay_latency as base


def packed_pixels(amplitude,phase):
    import torch
    if amplitude.shape!=phase.shape or amplitude.dtype!=torch.float32 or phase.dtype!=torch.float32:
        raise ValueError('Matching native float32 fields required')
    a=torch.round(amplitude*255).clamp(0,255).to(torch.uint8)
    p=torch.round(phase/(2*torch.pi)*255).clamp(0,255).to(torch.uint8)
    return torch.stack((a,p),dim=0).contiguous()


def cpu_reference_pixels(amplitude,phase):
    import numpy as np
    a=amplitude.detach().cpu().float().numpy();p=phase.detach().cpu().float().numpy()
    return np.stack((np.rint(a*255).clip(0,255).astype(np.uint8),np.rint(p/(2*np.pi)*255).clip(0,255).astype(np.uint8)))


class PackedReplayObserver(base.ReplayObserver):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        from .white_reference_scene_probe import digest
        if not self.verify_fields:
            proof=json.loads(Path(self.audit).read_text())
            if (not proof.get('all48_packed_payloads_byte_exact')
                or proof.get('packed_module_source_sha256')!=digest(Path(__file__))):
                raise ValueError('A new all48 packed-payload byte identity audit is required')

    def provide(self,stage,amplitude,phase,ideal):
        import numpy as np
        import torch
        assert ideal is None and stage==self.stages[len(self.calls)]
        frames=self.frames[self.current['sample_id']][stage]
        if self.verify_fields:
            assert torch.equal(amplitude.cpu(),frames['amplitude'])and torch.equal(phase.cpu(),frames['total_phase_radians']),stage
        payload=packed_pixels(amplitude,phase).cpu()
        if self.verify_fields:
            reference=cpu_reference_pixels(amplitude,phase)
            if not np.array_equal(payload.numpy(),reference):raise ValueError('GPU and CPU SLM quantization bytes differ: '+stage)
        elapsed=base.wait_optical_period()
        self.calls.append(dict(stage=stage,fixed_wait_target_ms=base.PERIOD_SECONDS*1000,actual_wait_ms=elapsed,
            host_SLM_payload_bytes=payload.numel(),payload_dtype='uint8',single_packed_host_transfer=True,
            GPU_quantization_and_device_copies_retained=True,exact_fields_checked=self.verify_fields,
            payload_byte_identity_checked=self.verify_fields,CCD_host_to_device_copy_performed_by_boundary=True))
        return frames['ideal_CCD']

    def finish(self):
        super().finish()
        from .white_reference_scene_probe import digest,write
        path=self.output/'CCD_replay_latency.json';value=json.loads(path.read_text())
        value.update(packed_module_source_sha256=digest(Path(__file__)),
            all48_packed_payloads_byte_exact=self.verify_fields,SLM_quantization_on_GPU=True,
            GPU_to_CPU_payload='one contiguous uint8 amplitude/phase tensor per stage',
            optical_wait_and_all_electronic_work_retained=True,physical_measurement=False)
        write(path,value)


def main():
    old=base.ReplayObserver;base.ReplayObserver=PackedReplayObserver
    try:base.main()
    finally:base.ReplayObserver=old


if __name__=='__main__':main()
