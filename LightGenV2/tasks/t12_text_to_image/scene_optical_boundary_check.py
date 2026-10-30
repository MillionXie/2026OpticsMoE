"""Offline six-stage boundary self-test. Never imports an SDK or contacts a bench."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .scene_optical_boundary import attach,STAGES


def pixels(image):return hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()


def main():
    from . import scene_paired_render as render
    original_render=render.render;original_features=render.raw_scene_features
    context={}
    def prepare(a):
        if not a.measure_software_latency or a.lg_frontend is None or not a.fuse_task_lora:
            raise ValueError('Require evaluated single-EG merged graph and the fixed B1 diagnostic cohort')
        records=json.loads((a.reference_run/'images.json').read_text())
        rows=[q for category in ['lamp','pillow'] for q in records if q['category']==category][:0]
        rows=[q for category in ['lamp','pillow'] for q in [x for x in records if x['category']==category][:4]]
        assert len(rows)==8
        expected=[]
        for q in rows:
            with Image.open(a.reference_run/(q['reference_id']+'.png')) as im:key=pixels(im)
            expected.append((q,key))
        context.update(output=a.output,expected=expected,done=[],args=a)
        original_render(a)
        assert len(context['done'])==8
        report=dict(complete=True,source_commit=render.os.environ['T12_SOURCE_COMMIT'],
            n=8,STAGES=STAGES,requests=context['done'],SDK_imported=False,hardware_accessed=False,
            ideal_boundary_replay_is_not_physical_CCD=True,old_17M_lab_runner_used=False,
            no_extra_generator_parameters=True,simulation_and_replay_hidden_exact=True,
            output_PNG_identity_vs_completed_quality_run=True,selftest_not_speed_measurement=True,
            exported_inputs_are_simulation_examples_not_final_measured_stage_inputs=True,
            measured_deployment_requires_layerwise_recomputation_and_dark_gain_ROI_calibration=True,
            TEST_opened=False)
        (a.output/'boundary_check.json').write_text(json.dumps(report,indent=2))
    def check(qwen,processor,image,prompt,side,keep_on_device=False):
        row,key=context['expected'][len(context['done'])]
        assert pixels(image)==key and prompt==row['prompt'] and keep_on_device
        baseline=original_features(qwen,processor,image,prompt,side,keep_on_device=True)
        frames={};order=[];folder=context['output']/'boundaries'/row['sample_id'];folder.mkdir(parents=True)
        def collect(stage,amplitude,phase,ideal):
            if stage!=STAGES[len(order)] or ideal is None:raise ValueError('Wrong actual optical dependency order')
            order.append(stage);frames[stage]=ideal.detach().clone()
            values={k:v.detach().cpu().float().numpy() for k,v in [('amplitude',amplitude),('total_phase_radians',phase),('ideal_CCD',ideal)]}
            path=folder/(stage+'.npz');np.savez_compressed(path,**values)
            Image.fromarray(np.rint(values['amplitude'][0].clip(0,1)*255).astype('uint8')).save(folder/(stage+'_bounded_amplitude_preview.png'))
            Image.fromarray(np.rint(values['total_phase_radians'][0]/(2*np.pi)*255).clip(0,255).astype('uint8')).save(folder/(stage+'_phase_preview_not_SLM_LUT.png'))
            return ideal
        restore=attach(qwen.model.hybrid,collect,simulation_diagnostic=True)
        try:observed=original_features(qwen,processor,image,prompt,side,keep_on_device=True)
        finally:restore()
        assert tuple(order)==STAGES
        assert all(torch.equal(a,b) for a,b in zip(baseline,observed))
        replay_order=[]
        def replay(stage,amplitude,phase,ideal):
            if ideal is not None or stage!=STAGES[len(replay_order)]:raise ValueError('Replay must not execute an FFT fallback')
            replay_order.append(stage);return frames[stage]
        restore=attach(qwen.model.hybrid,replay,simulation_diagnostic=False)
        try:actual=original_features(qwen,processor,image,prompt,side,keep_on_device=True)
        finally:restore()
        assert tuple(replay_order)==STAGES
        assert all(torch.equal(a,b) for a,b in zip(baseline,actual))
        files={p.name:dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in folder.iterdir() if p.is_file()}
        context['done'].append(dict(sample_id=row['sample_id'],prompt=prompt,reference_sha256=row['input_sha256'],
            stages=order,actual_Qwen_forwards=3,plain_vs_diagnostic_vs_CCD_replay_tensor_max_error=0,
            optical_simulation_calls_in_replay=0,files=files))
        (context['output']/'boundary_check_progress.json').write_text(json.dumps(dict(done=len(context['done']),total=8)))
        return actual
    render.render=prepare;render.raw_scene_features=check;render.main()


if __name__=='__main__':main()
