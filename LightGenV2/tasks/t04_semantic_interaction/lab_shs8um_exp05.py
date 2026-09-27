"""Pinned OpenMoji exp05 physical evaluation, whole dataset per plane."""
import argparse,json,sys,time,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import torch

SHA='dae370fff51a0d175a5c73d7b7a19d455c3f27f56232af800b85ecfe026a8a25'
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2),encoding='utf8');tmp.replace(path)
def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,default=1000);p.add_argument('--selftest',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--exposure-us',type=int,choices=(400,1000,2000,4000),default=400);a=p.parse_args()
    torch.set_num_threads(4);root=a.project.resolve();out=a.output.resolve()
    sys.path.insert(0,str(root/'source'))
    from LightGenV2.tasks.t04_semantic_interaction.settings import load_settings
    from LightGenV2.tasks.t04_semantic_interaction.modeling import build_model
    from LightGenV2.tasks.t04_semantic_interaction.lab_runtime import OpticalBoundary,StopAtPlane,STAGES,phase_planes
    from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.datasets import OpenMojiEditingDataset,collate_samples,load_prompt_cache
    from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.metrics import MetricAccumulator
    abo=root.parent/'ABO_I2I_Lab_DVP_8um';sys.path.insert(0,str(abo/'lab_dvp8um'))
    import four_image_flow as flow
    from shs_physical2400 import SHSBench,CORNERS
    flow.BASE_CORNERS=CORNERS.copy()
    ckpt=root/'weights/best_checkpoint.pt';assert flow.sha(ckpt)==SHA
    cfg=load_settings(root/'source/LightGenV2/tasks/t04_semantic_interaction/configs/layered_scene_electronic_exp05.yaml')
    cfg.data_dir=root/'data';cfg.prompt_cache_path=cfg.data_dir/'token_embeddings_v1.pt';cfg.output_dir=out
    legacy=root.parent/'OpenMoji_Lab_SHS_8um';cfg.qwen_checkpoint=legacy/'frontend';cfg.asset_dir=legacy/'assets'
    cfg.svg_asset_dir=cfg.asset_dir/'openmoji-17.0.0-svg'
    model=build_model(cfg,torch.device('cuda'));payload=torch.load(ckpt,map_location='cpu',weights_only=False)
    assert payload['architecture']==model.checkpoint_architecture
    model.load_state_dict(payload['model'],strict=True);model.eval().requires_grad_(False)
    for optic in model._optical_paths():optic.set_phase_dropout_active(False)
    dataset=OpenMojiEditingDataset(cfg.test_manifest,cfg,load_prompt_cache(cfg.prompt_cache_path));assert len(dataset)==1000
    total=min(a.limit,len(dataset));out.mkdir(parents=True,exist_ok=True)
    contract=dict(checkpoint_sha256=SHA,source_commit='2acb5bcf5de5a902d153e29e885131227ea61914',epoch=payload['epoch'],count=total,exposure_us=a.exposure_us,gain='Gain_X4',wait_ms=240,corners=CORNERS.tolist(),phase_orientation='hv',phase_inverse=True,camera_orientation='flip_v',amplitude_encoding='per-sample peak, no percentile clipping',test_manifest_sha256=flow.sha(cfg.test_manifest))
    if (out/'contract.json').exists():assert json.loads((out/'contract.json').read_text())==contract
    else:write(out/'contract.json',contract)
    def batch(index):return {k:v.cuda() if torch.is_tensor(v) else v for k,v in collate_samples([dataset[index]]).items()}
    def frames(index,stages):return {stage:torch.from_numpy(np.asarray(Image.open(out/'ccd'/stage/f'test_{index:05d}.png'),np.float32).copy()[None]/255) for stage in stages}
    if a.selftest:
        x=batch(0)
        with torch.inference_mode():
            with OpticalBoundary(model) as tap:baseline=model(x['source_image'],x['prompt_hidden'])
            with OpticalBoundary(model,tap.detectors):replayed=model(x['source_image'],x['prompt_hidden'])
            scaled={stage:value/tap.amplitudes[stage].amax().square() for stage,value in tap.detectors.items()}
            with OpticalBoundary(model,scaled):peak_replayed=model(x['source_image'],x['prompt_hidden'])
        errors={k:float((baseline[k]-replayed[k]).abs().max()) for k in baseline if torch.is_tensor(baseline[k])}
        assert max(errors.values(),default=0)<1e-5,errors
        peak_errors={k:float((baseline[k]-peak_replayed[k]).abs().max()) for k in baseline if torch.is_tensor(baseline[k]) and not k.endswith('_loss')}
        assert max(peak_errors.values(),default=0)<2e-3,peak_errors
        assert all(torch.equal(baseline[k].argmax(-1),peak_replayed[k].argmax(-1)) for k in ('category_logits','edit_logits','task_logits'))
        write(out/'report.json',dict(status='complete',selftest=True,errors=errors,peak_scaling_errors=peak_errors,contract=contract));return
    phases=phase_planes(model);phase_dir=out/'phase';phase_dir.mkdir(exist_ok=True)
    for stage,value in phases.items():Image.fromarray(flow.phase_gray(value,'hv',True)).save(phase_dir/(stage+'.bmp'))
    started=time.time();stats={};simacc=MetricAccumulator();actualacc=MetricAccumulator()
    with SHSBench(out,a.exposure_us,240,{}) as bench:
        if total<=4:
            health=out/'health';health.mkdir(exist_ok=True)
            flat=health/'phase_flat.bmp';Image.fromarray(flow.phase_gray(np.zeros((478,478)),'hv',True)).save(flat)
            levels={}
            for gray in (0,255):
                health_exposure=min(a.exposure_us,400)
                bench.camera.set('ExposureTime',health_exposure);bench.settings['exposure_us']=float(bench.camera.get('ExposureTime'))
                path=health/f'amplitude_{gray}.bmp';Image.fromarray(flow.active_to_native(np.full((478,478),gray,np.uint8))).save(path)
                values,_=bench.capture('health',flat,[path],[f'gray_{gray}'],'flip_v')
                levels[gray]=dict(exposure_us=health_exposure,p99=float(np.percentile(values[0],99)),mean=float(values[0].mean()),saturation=float(np.mean(values[0]==255)))
            write(out/'health.json',levels)
            assert levels[255]['p99']>levels[0]['p99']+20,'White optical signal missing'
            bench.camera.set('ExposureTime',a.exposure_us);bench.settings['exposure_us']=float(bench.camera.get('ExposureTime'));bench.camera.fresh()
        for stage_index,stage in enumerate(STAGES):
            folder=out/'ccd'/stage;folder.mkdir(parents=True,exist_ok=True);ampdir=out/'amplitude'/stage;ampdir.mkdir(parents=True,exist_ok=True)
            for start in range(0,total,4):
                paths=[];ids=[]
                for index in range(start,min(start+4,total)):
                    sid=f'test_{index:05d}'
                    if a.resume and (folder/(sid+'.png')).exists() and (folder/(sid+'.json')).exists():continue
                    x=batch(index)
                    with torch.inference_mode(),OpticalBoundary(model,frames(index,STAGES[:stage_index]),stage) as tap:
                        try:model(x['source_image'],x['prompt_hidden'])
                        except StopAtPlane:pass
                    amplitude=tap.amplitudes[stage][0].cpu().numpy();peak=float(amplitude.max());assert peak>0
                    gray=np.rint(np.clip(amplitude/peak,0,1)*255).astype(np.uint8)
                    path=ampdir/(sid+'.bmp');Image.fromarray(flow.active_to_native(gray)).save(path);paths.append(path);ids.append(sid)
                    write(ampdir/(sid+'.json'),dict(peak=peak,mean_power=float(np.mean((amplitude/peak)**2)),quantization_max_error=float(np.max(np.abs(gray.astype(float)/255-amplitude/peak))),phase_sha256=flow.sha(phase_dir/(stage+'.bmp'))))
                if paths:
                    values,_=bench.capture(stage,phase_dir/(stage+'.bmp'),paths,ids,'flip_v')
                    stats.setdefault(stage,[]).extend([dict(sample_id=sid,p99=float(np.percentile(v,99)),mean=float(v.mean()),saturation=float(np.mean(v==255))) for sid,v in zip(ids,values)])
                    for path in paths:path.unlink()
                write(out/'progress.json',dict(status='capturing',stage=stage,stage_completed=min(start+4,total),total=total,completed_stages=list(STAGES[:stage_index]),elapsed_seconds=time.time()-started))
    rows=[]
    with torch.inference_mode():
        for index in range(total):
            x=batch(index);simulation=model(x['source_image'],x['prompt_hidden'])
            with OpticalBoundary(model,frames(index,STAGES)):actual=model(x['source_image'],x['prompt_hidden'])
            simacc.update(simulation,x);actualacc.update(actual,x)
            rows.append(dict(test_index=index,sample_id=dataset.records[index]['sample_id']))
    write(out/'report.json',dict(status='complete',contract=contract,simulation_metrics=simacc.compute(),physical_metrics=actualacc.compute(),capture_statistics=stats,samples=rows,elapsed_seconds=time.time()-started))
    write(out/'progress.json',dict(status='complete',total=total,ccd_count=total*6,completed_stages=list(STAGES)))
    if total<=4:
        assert all(max(row['p99'] for row in stats[stage])>12 for stage in STAGES),'Pilot near camera background; full capture not authorized by health check'
if __name__=='__main__':main()
