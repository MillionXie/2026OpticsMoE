"""Read-only frozen CPU evaluation of original lab archives.

Executed from the audited script via stdin, not installed over lab sources.
All models/metrics come from each archive's original runtime. Output is JSONL
on stdout; no remote files, GPU contexts, training or camera/SLM calls.
"""
from __future__ import annotations
import argparse,hashlib,json,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def emit(kind,**values):print(json.dumps(dict(kind=kind,**values),ensure_ascii=True),flush=True)
def state_sha(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def sources(modules):return {m.__name__:dict(path=m.__file__,sha256=sha(m.__file__)) for m in modules}

def salicon(root,limit):
    import LightGenV2.tasks.t03_saliency.lab_runtime as runtime
    import experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency.objectives as objectives
    from LightGenV2.tasks.t03_saliency.reproduce_baseline import independent_cc
    session=root/'sessions/full01';old=read(session/'results.json');contract=read(session/'session.json');release=read(root/'release.json');cfg=read(root/'LAB.20260914.json')
    assert old['checkpoint_sha256']==runtime.CHECKPOINT_SHA and old['samples']==5000
    assert contract['release_sha256']==sha(root/'release.json')
    assert old['release_sha256']==sha(root/'release.json')
    assert contract['measured_stages']==list(runtime.STAGES)
    assert [r['sample_id'] for r in contract['fields']]==[r['sample_id'] for r in old['rows']]
    before_checkpoint=sha(root/'weights/best_checkpoint.pt')
    model=runtime.load_model(root,'cpu').eval().requires_grad_(False);initial=state_sha(model)
    accumulators={s:objectives.SaliencyAccumulator() for s in ('Sim','Exp')}
    rows=[];input_evidence=[];t0=time.monotonic()
    with torch.inference_mode():
        for i,(item,reference) in enumerate(zip(contract['fields'][:limit],old['rows'][:limit])):
            p=root/item['file'];assert sha(p)==item['sha256'];batch=torch.load(p,map_location='cpu',weights_only=False)
            frames={}
            for stage in runtime.STAGES:
                p=session/'ccd'/stage/(item['key']+'.png');assert sha(p)==reference['ccd_sha256'][stage]
                image=np.array(Image.open(p),dtype=np.float32)
                assert image.shape==(478,478)
                frames[stage]=torch.from_numpy(image[None])*float(cfg.get('detector_intensity_scale',{}).get(stage,1/255))
            input_evidence.append(dict(sample_id=item['sample_id'],input_sha256=item['sha256'],ccd_sha256=reference['ccd_sha256']))
            for stage,measured in [('Sim',None),('Exp',frames)]:
                logits,tap=runtime.replay(model,batch,measured);assert tap.index==3
                acc=objectives.SaliencyAccumulator();acc.update(logits,batch['density'],batch['fixation']);values=acc.compute();values.pop('samples')
                accumulators[stage].update(logits,batch['density'],batch['fixation'])
                cc=float(independent_cc(objectives.density_from_logits(logits).numpy(),batch['density'].numpy())[0]);ref=item['simulation_cc'] if stage=='Sim' else reference['cc_float64']
                if abs(cc-ref)>3e-4:raise ValueError('Frozen per-image CC differs: '+stage+'/'+item['sample_id'])
                row=dict(stage=stage,sample_id=item['sample_id'],cc_float64=cc,original_cc_float64=ref,**values);rows.append(row);emit('row',row=row)
            if i==0 or (i+1)%100==0:emit('progress',done=i+1,total=5000,elapsed_seconds=time.monotonic()-t0)
    metrics={k:v.compute() for k,v in accumulators.items()}
    differences=[]
    if limit==5000:
        for k,v in old['metrics'].items():
            if k!='samples' and abs(metrics['Exp'][k]-v)>3e-5:differences.append(dict(stage='Exp',metric=k,replayed=metrics['Exp'][k],archived=v))
        if abs(metrics['Sim']['cc']-old['simulation_same_subset_cc'])>3e-5:differences.append(dict(stage='Sim',metric='cc',replayed=metrics['Sim']['cc'],archived=old['simulation_same_subset_cc']))
    assert initial==state_sha(model) and before_checkpoint==sha(root/'weights/best_checkpoint.pt')
    return dict(task='salicon',expected_samples=5000,completed_samples=limit,metrics=metrics,rows=rows,checkpoint_sha256=before_checkpoint,weights_unchanged_sha256=initial,input_evidence=input_evidence,source_modules=sources([runtime,objectives]),aggregate_validation_passed=not differences,aggregate_differences=differences,elapsed_seconds=time.monotonic()-t0)

def openmoji(root,limit):
    import LightGenV2.tasks.t04_semantic_interaction.lab_runtime as runtime
    import LightGenV2.tasks.t04_semantic_interaction.modeling as modeling
    import experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.metrics as metric_module
    from LightGenV2.tasks.t04_semantic_interaction.settings import load_settings
    from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.datasets import OpenMojiEditingDataset,collate_samples,load_prompt_cache
    session=root/'runs/full1000_20260925_01';contract=read(session/'run_contract.json');old=read(session/'report.json')
    checkpoint=root/'weights/selected_checkpoint.pt';assert sha(checkpoint)==contract['checkpoint_sha256']==old['checkpoint_sha256']=='03cb861c3ac344556601eb3eb6d7d1a22b77a54d2e7e68e85d77ee30fb09eb21'
    cfg=load_settings(root/'runtime_exact/LightGenV2/tasks/t04_semantic_interaction/configs/layered_scene_exp05_dc30_ccdsmall.yaml');cfg.data_dir=root/'data';cfg.prompt_cache_path=cfg.data_dir/'token_embeddings_v1.pt';cfg.qwen_checkpoint=root.parent/'OpenMoji_Lab_SHS_8um/frontend';cfg.asset_dir=root.parent/'OpenMoji_Lab_SHS_8um/assets'
    assert sha(cfg.test_manifest)==contract['test_manifest_sha256'] and sha(cfg.prompt_cache_path)==contract['prompt_cache_sha256']
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False);model=modeling.build_model(cfg,torch.device('cpu'))
    assert payload['epoch']==45 and payload['architecture']==model.checkpoint_architecture
    model.load_state_dict(payload['model'],strict=True);model.eval().requires_grad_(False)
    for p in model._optical_paths():p.set_phase_dropout_active(False)
    initial=state_sha(model);data=OpenMojiEditingDataset(cfg.test_manifest,cfg,load_prompt_cache(cfg.prompt_cache_path));assert len(data)==1000
    phases={stage:sha(session/'phase'/(stage+'.bmp')) for stage in runtime.STAGES}
    accumulator=metric_module.MetricAccumulator();rows=[];evidence=[];camera_readback={};t0=time.monotonic()
    with torch.inference_mode():
        for i in range(limit):
            record=data.records[i];sid=record['sample_id'];batch=collate_samples([data[i]]);frames={};hashes={}
            for stage in runtime.STAGES:
                p=session/'ccd'/stage/(sid+'.png');rec=read(p.with_suffix('.json'))
                assert rec['sample_id']==sid and rec['phase_sha256']==phases[stage] and rec['no_photometric_normalization']
                # Preserve the camera's ACTUAL readback, not the requested
                # rounded value: the original receipts report 20002 us for
                # a 20000-us request. Require exact constancy within a plane.
                observed=dict(exposure=rec['exposure'],wait_ms=rec['wait_ms'])
                if stage in camera_readback:assert observed==camera_readback[stage],('Camera readback changed',sid,stage)
                else:camera_readback[stage]=observed
                assert rec['wait_ms']==contract['wait_ms']
                array=np.array(Image.open(p),dtype=np.float32);assert array.shape==(478,478)
                frames[stage]=torch.from_numpy(array[None]/255);hashes[stage]=sha(p)
            with runtime.OpticalBoundary(model,frames) as tap:output=model(batch['source_image'],batch['prompt_hidden'])
            assert tap.index==6
            records,pred,edit=accumulator.update(output,batch);row=records[0];row.update(predicted_grid=pred[0].tolist(),predicted_edit=edit[0].to(torch.uint8).tolist());rows.append(row);emit('row',row=row)
            evidence.append(dict(sample_id=sid,source_image_sha256=sha(cfg.data_dir/record['relative_dir']/'source.png'),ccd_sha256=hashes))
            if i==0 or (i+1)%100==0:emit('progress',done=i+1,total=1000,elapsed_seconds=time.monotonic()-t0)
    metrics=accumulator.compute();differences=[]
    if limit==1000:
        for group,values in old['metrics'].items():
            for k,v in values.items():
                if abs(metrics[group][k]-v)>2e-6:differences.append(dict(group=group,metric=k,replayed=metrics[group][k],archived=v))
    assert initial==state_sha(model) and sha(checkpoint)==contract['checkpoint_sha256']
    return dict(task='openmoji',expected_samples=1000,completed_samples=limit,metrics=metrics,rows=rows,checkpoint_sha256=contract['checkpoint_sha256'],weights_unchanged_sha256=initial,test_manifest_sha256=contract['test_manifest_sha256'],phase_sha256=phases,requested_camera_exposure_us=contract['exposure_us'],observed_camera_settings=camera_readback,input_evidence=evidence,source_modules=sources([runtime,modeling,metric_module]),aggregate_validation_passed=not differences,aggregate_differences=differences,elapsed_seconds=time.monotonic()-t0)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task',choices=['salicon','openmoji'],required=True);p.add_argument('--limit',type=int);a=p.parse_args()
    total={'salicon':5000,'openmoji':1000}[a.task];limit=a.limit or total;assert 1<=limit<=total
    root=Path('E:/code/guest/2026OpticsMoE')/('SALICON_Lab_SHS_8um' if a.task=='salicon' else 'OpenMoji_Lab_DVP_8um_e45')
    sys.path.insert(0,str(root/('runtime' if a.task=='salicon' else 'runtime_exact')))
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(73)
    emit('start',task=a.task,requested=limit,torch=torch.__version__,device='cpu',hardware_called=False,remote_writes=False)
    result=globals()[a.task](root,limit);result.update(mode='frozen_original_CCD_cpu_replay',torch=torch.__version__,device='cpu',threads=2,training=False,hardware_called=False,remote_writes=False,complete=limit==total)
    assert not torch.cuda.is_initialized(),'CPU-only replay unexpectedly initialized CUDA'
    emit('result',result=result)
if __name__=='__main__':main()
