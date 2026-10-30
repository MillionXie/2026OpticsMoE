"""TRAIN-only full electronic path targets and explicitly compressed few-step LG.

Keeps the accepted F9 head, single E/G, CFG7.5, original VAE and512 output.
Neither fewer parameters nor fewer steps establishes a quality/speed result.
"""
import argparse
import json
import os
from pathlib import Path
import sys
from .scene_depth_probe import checked_config,prune_unet,prune_frontend
from .scene_depth_recovery import expand_lora


def selected_edges(teacher_steps,steps):
    if teacher_steps!=30 or steps not in (6,8):raise ValueError('Explicit30-to6/8 full-teacher path required')
    return [i*teacher_steps//steps for i in range(steps+1)]


def regrid_paths(saved,steps):
    import torch
    from .scene_tail_distill import collapsed_epsilon
    edges=selected_edges(saved['teacher_steps'],steps);records=[]
    if (len(saved['sigmas'])!=31 or len(saved['timesteps'])!=30 or saved['sigmas'][-1]!=0
        or not all(x>y for x,y in zip(saved['sigmas'],saved['sigmas'][1:]))):
        raise ValueError('Complete descending original thirty-step path required')
    for key,waypoints in saved['paths'].items():
        if waypoints.shape!=(31,4,64,64) or not torch.isfinite(waypoints).all():raise ValueError('Invalid full teacher waypoints')
        for index,(start,end) in enumerate(zip(edges,edges[1:])):
            effective=collapsed_epsilon(waypoints[start],waypoints[end],saved['sigmas'][start],saved['sigmas'][end])
            assert torch.isfinite(effective).all()
            records.append(dict(split='train',input_key=key,interval=index,x=waypoints[start].clone(),
                target_epsilon=effective,sigma=saved['sigmas'][start],timestep=saved['timesteps'][start]))
    return records,edges


def targets(a):
    import torch
    from .scene_tail_distill import load_cache
    from .white_reference_scene_probe import digest,write
    report=json.loads((a.source/'report.json').read_text());path=a.source/'TRAIN_full_teacher_paths.pt'
    assert report['complete'] and report['teacher_is_full_electronic_Qwen'] and not report['DEV_or_TEST_targets_created']
    assert digest(path)==report['full_TRAIN_teacher_paths_sha256']
    saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    cache,cr=load_cache(a.student_cache)
    train={r['input_key'] for r in cache['records'] if r['split']=='train'}
    dev={r['input_key'] for r in cache['records'] if r['split']=='dev'}
    assert saved['all_TRAIN'] and not set(saved['paths'])&dev and set(saved['paths'])<=train
    assert saved['pair_sha256']==cache['pair_sha256'] and len(saved['paths'])==192
    assert saved['teacher_sha256']=='662ff90f9a46670b28bf86d61629c59e4fb619ed69a62e4ae52ed8527ea76f76'
    records,edges=regrid_paths(saved,a.steps)
    grid=[saved['timesteps'][i] for i in edges[:-1]];sigmas=[saved['sigmas'][i] for i in edges]
    from diffusers import EulerDiscreteScheduler
    scheduler=EulerDiscreteScheduler.from_pretrained(a.model_path,subfolder='scheduler',local_files_only=True)
    scheduler.set_timesteps(timesteps=grid);torch.testing.assert_close(scheduler.sigmas,torch.tensor(sigmas),rtol=1e-5,atol=1e-5)
    out=dict(records=records,pair_sha256=saved['pair_sha256'],teacher_sha256=saved['teacher_sha256'],
        teacher_cache_sha256=saved['teacher_cache_sha256'],teacher_steps=30,student_steps=a.steps,
        timesteps=grid,sigmas=sigmas,noise_index=saved['noise_index'])
    torch.save(out,a.output/'TRAIN_trajectories.pt')
    write(a.output/'report.json',dict(complete=True,source_commit=os.environ['T12_SOURCE_COMMIT'],
        trajectory_sha256=digest(a.output/'TRAIN_trajectories.pt'),full_teacher_path_sha256=digest(path),
        TRAIN_contexts=192,TRAIN_intervals=len(records),student_steps=a.steps,teacher_steps=30,teacher_CFG=7.5,
        teacher_sha256=saved['teacher_sha256'],original_teacher_step_edges=edges,student_CFG=7.5,
        teacher_is_full_electronic_Qwen=True,teacher_GPU_forward_repeated=False,CPU_only=True,
        DEV_or_TEST_targets_created=False,TEST_opened=False,quality_or_speed_accepted=False))


def training(config,steps,remaining):
    import copy,torch
    from . import scene_tail_distill as trainer
    from .scene_fusion_aware_tail import install
    from .white_reference_scene_probe import digest,write
    old_load,old_train=trainer.load_generator,trainer.train
    def load(a,state,trainable=False):
        if not trainable:raise ValueError('TRAIN-only compact candidate')
        pipe,head=old_load(a,state,True);graph=prune_unet(pipe.unet,config['UNet_deep_keep'])
        from peft.tuners.lora.layer import Linear
        ref=copy.deepcopy(pipe.unet).eval().requires_grad_(False)
        for layer in ref.modules():
            if isinstance(layer,Linear):layer.merge(safe_merge=True,adapter_names=['scene_pair'])
        fusion=install(pipe.unet);held=[ref];del ref;forward=pipe.unet.forward
        def checked(*args,**kwargs):
            result=forward(*args,**kwargs)
            with torch.no_grad():expected=held[0](*args,**kwargs)
            assert torch.equal(result.sample,expected.sample),'Training must equal ordinary merged prediction'
            write(a.output/'fusion_GPU_identity.json',dict(complete=True,compressed_UNet_prediction_exact=True,graph=graph,fusion=fusion))
            held.clear();pipe.unet.forward=forward;return result
        pipe.unet.forward=checked;write(a.output/'compression_graph.json',graph);return pipe,head
    def train(a):
        state,parent=trainer.complete_state(a.student);ready=json.loads((a.trajectories/'report.json').read_text())
        assert parent['best_sha256']==config['parent_generator_sha256'] and not state.get('depth_compression')
        assert a.student_guidance==7.5 and not a.adapt_head and ready['student_steps']==steps
        assert ready['TRAIN_contexts']==192 and not ready['DEV_or_TEST_targets_created']
        old_train(a);out=json.loads((a.output/'report.json').read_text())
        descriptor=dict(schema='scene_compact_recovered_v1',profile=config,sampling_steps=steps,
            parent_checkpoint_sha256=parent['best_sha256'],parent_head_unchanged=True,
            original_UNet_tensor_names_restored=True,active_graph_requires_explicit_depth_loader=True)
        if a.stage=='train':
            for filename in ('best_checkpoint.pt','last_checkpoint.pt'):
                p=a.output/filename;new=torch.load(p,map_location='cpu',weights_only=False)
                assert new['sampling_contract']['steps']==steps and new['sampling_contract']['CFG']==7.5
                assert set(new['head'])==set(state['head']) and all(torch.equal(v,state['head'][k])for k,v in new['head'].items())
                new['unet_lora']=expand_lora(state['unet_lora'],new['unet_lora'],config['UNet_deep_keep'])
                assert set(new['unet_lora'])==set(state['unet_lora'])
                new['depth_compression']=descriptor;torch.save(new,p)
            out.update(best_sha256=digest(a.output/'best_checkpoint.pt'),last_sha256=digest(a.output/'last_checkpoint.pt'))
        out.update(depth_compression=descriptor,compression_training_performed=a.stage=='train',sampling_steps=steps)
        write(a.output/'report.json',out)
    trainer.load_generator,trainer.train=load,train;sys.argv=[sys.argv[0],*remaining]
    try:trainer.main()
    finally:trainer.load_generator,trainer.train=old_load,old_train


class GraphObserver:
    def __init__(self,config):self.config=config
    def prepare(self,qwen,processor,head,pipe):
        self.graph=dict(frontend=prune_frontend(qwen,6,4),backend=prune_unet(pipe.unet,self.config['UNet_deep_keep']))
        self.graph['backend']['weights_retrained']=True
        return processor
    def begin(self,row):pass
    def end(self,row):pass


def rendering(config,steps,remaining):
    import torch
    from . import scene_paired_render as renderer
    from .white_reference_scene_probe import digest,write
    old=renderer.render
    def render(a):
        report=json.loads((a.checkpoint/'report.json').read_text())
        state=torch.load(a.checkpoint/'best_checkpoint.pt',map_location='cpu',weights_only=False)
        descriptor=state.get('depth_compression',{})
        assert descriptor.get('schema')=='scene_compact_recovered_v1' and descriptor['profile']==config
        assert descriptor['sampling_steps']==steps==a.steps and report['compression_training_performed']
        assert digest(a.checkpoint/'best_checkpoint.pt')==report['best_sha256']
        assert a.scope=='dev' and a.prompt_revision and a.fuse_task_lora and a.sampler=='original' and not a.lcm_adapter
        a._verified_depth_compression=descriptor;observer=GraphObserver(config);a.request_observer=observer
        old(a);out=json.loads((a.output/'report.json').read_text())
        assert out['generated_count']==(8 if a.measure_software_latency else 128)
        out.update(compression_training_performed=True,experimental_compression=observer.graph,depth_compression=descriptor)
        write(a.output/'report.json',out)
    renderer.render=render;sys.argv=[sys.argv[0],*remaining]
    try:renderer.main()
    finally:renderer.render=old


def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--mode',choices=['targets','train','render'],required=True)
    p.add_argument('--compact-steps',type=int,choices=[6,8],required=True);p.add_argument('--compression-config',type=Path)
    a,remaining=p.parse_known_args()
    if a.mode=='targets':
        q=argparse.ArgumentParser();q.add_argument('--source',type=Path,required=True);q.add_argument('--student-cache',type=Path,required=True)
        q.add_argument('--model-path',type=Path,required=True);q.add_argument('--output',type=Path,required=True);args=q.parse_args(remaining)
        assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and len(os.environ.get('T12_SOURCE_COMMIT',''))==40 and not args.output.exists()
        args.output.mkdir(parents=True);args.steps=a.compact_steps;targets(args);return
    if not a.compression_config:raise ValueError('Explicit compression configuration required')
    config=checked_config(json.loads(a.compression_config.read_text()))
    if (config['vision_keep'],config['language_keep'])!=(6,4):raise ValueError('Preserve the accepted optical frontend')
    (training if a.mode=='train'else rendering)(config,a.compact_steps,remaining)


if __name__=='__main__':main()
