"""Recover a structurally reduced UNet using existing TRAIN Euler targets.

The ordinary paired renderer rejects these checkpoints unless this entry has
verified and installed the exact compressed graph. Old F9 remains unchanged.
"""
import argparse
import json
import os
from pathlib import Path
import re
import sys
from .scene_depth_probe import checked_config,keep_indices,prune_unet,Observer

DEEP = frozenset(['down_blocks.2.attentions.0.transformer_blocks',
    'down_blocks.2.attentions.1.transformer_blocks','mid_block.attentions.0.transformer_blocks',
    *['up_blocks.0.attentions.'+str(i)+'.transformer_blocks' for i in range(3)]])


def parent_key(name, depth):
    match=re.match(r'(.+\.transformer_blocks)\.(\d+)(\..+)',name)
    if match and match.group(1) in DEEP:
        ids=keep_indices(10,depth);index=int(match.group(2))
        if not 0<=index<len(ids):raise ValueError('Unknown compressed block index')
        return match.group(1)+'.'+str(ids[index])+match.group(3)
    return name


def expand_lora(parent, partial, depth):
    """Restore original tensor names; unused parent entries remain inert at inference."""
    result=dict(parent);seen=set()
    for name,value in partial.items():
        original=parent_key(name,depth)
        if original not in parent or original in seen or value.shape!=parent[original].shape:
            raise ValueError('Compressed LoRA cannot be mapped exactly to its original parent')
        seen.add(original);result[original]=value
    return result


def train_mode(config, remaining):
    import torch
    from . import scene_tail_distill as trainer
    from .scene_fusion_aware_tail import install
    from .white_reference_scene_probe import digest,write
    original_load,original_train=trainer.load_generator,trainer.train
    if (config['vision_keep'],config['language_keep'])!=(6,4) or config['UNet_deep_keep']==10:
        raise ValueError('This recovery changes UNet only; frozen features cannot train a changed frontend')
    def load(a,state,trainable=False):
        if not trainable:raise ValueError('Only the existing TRAIN target-bank recovery is supported')
        pipe,head=original_load(a,state,trainable)
        graph=prune_unet(pipe.unet,config['UNet_deep_keep'])
        # Independent ordinary PEFT merge proves the training arithmetic before
        # the first real backward. The copy is discarded after one forward.
        import copy
        from peft.tuners.lora.layer import Linear
        reference=copy.deepcopy(pipe.unet).eval().requires_grad_(False)
        for module in reference.modules():
            if isinstance(module,Linear):module.merge(safe_merge=True,adapter_names=['scene_pair'])
        fusion=install(pipe.unet);held=[reference];del reference
        forward=pipe.unet.forward
        def checked_forward(*args,**kwargs):
            result=forward(*args,**kwargs)
            with torch.no_grad():expected=held[0](*args,**kwargs)
            if not torch.equal(result.sample,expected.sample):
                raise ValueError('Compressed training forward differs from actual merged UNet')
            write(a.output/'fusion_GPU_identity.json',dict(complete=True,compressed_UNet_prediction_exact=True,
                source_commit=os.environ['T12_SOURCE_COMMIT'],graph=graph,fusion=fusion))
            held.clear();pipe.unet.forward=forward
            return result
        pipe.unet.forward=checked_forward
        write(a.output/'compression_config.json',config)
        write(a.output/'compression_graph.json',graph)
        return pipe,head
    def train(a):
        if a.stage not in ('smoke','train') or a.adapt_head or a.student_guidance!=7.5:
            raise ValueError('Keep the head frozen and the original guided Euler12 contract')
        state,report=trainer.complete_state(a.student)
        if report['best_sha256']!=config['parent_generator_sha256'] or state.get('depth_compression'):
            raise ValueError('Use the unchanged accepted F9 parent')
        if state.get('sampling_contract',{}).get('steps')!=12:
            raise ValueError('No simultaneous sampling-step change')
        original_train(a)
        result=json.loads((a.output/'report.json').read_text())
        descriptor=dict(schema='scene_recovered_depth_v1',profile=config,
            parent_checkpoint_sha256=report['best_sha256'],parent_head_unchanged=True,
            original_UNet_tensor_names_restored=True,active_graph_requires_explicit_depth_loader=True)
        result.update(depth_compression=descriptor,compression_training_performed=a.stage=='train',
            sampling_steps=12,partial_checkpoint_sha256_before_namespace_binding=result['best_sha256'])
        if a.stage=='train':
            for filename in ('best_checkpoint.pt','last_checkpoint.pt'):
                path=a.output/filename;saved=torch.load(path,map_location='cpu',weights_only=False)
                if set(saved['head'])!=set(state['head']) or not all(torch.equal(v,state['head'][k]) for k,v in saved['head'].items()):
                    raise ValueError('Frozen parent head changed during UNet recovery')
                saved['unet_lora']=expand_lora(state['unet_lora'],saved['unet_lora'],config['UNet_deep_keep'])
                assert set(saved['unet_lora'])==set(state['unet_lora'])
                saved['depth_compression']=descriptor
                torch.save(saved,path)
            result['best_sha256']=digest(a.output/'best_checkpoint.pt')
            result['last_sha256']=digest(a.output/'last_checkpoint.pt')
        write(a.output/'report.json',result)
    trainer.load_generator,trainer.train=load,train
    sys.argv=[sys.argv[0],*remaining]
    try:trainer.main()
    finally:trainer.load_generator,trainer.train=original_load,original_train


def render_mode(config,remaining):
    import torch
    from . import scene_paired_render as renderer
    from .white_reference_scene_probe import digest,write
    original=renderer.render
    def render(a):
        state=torch.load(a.checkpoint/'best_checkpoint.pt',map_location='cpu',weights_only=False)
        descriptor=state.get('depth_compression',{})
        report=json.loads((a.checkpoint/'report.json').read_text())
        if (descriptor.get('schema')!='scene_recovered_depth_v1' or descriptor.get('profile')!=config or
            report.get('depth_compression')!=descriptor or not report.get('compression_training_performed')):
            raise ValueError('Missing completed, bound compression recovery')
        if digest(a.checkpoint/'best_checkpoint.pt')!=report['best_sha256']:
            raise ValueError('Compressed weights changed')
        if (a.scope!='dev' or not a.prompt_revision or a.steps!=12 or not a.fuse_task_lora or
            a.sampler!='original' or a.lcm_adapter or a.measure_software_latency):
            raise ValueError('First validate all128 natural DEV requests; no TEST or formal timing yet')
        if (config['vision_keep'],config['language_keep'])!=(6,4):raise ValueError('Unexpected frontend compression')
        a._verified_depth_compression=descriptor
        observer=Observer(config);a.request_observer=observer
        original(a)
        rows=json.loads((a.output/'images.json').read_text());out=json.loads((a.output/'report.json').read_text())
        assert len(rows)==out['generated_count']==128
        observer.description.update(training_performed=True,depth_compression=descriptor)
        observer.description['backend']['weights_retrained']=True
        for row in rows:row['experimental_compression']=observer.description
        out.update(experimental_compression=observer.description,compression_training_performed=True,
                   compressed_parent_namespace_binding_verified=True)
        write(a.output/'images.json',rows);write(a.output/'report.json',out)
        write(a.output/'component_diagnostic.json',dict(complete=True,records=observer.records,
            one_process_only=True,formal_speed_claim=False,physical_optical_measurement=False))
    renderer.render=render;sys.argv=[sys.argv[0],*remaining]
    try:renderer.main()
    finally:renderer.render=original


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--mode',choices=['train','render'],required=True)
    parser.add_argument('--compression-config',type=Path,required=True)
    custom,remaining=parser.parse_known_args()
    config=checked_config(json.loads(custom.compression_config.read_text()))
    (train_mode if custom.mode=='train'else render_mode)(config,remaining)


if __name__=='__main__':main()
