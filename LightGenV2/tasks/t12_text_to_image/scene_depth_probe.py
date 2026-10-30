"""Explicit target-free DEV architecture ablations of frozen F9.

This entry does not alter the ordinary renderer, train weights, open TEST, or
claim that removing blocks preserves quality. Sampling remains guided Euler12.
The full audited natural DEV list is checked before selecting scenes 0 and 1.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time


def keep_indices(depth, target):
    if type(depth) is not int or type(target) is not int or not 2 <= target <= depth:
        raise ValueError('Keep at least two ordered blocks, including both ends')
    ids = tuple((2*i*(depth-1)+(target-1))//(2*(target-1)) for i in range(target))
    assert ids[0] == 0 and ids[-1] == depth-1 and len(set(ids)) == target
    return ids


def checked_config(config):
    required = {'schema','vision_keep','language_keep','UNet_deep_keep',
                'parent_generator_sha256','parent_frontend_sha256'}
    if set(config) != required or config['schema'] != 'scene_depth_probe_v1':
        raise ValueError('Explicit immutable compression profile required')
    if config['vision_keep'] not in (4,6) or config['language_keep'] not in (2,3,4):
        raise ValueError('Unreviewed Qwen prefix budget')
    if config['UNet_deep_keep'] not in (6,8,9,10):
        raise ValueError('Unreviewed UNet depth')
    if config['parent_generator_sha256'] != 'f9fbdb7fd36d1c862686616a624e7c18f4e9149cef61b0704bbf97767de6eaec':
        raise ValueError('This probe is bound to the accepted F9 parent')
    if config['parent_frontend_sha256'] != '6f34f2d720060c9193cda0e19ffb8fb4d157bc03e3fd5e5c1433c60ad884471e':
        raise ValueError('This probe requires the single-EG F6 parent')
    return dict(config)


def prune_unet(unet, target):
    from torch import nn
    ids = keep_indices(10, target)
    groups = [(name,module) for name,module in unet.named_modules()
              if name.endswith('transformer_blocks') and isinstance(module,nn.ModuleList)]
    deep = [(name,module) for name,module in groups if len(module) == 10]
    if len(deep) != 6 or sorted(len(module) for _,module in groups) != [2]*5+[10]*6:
        raise ValueError('Require the exact SDXL base attention layout')
    before = sum(p.numel() for p in unet.parameters())
    if target != 10:
        for name,blocks in deep:
            parent,leaf = name.rsplit('.',1)
            unet.get_submodule(parent).__setattr__(leaf,nn.ModuleList([blocks[i] for i in ids]))
    after = sum(p.numel() for p in unet.parameters())
    count = sum(len(module) for name,module in unet.named_modules()
                if name.endswith('transformer_blocks') and isinstance(module,nn.ModuleList))
    assert count == 10+6*target
    if target != 10 and not after < before:
        raise ValueError('Structural pruning did not remove registered parameters')
    return dict(original_blocks=70,actual_blocks=count,deep_block_original_indices=list(ids),
                original_UNet_with_IP_parameters=before,actual_UNet_with_IP_parameters=after,
                removed_parameters=before-after,weights_retrained=False)


def prune_frontend(qwen, vision_keep, language_keep):
    from .qwen_vl_hybrid import IdentityBlock
    before = sum(p.numel() for p in qwen.parameters())
    counts = {}
    for label,blocks,old,target in [('vision',qwen.model.visual.blocks,6,vision_keep),
                                   ('language',qwen.model.language_model.layers,4,language_keep)]:
        native = type(blocks[0])
        if sum(type(block) is native for block in blocks) != old:
            raise ValueError('Require the original six/four single-EG parent')
        for i in range(target,old):
            if type(blocks[i]) is not native:raise ValueError('Prefix identity changed')
            blocks[i] = IdentityBlock()
        counts[label] = sum(type(block) is native for block in blocks)
        assert counts[label] == target
    if any(m.pairs != 1 for m in qwen.model.hybrid.values()):
        raise ValueError('Never add an optical E/G pair')
    if any(not .6 <= f.alpha.item() <= .900001 for m in qwen.model.hybrid.values() for f in m.fusions):
        raise ValueError('Optical fusion contract changed')
    return dict(retained_original_TF=counts,actual_removed_TF=52-sum(counts.values()),
                original_Qwen_parameters=before,actual_Qwen_parameters=sum(p.numel() for p in qwen.parameters()),
                optical_cycles=6,weights_retrained=False)


class Observer:
    def __init__(self, config):
        self.config = checked_config(config)
        self.records = []
        self.events = None

    def prepare(self,qwen,processor,head,pipe):
        import torch
        self.description = dict(schema='scene_depth_probe_actual_graph_v1',profile=self.config,
            frontend=prune_frontend(qwen,self.config['vision_keep'],self.config['language_keep']),
            backend=prune_unet(pipe.unet,self.config['UNet_deep_keep']),
            sampling_steps=12,CFG=7.5,resolution=512,training_performed=False,
            quality_or_speed_accepted=False,TEST_opened=False)
        def wrap(original,key):
            def call(*args,**kwargs):
                start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                result = original(*args,**kwargs)
                end.record()
                self.events[key].append((start,end))
                return result
            return call
        for module,key in [(qwen.model,'Qwen'),(head,'head'),(pipe.unet,'UNet')]:
            module.forward = wrap(module.forward,key)
        pipe.vae.decode = wrap(pipe.vae.decode,'VAE_decode')
        return processor

    def begin(self,row):
        import torch
        torch.cuda.synchronize()
        self.events = {k:[] for k in ('Qwen','head','UNet','VAE_decode')}
        self.started = time.perf_counter()

    def end(self,row):
        import torch
        torch.cuda.synchronize()
        self.records.append(dict(sample_id=row['sample_id'],category=row['category'],
            diagnostic_wall_ms=(time.perf_counter()-self.started)*1000,
            GPU_component_ms={key:sum(a.elapsed_time(b) for a,b in values) for key,values in self.events.items()},
            calls={key:len(values) for key,values in self.events.items()}))
        assert self.records[-1]['calls'] == dict(Qwen=1,head=1,UNet=12,VAE_decode=1)


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--compression-config',type=Path,required=True)
    custom,remaining = p.parse_known_args()
    config = checked_config(json.loads(custom.compression_config.read_text()))
    from . import scene_paired_render as renderer
    from . import scene_natural_prompt as prompts
    from .white_reference_scene_probe import digest,write
    original_render,original_revision = renderer.render,prompts.checked_revision
    def selected_revision(*args):
        rows,metadata = original_revision(*args)
        selected = [row for row in rows if row['condition_id'] in ('0','1')]
        if len(selected) != 32 or len({r['reference_id'] for r in selected}) != 16:
            raise ValueError('Expected two fixed scenes for all sixteen DEV references')
        return selected,dict(metadata,contexts=32,full_audited_contexts=128,
            deterministic_scene_subset=['0','1'],selection_by_quality=False)
    def candidate(a):
        if a.scope != 'dev' or a.frozen_evaluation or a.measure_software_latency or not a.prompt_revision:
            raise ValueError('Diagnostic target-free natural DEV only; never frozen TEST or formal timing')
        if a.steps != 12 or not a.fuse_task_lora or a.sampler != 'original' or a.lcm_adapter:
            raise ValueError('Keep the accepted merged guided Euler12 sampler')
        for folder,key in [(a.checkpoint,'parent_generator_sha256'),(a.lg_frontend,'parent_frontend_sha256')]:
            if folder is None or digest(folder/'best_checkpoint.pt') != config[key]:
                raise ValueError('Parent identity mismatch')
        write(a.output/'compression_config.json',config)
        observer = Observer(config)
        a.request_observer = observer
        original_render(a)
        report = json.loads((a.output/'report.json').read_text())
        rows = json.loads((a.output/'images.json').read_text())
        report.update(experimental_compression=observer.description,compression_training_performed=False,
                      scope='DEV_architecture_diagnostic',quality_accepted=False)
        for row in rows:
            row['experimental_compression'] = observer.description
            row['optical_frontend'].update(observer.description['frontend'])
        report['optical_frontend'].update(observer.description['frontend'])
        assert report['generated_count'] == len(observer.records) == len(rows) == 32
        write(a.output/'images.json',rows)
        write(a.output/'report.json',report)
        write(a.output/'component_diagnostic.json',dict(complete=True,records=observer.records,
            one_process_only=True,first_request_retained=True,extra_warmup=False,
            scope='instrumented software diagnostic; not formal timing or optical hardware',
            source_commit=os.environ['T12_SOURCE_COMMIT']))
    renderer.render,prompts.checked_revision = candidate,selected_revision
    sys.argv = [sys.argv[0],*remaining]
    try:renderer.main()
    finally:renderer.render,prompts.checked_revision = original_render,original_revision


if __name__ == '__main__':main()
