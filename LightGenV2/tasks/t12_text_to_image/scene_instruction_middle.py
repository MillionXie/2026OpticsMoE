"""Apply the existing compressed trainer and optical replay to the fixed full R2 DEV48."""
import argparse
import importlib
import json
from pathlib import Path
import sys
from .scene_R2_middle_protocol import checked_profile,checked_training


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--middle-mode',choices=['train','render','export','packed-replay'],required=True)
    parser.add_argument('--middle-profile',type=Path,required=True)
    custom,remaining=parser.parse_known_args();config=checked_profile(json.loads(custom.middle_profile.read_text()))
    from . import scene_compact_candidates as compact
    if custom.middle_mode=='train':
        compact.training(config,8,remaining,admission=lambda state,parent,targets,a:checked_training(config,state,parent,targets,a.student_guidance,a.adapt_head))
        return
    from . import scene_natural_prompt as prompts
    from .scene_instruction_render import checked_revision
    old_revision=prompts.checked_revision;prompts.checked_revision=checked_revision
    if custom.middle_mode=='render':
        try:compact.rendering(config,8,remaining,expected_contexts=48)
        finally:prompts.checked_revision=old_revision
        return
    import torch
    from . import scene_paired_render as renderer
    from .white_reference_scene_probe import digest
    old_render=renderer.render
    def render(a):
        state=torch.load(a.checkpoint/'best_checkpoint.pt',map_location='cpu',weights_only=False)
        report=json.loads((a.checkpoint/'report.json').read_text());description=state.get('depth_compression',{})
        if(description.get('schema')!='scene_compact_recovered_v1' or description.get('profile')!=config
           or report.get('depth_compression')!=description or not report.get('compression_training_performed')
           or digest(a.checkpoint/'best_checkpoint.pt')!=report['best_sha256'] or a.steps!=8 or a.scope!='dev'):
            raise ValueError('Exact completed R2 middle graph required')
        a._verified_depth_compression=description
        prior=getattr(a,'request_observer',None);graph=compact.GraphObserver(config)
        if prior is None:a.request_observer=graph
        else:
            prepare=prior.prepare
            def both(qwen,processor,head,pipe):return prepare(qwen,graph.prepare(qwen,processor,head,pipe),head,pipe)
            prior.prepare=both
        return old_render(a)
    renderer.render=render;sys.argv=[sys.argv[0],*remaining]
    module='scene_optical_boundary_check'if custom.middle_mode=='export'else'scene_packed_CCD_latency'
    try:importlib.import_module('.'+module,__package__).main()
    finally:renderer.render=old_render;prompts.checked_revision=old_revision


if __name__=='__main__':main()
