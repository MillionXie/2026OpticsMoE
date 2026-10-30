"""Frozen independent TEST entry for an explicitly reconstructed compact graph.

No TEST request is accessed until the selected checkpoint, graph, sampling
contract and completed development image review are bound in a freeze manifest.
This does not waive the existing role-disjoint input, asset or pixel checks.
"""
import argparse
import json
from pathlib import Path
import sys


def verify_compact_freeze(state,ready,frozen,key,review,config):
    model=frozen['models'][key];description=state.get('depth_compression',{})
    if (frozen['schema']!='scene_frozen_evaluation_v1' or not frozen['development_selection_finished']
        or frozen['TEST_used_for_model_selection'] or not frozen['frozen_before_first_TEST_prediction']
        or not ready['complete'] or not ready['compression_training_performed']
        or description.get('schema')!='scene_compact_recovered_v1' or description.get('profile')!=config
        or model.get('depth_compression')!=description or model['checkpoint_sha256']!=ready['best_sha256']
        or model['sampling_contract']!=state['sampling_contract'] or model['sampling_steps']!=state['sampling_contract']['steps']
        or not review['complete'] or not review['all128_contexts_reviewed'] or not review['all16_boards_reviewed']
        or review['checkpoint_sha256']!=ready['best_sha256'] or not review['proxy_all12_mean_gate_pass']
        or not review['approved_as_final_lightweight_model'] or review['independent_TEST_opened']):
        raise ValueError('Freeze must bind a completed, development-approved compact checkpoint and exact graph')
    return description


def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--compression-config',type=Path,required=True)
    p.add_argument('--development-review',type=Path,required=True);custom,remaining=p.parse_known_args()
    from . import scene_paired_render as renderer
    from .scene_compact_candidates import GraphObserver
    from .scene_depth_probe import checked_config
    from .white_reference_scene_probe import digest,write
    import torch
    config=checked_config(json.loads(custom.compression_config.read_text()));old=renderer.render
    def render(a):
        if a.scope!='test' or a.measure_software_latency or a.prompt_revision or a.expanded_development:
            raise ValueError('Only a previously frozen independent TEST scope; no DEV selection or timing')
        state=torch.load(a.checkpoint/'best_checkpoint.pt',map_location='cpu',weights_only=False)
        ready=json.loads((a.checkpoint/'report.json').read_text());frozen=json.loads(a.frozen_evaluation.read_text())
        review=json.loads(custom.development_review.read_text())
        assert digest(a.checkpoint/'best_checkpoint.pt')==ready['best_sha256']
        model=frozen['models'][a.frozen_model_key]
        assert model['development_review_sha256']==digest(custom.development_review)
        assert model['compression_config_sha256']==digest(custom.compression_config)
        a._verified_depth_compression=verify_compact_freeze(state,ready,frozen,a.frozen_model_key,review,config)
        if not a.fuse_task_lora or a.sampler!='original' or a.lcm_adapter:raise ValueError('Only the development-reviewed merged sampler graph')
        observer=GraphObserver(config);a.request_observer=observer;old(a)
        path=a.output/'report.json';report=json.loads(path.read_text())
        assert report['scope']=='TEST'and report['generated_count']==500
        report.update(compression_training_performed=True,depth_compression=a._verified_depth_compression,
            actual_compressed_graph=observer.graph,compact_freeze_source_sha256=digest(Path(__file__)),
            development_review_sha256=digest(custom.development_review),new_TEST_scores_not_for_tuning=True)
        write(path,report)
    renderer.render=render;sys.argv=[sys.argv[0],*remaining]
    try:renderer.main()
    finally:renderer.render=old


if __name__=='__main__':main()
