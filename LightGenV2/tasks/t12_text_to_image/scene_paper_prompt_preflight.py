"""Actual full RGB/text generation on the informative v7 requests, without training."""
import argparse
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--paper-backend',choices=['electronic','middle','omnigen'],required=True)
    p.add_argument('--middle-profile',type=Path);custom,remaining=p.parse_known_args()
    from .scene_paper_prompt_protocol import checked_revision,PARENT_PAIRS
    from . import scene_natural_prompt as prompts
    old=prompts.checked_revision;prompts.checked_revision=checked_revision
    sys.argv=[sys.argv[0],*remaining]
    try:
        if custom.paper_backend=='electronic':
            from .scene_paired_render import main as render
            render()
        elif custom.paper_backend=='middle':
            if not custom.middle_profile:raise ValueError('Exact trained middle graph required')
            from .scene_R2_middle_protocol import checked_profile
            from .scene_compact_candidates import rendering
            rendering(checked_profile(json.loads(custom.middle_profile.read_text())),8,remaining,expected_contexts=48)
        else:
            from . import scene_omnigen_peer as peer
            original=peer.matched_visible_inputs
            def checked(data):
                refs=json.loads((data/'inference_dev_inputs.json').read_text());rows,protocol=checked_revision(data,data,{'pair_sha256':PARENT_PAIRS},refs)
                return rows,dict(protocol,peer_has_no_task_tuning=True,paired_model_not_loaded=True)
            peer.matched_visible_inputs=checked
            try:peer.main()
            finally:peer.matched_visible_inputs=original
    finally:prompts.checked_revision=old


if __name__=='__main__':main()
