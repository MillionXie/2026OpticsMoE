"""Reuse the exact optical boundary/replay gate on the explicit visible-anchor DEV cohort."""
import argparse
import importlib
import sys


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--boundary-mode',choices=['export','packed-replay'],required=True)
    custom,remaining=parser.parse_known_args()
    from . import scene_natural_prompt as prompts
    from .scene_instruction_render import checked_revision
    old=prompts.checked_revision;prompts.checked_revision=checked_revision
    module='scene_optical_boundary_check' if custom.boundary_mode=='export' else 'scene_packed_CCD_latency'
    sys.argv=[sys.argv[0],*remaining]
    try:importlib.import_module('.'+module,__package__).main()
    finally:prompts.checked_revision=old


if __name__=='__main__':main()
