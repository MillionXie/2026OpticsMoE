"""Pin the original TRAIN-only head without importing an old paired UNet LoRA."""
import json
from .scene_instruction_data import sha


def checked_alignment(root,state,original_sha=None):
    name='alignment_report.json'if original_sha is not None else'adaptation_report.json'
    report=json.loads((root/name).read_text(encoding='utf8'));actual=sha(root/'best_checkpoint.pt')
    if not report.get('complete')or actual!=report['best_sha256']:raise ValueError('Unfinished or changed initial head')
    if original_sha is not None:
        if len(original_sha)!=64 or actual!=original_sha or report.get('scene_GT_used')is not False or state.get('schema')!='fresh_qwen3_sdxl_conditions_v1'or not state.get('not_yet_diffusion_finetuned')or state.get('step',0)<=0:
            raise ValueError('Explicit completed original TRAIN-only alignment required')
    return dict(checkpoint_sha256=actual,report_sha256=sha(root/name),report_name=name,original_unpaired_alignment=original_sha is not None)
