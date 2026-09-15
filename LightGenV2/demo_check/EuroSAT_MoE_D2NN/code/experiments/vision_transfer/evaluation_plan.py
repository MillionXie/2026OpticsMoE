"""Seal terminal outcomes and validation-only choices before official tests."""
import hashlib
import json
from pathlib import Path

STAGES=('moe_A','moe_reserved_B','moe_all_B','d2nn_A','d2nn_d2nn_B')


def make_evaluation_plan(out):
    out=Path(out).resolve();plan={}
    for name in STAGES:
        final_path=out/name/'final.json'
        if not final_path.exists():raise RuntimeError('Stage is not terminal: '+name)
        final=json.loads(final_path.read_text())
        if final.get('status') not in ('complete','routing_failed','blocked') or final.get('smoke'):
            raise RuntimeError('Stage is unfinished or is only a smoke run: '+name)
        checkpoint=final.get('checkpoint')
        digest=None
        if checkpoint:
            checkpoint=Path(checkpoint).resolve()
            if not checkpoint.is_relative_to(out/name) or not checkpoint.is_file():
                raise RuntimeError('Checkpoint is outside its stage or is missing: '+name)
            digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        elif final['status']!='blocked':raise RuntimeError('Terminal trained stage has no selected checkpoint: '+name)
        plan[name]=dict(training_status=final['status'],epochs_completed=final['epochs_completed'],planned_epochs=final['planned_epochs'],
                        routing_qualified=final.get('routing_qualified',False),protocol_succeeded=final.get('protocol_succeeded',False),
                        retention_satisfied=final.get('retention_satisfied'),reason=final.get('failure_reason',final.get('reason')),
                        checkpoint=None if checkpoint is None else str(checkpoint),checkpoint_file_sha256=digest)
    return plan


def seal_plan(out,plan):
    path=Path(out)/'evaluation_plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:
        raise RuntimeError('Validation-selected checkpoints changed after the evaluation plan was sealed')
    if not path.exists():path.write_text(json.dumps(plan,indent=2),encoding='utf-8')
