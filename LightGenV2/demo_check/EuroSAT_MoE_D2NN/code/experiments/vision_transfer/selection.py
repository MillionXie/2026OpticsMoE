"""Selection metadata preserves failed arms instead of inventing completion."""
from pathlib import Path


def validate_source(source, architecture, version, policy_hash, smoke):
    if source.get('task') != 'A' or source.get('architecture') != architecture:
        raise RuntimeError('B requires a source A checkpoint of the same architecture')
    if source.get('version') != version or source.get('policy_sha256') != policy_hash:
        raise RuntimeError('Source belongs to another training protocol; V2 is not a Vision-only source')
    if bool(source.get('smoke',False)) != bool(smoke):
        raise RuntimeError('Smoke and formal checkpoints cannot be mixed')
    if not smoke and not source.get('routing_qualified',False):
        raise RuntimeError('Source A has not passed automatic routing qualification')


def selected_path(directory, task, qualified):
    directory = Path(directory)
    if task == 'A':
        candidates = ['best.pt'] if qualified else ['best_observed.pt']
    else:
        candidates = (['best_retained.pt','best_mean.pt'] if qualified else
                      ['best_retained.pt','best_mean.pt','best_observed_retained.pt','best_observed_mean.pt'])
    for filename in candidates:
        path = directory / filename
        if path.exists():
            return path
    raise RuntimeError('No validation-selected checkpoint was saved')
