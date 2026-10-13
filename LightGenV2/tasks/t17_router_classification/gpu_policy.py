"""Reject A100 before any PATCH16 trainer initializes a CUDA context."""
import csv
import io
import os
import subprocess


def gpu_inventory():
    text = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=uuid,name', '--format=csv,noheader'], text=True)
    return {row[0].strip(): row[1].strip()
            for row in csv.reader(io.StringIO(text)) if len(row) == 2}


def validate_gpu_uuids(uuids, inventory):
    if not uuids or len(set(uuids)) != len(uuids):
        raise ValueError('distinct explicit GPU UUIDs required')
    for uuid in uuids:
        if not uuid.startswith('GPU-') or uuid not in inventory:
            raise ValueError('use a verified full GPU UUID, not a CUDA numeric index')
        if 'A100' in inventory[uuid].upper():
            raise ValueError('user prohibits all A100 GPUs for T17')
    return uuids


def require_non_a100_gpu():
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    uuids = [value.strip() for value in visible.split(',') if value.strip()]
    if len(uuids) != 1:
        raise ValueError('each PATCH16 trainer requires one explicit GPU UUID')
    validate_gpu_uuids(uuids, gpu_inventory())
    return uuids[0]
