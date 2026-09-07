"""Launch staged training on one verified RTX UUID, never a numeric ordinal."""
import argparse
import csv
import os
import subprocess
import sys


def validate_device(uuid, inventory):
    if not uuid.startswith('GPU-'):
        raise ValueError('Specify a full NVIDIA GPU UUID, not a numeric index')
    matches = [row for row in inventory if row[0].strip() == uuid]
    if len(matches) != 1 or 'RTX' not in matches[0][1] or 'A100' in matches[0][1]:
        raise ValueError('Only an exactly matched RTX GPU UUID is allowed')
    return matches[0][1].strip()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--gpu-uuid', required=True)
    args, training_args = parser.parse_known_args()
    inventory = csv.reader(subprocess.check_output(
        ['nvidia-smi', '--query-gpu=uuid,name', '--format=csv,noheader'], text=True).splitlines())
    expected = validate_device(args.gpu_uuid, list(inventory))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_uuid
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    import torch
    actual = torch.cuda.get_device_name(0)
    if torch.cuda.device_count() != 1 or actual != expected:
        raise RuntimeError(f'CUDA device mismatch: expected {expected}, got {actual}')
    print(f'Verified GPU: {args.gpu_uuid} / {actual}', flush=True)
    sys.argv = ['train_staged', *training_args]
    from .train_staged import main as train
    train()


if __name__ == '__main__':
    main()
