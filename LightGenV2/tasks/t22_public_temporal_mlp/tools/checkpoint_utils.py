"""Checkpoint fingerprints and a metadata-minimal public export."""
from pathlib import Path
import hashlib

import torch


def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def tensor_fingerprint(state):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        value=value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((value.dtype,tuple(value.shape))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def export_public_checkpoint(source,destination):
    payload=torch.load(source,map_location='cpu',weights_only=True)
    metadata=payload['metadata']
    fingerprint=tensor_fingerprint(payload['model'])
    source_hash=file_sha256(source)
    reference=metadata.get('metrics',{}).get('srcc',metadata.get('evaluation',{}).get('cached_srcc'))
    public_metadata={
        'role':'PUBLIC_REFERENCE_INFERENCE',
        'trained_checkpoint_sha256':metadata.get('trained_checkpoint_sha256',source_hash),
        'model_state_sha256':fingerprint,
        'upstream_model':'Qwen/Qwen3-VL-2B-Instruct',
        'license':'Apache-2.0',
        'evaluation':{'scope':'original_TEST_development','videos':558,'cached_srcc':reference},
    }
    front_source = metadata.get('front_source', metadata.get('parent_training_metadata', {}).get('front_source', {}))
    if front_source:
        public_metadata['upstream_revision'] = front_source.get('qwen_revision')
        public_metadata['fixed_prompt'] = front_source.get('prompt')
    components = {name: value for name, value in payload['model'].items()
                  if name.startswith('front.') or name == 'prompt_embedding'}
    if components:
        public_metadata['frozen_component_state_sha256'] = tensor_fingerprint(components)
    if 'dataset' in metadata:
        public_metadata['dataset']={'manifest':{'sha256':metadata['dataset']['manifest']['sha256']}}
    public={key:payload[key] for key in ('schema','architecture','config','model')}
    public['metadata']=public_metadata
    destination=Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    torch.save(public,destination)
    restored=torch.load(destination,map_location='cpu',weights_only=True)
    if tensor_fingerprint(restored['model'])!=fingerprint:
        raise ValueError('Public export changed model tensors')
    return {'input_checkpoint_sha256':source_hash,'public_checkpoint_sha256':file_sha256(destination),
            'model_state_sha256':fingerprint,'model_tensors_bitwise_equal':True}
