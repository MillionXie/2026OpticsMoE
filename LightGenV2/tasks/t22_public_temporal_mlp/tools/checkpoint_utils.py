"""Checkpoint fingerprints and a metadata-minimal public export."""
from pathlib import Path
from dataclasses import replace
import hashlib
import math
import tempfile

import torch


def share_global_phases(model):
    """Initialize one learnable phase tile per branch from the parent tiles."""
    if model.cfg.shared_global_phase:
        return model
    state = model.state_dict()
    for name in ('frame_global.raw_global_phase', 'video_global.raw_global_phase'):
        phase = 2 * math.pi * state[name].double().sigmoid()
        shared = torch.atan2(phase.sin().mean(0, keepdim=True),
                             phase.cos().mean(0, keepdim=True)).remainder(2 * math.pi)
        probability = (shared / (2 * math.pi)).clamp(1e-7, 1 - 1e-7)
        state[name] = torch.logit(probability).to(state[name].dtype)
    shared_model = type(model)(replace(model.cfg, shared_global_phase=True))
    shared_model.to(next(model.parameters()).device)
    shared_model.load_state_dict(state, strict=True)
    return shared_model.eval()


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


def inference_state_name(name):
    """Map private checkpoint names to the public vision/language vocabulary."""
    prefixes = {
        'front.': 'vision_frontend.',
        'frame_router.': 'vision_router.',
        'frame_experts.': 'vision_experts.',
        'frame_global.': 'vision_global.',
        'frame_merger.': 'vision_merger.',
        'video_router.': 'language_router.',
        'video_experts.': 'language_experts.',
        'video_global.': 'language_global.',
    }
    exact = {
        'frame_position': 'vision_frame_position',
        'sequence_position': 'language_sequence_position',
        'prompt_embedding': 'language_prompt_embedding',
        'prompt_ready': 'language_prompt_ready',
        'level_scores': 'grade_scores',
        'quality_ready': 'readout_ready',
    }
    result = exact.get(name, name)
    for old, new in prefixes.items():
        if name.startswith(old):
            result = new + name[len(old):]
            break
    return result.replace('.raw_global_phase', '.raw_phase')


def export_inference_checkpoint(source, destination, config_fields, architecture):
    """Export the selected forward tensors and inference configuration."""
    with tempfile.TemporaryDirectory(prefix='lightgen-inference-export-') as directory:
        intermediate = Path(directory)/'LightGen.pt'
        identity = export_public_checkpoint(source, intermediate)
        payload = torch.load(intermediate, map_location='cpu', weights_only=True)
    source_fingerprint = identity['model_state_sha256']
    omitted = ('target_mean', 'target_std', 'level_boundaries')
    state = {inference_state_name(name): tensor for name, tensor in payload['model'].items()
             if name not in omitted}
    if len(state) != len(payload['model']) - len(omitted):
        raise ValueError('Inference key mapping has a collision or missing buffer')
    fingerprint = tensor_fingerprint(state)
    metadata = {name: payload['metadata'][name]
                for name in ('upstream_model', 'upstream_revision', 'fixed_prompt', 'license')
                if name in payload['metadata']}
    exported = {
        'schema': 2,
        'architecture': architecture,
        'config': {name: payload['config'][name] for name in config_fields},
        'model': state,
        'metadata': metadata,
    }
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(exported, destination)
    restored = torch.load(destination, map_location='cpu', weights_only=True)
    if tensor_fingerprint(restored['model']) != fingerprint:
        raise ValueError('Inference export changed retained tensors')
    for name, tensor in payload['model'].items():
        if name not in omitted and not torch.equal(tensor, restored['model'][inference_state_name(name)]):
            raise ValueError('Inference export changed tensor: ' + name)
    return {
        'input_checkpoint_sha256': identity['input_checkpoint_sha256'],
        'public_checkpoint_sha256': file_sha256(destination),
        'source_model_state_sha256': source_fingerprint,
        'model_state_sha256': fingerprint,
        'retained_model_tensors_bitwise_equal': True,
        'omitted_auxiliary_buffers': list(omitted),
    }
