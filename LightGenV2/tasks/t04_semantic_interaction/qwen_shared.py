"""Frozen complete Qwen baseline; exact same final readout, supervision and trainer."""
from pathlib import Path
import json
import torch
from torch import nn
from torch.utils.data import DataLoader
from PIL import Image
from .shared_readout import create_shared_readout, head_signature
from .embedding_data import sha256
from experiments.qwen3_vl_2b_synthetic_instruction_four_stage_optical_editing.modeling import restore_block_major
from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.datasets import OpenMojiEditingDataset, collate_samples


class QwenSharedReadout(nn.Module):
    """Training on frozen exact features equals training only the online readout."""
    router_backend = 'none'
    checkpoint_architecture = 't04_full_frozen_qwen_premerge196_instructiontokens_sharedhead_v1'

    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        if settings.shared_readout_variant != 'standard':
            self.checkpoint_architecture += '_' + settings.shared_readout_variant
        self.vision_stem = nn.Identity()  # No backbone training; exact frozen forwards cached separately.
        self.image_adapter = nn.Sequential(nn.Linear(1024, 192), nn.LayerNorm(192))
        self.text_adapter = nn.Sequential(nn.Linear(2048, 192), nn.LayerNorm(192))
        self.shared_readout = create_shared_readout(settings)
        self.readout_signature = head_signature(self.shared_readout)

    def forward(self, visual, text_groups):
        device = self.image_adapter[0].weight.device
        spatial = restore_block_major(self.image_adapter(visual.to(device, dtype=torch.float32)))
        language = [self.text_adapter(g.to(device, dtype=torch.float32)) for g in text_groups]
        condition = self.shared_readout.summarize(language)
        result = self.shared_readout(spatial, condition)
        result.update(ccd_operating_loss=spatial.new_zeros(()), router_balance_loss=spatial.new_zeros(()))
        return result

    def _optical_paths(self):
        return []

    def set_phase_trainable(self, enabled):
        pass

    def router_importance_loss(self):
        return next(self.parameters()).new_zeros(())

    def router_hard_load_balance_loss(self):
        return next(self.parameters()).new_zeros(())

    def architecture_report(self):
        return {'type': self.checkpoint_architecture, 'qwen_frozen': True, 'lora': False,
                'vision_feature': 'last native vision block, pre-merger, [B,196,1024]',
                'text_feature': 'last native language layer, instruction positions only, [B,L,2048]',
                'full_inference': 'processor -> all native Qwen vision/language layers -> two dimension adapters -> shared readout',
                'training_cache': 'exact frozen forwards; no target labels supplied to Qwen; not a backbone speed measurement',
                'shared_readout': self.readout_signature,
                'trainable_parameters': sum(p.numel() for p in self.parameters()),
                'additional_adapters': 'Linear1024->192+LN and Linear2048->192+LN, no extra mixers or branches'}


def instruction_span(token_ids, instruction_ids):
    matches = [i for i in range(len(token_ids) - len(instruction_ids) + 1)
               if token_ids[i:i + len(instruction_ids)] == instruction_ids]
    if len(matches) != 1:
        raise ValueError(f'Expected one exact instruction token span, found {len(matches)}')
    return slice(matches[0], matches[0] + len(instruction_ids))


def cache_path(settings, split):
    return settings.data_dir / 'qwen_shared_head_v1' / f'{split}_native.pt'


@torch.inference_mode()
def extract_native_one(model, processor, row, settings, device):
    with Image.open(settings.data_dir / row['relative_dir'] / row['files']['source']) as handle:
        image = handle.convert('RGB')
    # Same instruction content as optical method. No extra task description or answer labels.
    messages = [{'role': 'user', 'content': [{'type': 'image', 'image': image},
                                          {'type': 'text', 'text': row['instruction']}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    inputs = processor(text=[text], images=[image], return_tensors='pt')
    ids = processor.tokenizer.encode(row['instruction'], add_special_tokens=False)
    span = instruction_span(inputs['input_ids'][0].tolist(), ids)
    if not 0 < len(ids) <= settings.max_language_tokens:
        raise ValueError('Instruction length exceeds shared readout; no truncation')
    captured = []
    counts = {'vision': 0, 'language': 0}
    def count(name):
        def hook(module, args, output):
            counts[name] += 1
        return hook
    handles = [layer.register_forward_hook(count('vision')) for layer in model.model.visual.blocks]
    handles += [layer.register_forward_hook(count('language')) for layer in model.model.language_model.layers]
    handles.append(model.model.visual.blocks[-1].register_forward_hook(lambda module, args, output: captured.append(output.detach().clone())))
    try:
        output = model.model(**{k: v.to(device) for k, v in inputs.items()}, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    assert counts == {'vision': len(model.model.visual.blocks), 'language': len(model.model.language_model.layers)}
    visual = captured[0]
    language = output.last_hidden_state[0, span]
    if visual.shape != (196, 1024) or language.shape != (len(ids), 2048):
        raise ValueError(f'Unexpected Qwen feature shape: {visual.shape}, {language.shape}')
    return visual.cpu().to(torch.bfloat16), language.cpu().to(torch.bfloat16), counts


def prepare_native_cache(settings, device):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    paths = [cache_path(settings, split) for split in ('train', 'test')]
    contracts = {split: {'type': 'full_native_qwen_shared_head_v1', 'manifest_sha256': sha256(settings.data_dir / (split + '.jsonl')),
                        'checkpoint': str(settings.qwen_checkpoint), 'instruction': 'plain original instruction inside native chat wrapper',
                        'image_size': 224} for split in ('train', 'test')}
    missing = []
    for split, path in zip(('train', 'test'), paths):
        if path.exists():
            meta = torch.load(path, map_location='cpu', weights_only=False, mmap=True)['meta']
            if any(meta.get(k) != v for k, v in contracts[split].items()):
                raise ValueError('Native Qwen cache does not match this data/model contract')
        else:
            missing.append(split)
    if not missing:
        return
    if device.type != 'cuda':
        raise ValueError('Frozen full Qwen feature extraction requires --device cuda in this experiment')
    processor = AutoProcessor.from_pretrained(settings.qwen_checkpoint, local_files_only=True,
                                             min_pixels=224**2, max_pixels=224**2)
    model = Qwen3VLForConditionalGeneration.from_pretrained(settings.qwen_checkpoint, local_files_only=True,
             dtype=torch.bfloat16, attn_implementation='sdpa').to(device).eval().requires_grad_(False)
    for split in missing:
        rows = [json.loads(line) for line in (settings.data_dir / (split + '.jsonl')).read_text().splitlines()]
        visuals, languages = [], []
        for i, row in enumerate(rows):
            visual, language, counts = extract_native_one(model, processor, row, settings, device)
            visuals.append(visual); languages.append(language)
            if i == 0 or (i + 1) % 100 == 0:
                print(f'[Qwen frozen cache] {split} {i+1}/{len(rows)}; native layers {counts}', flush=True)
        path = cache_path(settings, split)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'meta': {**contracts[split], 'executed_layers_per_sample': counts},
                    'sample_ids': [r['sample_id'] for r in rows], 'visual': torch.stack(visuals), 'language': languages}, path)
        path.with_suffix('.json').write_text(json.dumps({**contracts[split], 'sha256': sha256(path),
                                      'executed_layers_per_sample': counts}, indent=2))
    del model
    torch.cuda.empty_cache()


class CachedDataset(OpenMojiEditingDataset):
    def __init__(self, settings, split):
        self.settings = settings
        self.records = [json.loads(line) for line in (settings.data_dir / (split + '.jsonl')).read_text().splitlines()]
        self.cache = torch.load(cache_path(settings, split), map_location='cpu', weights_only=False, mmap=True)
        if [r['sample_id'] for r in self.records] != self.cache['sample_ids']:
            raise ValueError('Cache/manifest sample order mismatch')

    def __getitem__(self, index):
        row = self.records[index]
        source, target = torch.tensor(row['source_grid']), torch.tensor(row['target_grid'])
        return {'sample_id': row['sample_id'], 'task': row['task'], 'instruction': row['instruction'], 'program': row['program'],
                'source_image': self.cache['visual'][index], 'prompt_hidden': self.cache['language'][index],
                'source_grid': source, 'target_grid': target, 'edit_grid': source.ne(target).float(),
                'preserve_grid': source.eq(target).float(), 'task_index': torch.tensor(row['task_index'])}


def build_cached_loaders(settings):
    common = dict(batch_size=settings.batch_size, num_workers=settings.num_workers, pin_memory=torch.cuda.is_available(),
                  persistent_workers=settings.num_workers > 0, collate_fn=collate_samples)
    return (DataLoader(CachedDataset(settings, 'train'), shuffle=True, generator=torch.Generator().manual_seed(settings.seed), **common),
            DataLoader(CachedDataset(settings, 'test'), shuffle=False, **common))
