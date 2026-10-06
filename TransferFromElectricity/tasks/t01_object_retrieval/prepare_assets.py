"""Prepare owned Imagenette and CLIP text artifacts without modifying caches."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def imagenette(archive, destination):
    if (destination / 'imagenette2-160').exists():
        raise FileExistsError('Existing extracted dataset is preserved')
    with tarfile.open(archive) as handle:
        for member in handle.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination.resolve()) or member.issym() or member.islnk():
                raise ValueError('Unsafe archive member')
        handle.extractall(destination, filter='data')
    files = list((destination / 'imagenette2-160').glob('*/*/*.JPEG'))
    if len(files) != 13394:
        raise ValueError(f'Unexpected Imagenette2 image count: {len(files)}')
    (destination / 'asset_manifest.json').write_text(json.dumps({
        'source': 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz',
        'archive_sha256': digest(archive), 'image_count': len(files)}, indent=2))


def clip_text(source, destination):
    import clip
    import clip.clip as original
    import torch
    from transformers import CLIPTextConfig, CLIPTextModel
    torch.set_num_threads(4)
    expected = original._MODELS['ViT-B/16'].split('/')[-2]
    checksum = digest(source)
    if checksum != expected:
        raise ValueError('Cached CLIP weight checksum differs from OpenAI release')
    if destination.exists():
        raise FileExistsError(destination)
    full, _ = clip.load(str(source), device='cpu', jit=False)
    full = full.float().eval()
    state = full.state_dict()
    model = CLIPTextModel(CLIPTextConfig(hidden_size=512, intermediate_size=2048,
        num_hidden_layers=12, num_attention_heads=8, hidden_act='quick_gelu',
        vocab_size=49408, max_position_embeddings=77, bos_token_id=49406,
        eos_token_id=49407, pad_token_id=0)).eval()
    converted = {
        'text_model.embeddings.token_embedding.weight': state['token_embedding.weight'],
        'text_model.embeddings.position_embedding.weight': state['positional_embedding'],
        'text_model.final_layer_norm.weight': state['ln_final.weight'],
        'text_model.final_layer_norm.bias': state['ln_final.bias']}
    for i in range(12):
        src = f'transformer.resblocks.{i}.'
        dst = f'text_model.encoder.layers.{i}.'
        for suffix in ('weight', 'bias'):
            for name, value in zip(('q_proj', 'k_proj', 'v_proj'), state[src + 'attn.in_proj_' + suffix].chunk(3)):
                converted[dst + f'self_attn.{name}.{suffix}'] = value
            for old, new in [('attn.out_proj', 'self_attn.out_proj'), ('ln_1', 'layer_norm1'),
                             ('ln_2', 'layer_norm2'), ('mlp.c_fc', 'mlp.fc1'), ('mlp.c_proj', 'mlp.fc2')]:
                converted[dst + new + '.' + suffix] = state[src + old + '.' + suffix]
    model.load_state_dict(converted, strict=True)
    ids = clip.tokenize(['a photo of a dog', 'an optical phase mask', 'Design a fixed expert bank.'])
    with torch.no_grad():
        reference = full.encode_text(ids)
        actual = model(input_ids=ids, attention_mask=(ids != 0).long()).pooler_output @ state['text_projection']
    error = float((reference - actual).abs().max())
    if not torch.allclose(reference, actual, atol=1e-4, rtol=1e-4):
        raise RuntimeError(f'CLIP conversion mismatch: {error}')
    model.save_pretrained(destination, safe_serialization=True)
    (destination / 'asset_manifest.json').write_text(json.dumps({
        'source': str(source), 'source_url': original._MODELS['ViT-B/16'],
        'source_sha256': checksum, 'conversion_max_error': error,
        'weights_sha256': digest(destination / 'model.safetensors')}, indent=2))
    print('CLIP text conversion verified', error, flush=True)


def clip_vision(source,destination):
    import clip
    import clip.clip as original
    import torch
    from transformers import CLIPVisionConfig, CLIPVisionModel
    torch.set_num_threads(4)
    checksum=digest(source)
    if checksum!=original._MODELS['ViT-B/16'].split('/')[-2]:raise ValueError('CLIP source SHA256 mismatch')
    if destination.exists():raise FileExistsError(destination)
    full,_=clip.load(str(source),device='cpu',jit=False);full=full.float().eval()
    visual=full.visual;state=visual.state_dict()
    model=CLIPVisionModel(CLIPVisionConfig(hidden_size=768,intermediate_size=3072,num_hidden_layers=12,
        num_attention_heads=12,image_size=224,patch_size=16,hidden_act='quick_gelu')).eval()
    converted={'vision_model.embeddings.class_embedding':state['class_embedding'],
        'vision_model.embeddings.patch_embedding.weight':state['conv1.weight'],
        'vision_model.embeddings.position_embedding.weight':state['positional_embedding']}
    for suffix in ('weight','bias'):
        converted['vision_model.pre_layrnorm.'+suffix]=state['ln_pre.'+suffix]
        converted['vision_model.post_layernorm.'+suffix]=state['ln_post.'+suffix]
        for i in range(12):
            src=f'transformer.resblocks.{i}.';dst=f'vision_model.encoder.layers.{i}.'
            for name,value in zip(('q_proj','k_proj','v_proj'),state[src+'attn.in_proj_'+suffix].chunk(3)):
                converted[dst+'self_attn.'+name+'.'+suffix]=value
            for old,new in [('attn.out_proj','self_attn.out_proj'),('ln_1','layer_norm1'),('ln_2','layer_norm2'),
                            ('mlp.c_fc','mlp.fc1'),('mlp.c_proj','mlp.fc2')]:
                converted[dst+new+'.'+suffix]=state[src+old+'.'+suffix]
    model.load_state_dict(converted,strict=True)
    torch.manual_seed(42);pixels=torch.randn(2,3,224,224)
    with torch.no_grad():
        x=visual.conv1(pixels).flatten(2).permute(0,2,1)
        x=torch.cat((visual.class_embedding[None,None].expand(2,1,-1),x),dim=1)
        x=visual.ln_pre(x+visual.positional_embedding)
        reference=visual.transformer(x.permute(1,0,2)).permute(1,0,2)
        actual=model(pixel_values=pixels).last_hidden_state
        error=float((reference-actual).abs().max())
        torch.testing.assert_close(reference,actual,atol=1e-4,rtol=1e-4)
    model.save_pretrained(destination,safe_serialization=True)
    (destination/'asset_manifest.json').write_text(json.dumps({'source':str(source),'source_url':original._MODELS['ViT-B/16'],
        'source_sha256':checksum,'patch_token_conversion_max_error':error,'weights_sha256':digest(destination/'model.safetensors')},indent=2))
    print('CLIP vision patch-token conversion verified',error,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--kind', choices=['imagenette', 'clip', 'clip_vision'], required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    {'imagenette':imagenette,'clip':clip_text,'clip_vision':clip_vision}[args.kind](args.source,args.destination)
