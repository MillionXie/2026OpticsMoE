"""One-time CPU conversion: selective safetensors reads, never AutoModel.

Runtime needs neither the original checkpoint nor the original Qwen directory.
This converter depends only on this package and installed Python libraries.
"""
import argparse
import json
from pathlib import Path
import torch
from PIL import Image
from safetensors import safe_open
from .model import OpticalRetrieval
from .io import inputs, sha256, write_json
from .data import INSTRUCTION

BEST_SHA = '3674981c3499555077c7eadc3072a11e07583675000ec4c012616a96185867f5'


def convert_student(payload):
    state, excluded = {}, []
    for mode,key in [('vision','vision_optical'),('language','language_optical')]:
        for name,value in payload[key].items():
            name = name.removeprefix('core.')
            if mode=='language' and (name.startswith('output_adapter.') or name=='residual_logit'):
                excluded.append(f'{key}.{name}')  # Ignored downstream language hidden reconstruction only.
                continue
            name = name.replace('optical_branch.core.input_adapter.','optics.input_adapter.')
            name = name.replace('optical_branch.core.input_norm.','optics.input_norm.')
            name = name.replace('optical_branch.core.router.','optics.router.')
            name = name.replace('optical_branch.core.expert_layers.0.experts.','optics.experts.')
            if name.startswith('optics.experts.'):
                name = name.removesuffix('.raw_phase')
            name = name.replace('optical_branch.core.global_phase.phase.raw_phase','optics.global_phase')
            name = name.replace('optical_branch.core.output_adapter.','optics.global_output.')
            name = name.replace('optical_branch.expert_output_adapter.','optics.expert_output.')
            state[mode+'.'+name] = value.detach().cpu()
    state.update({'readout.'+k:v.detach().cpu() for k,v in payload['retrieval_readout'].items()})
    return state, excluded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qwen',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--teacher-cache',type=Path,help='Optional old frozen features.pt; export only TRAIN vectors')
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Export requires an empty directory')
    if sha256(args.checkpoint)!=BEST_SHA:
        raise ValueError('Expected audited 70% best checkpoint SHA')
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(str(args.qwen),local_files_only=True,min_pixels=50176,max_pixels=50176)
    batch = inputs(processor,[Image.new('RGB',(224,224))],'cpu')
    ids = batch['input_ids'][0]
    token_ids = ids.unique(sorted=True)
    qconfig = json.loads((args.qwen/'config.json').read_text())
    meta = dict(schema=1,architecture='t07_standalone_six_capture_v1',input_rms=.5,
                token_count=len(token_ids),template_ids=ids.tolist(),prompt=INSTRUCTION,
                image_token_id=qconfig['image_token_id'],source_checkpoint_sha256=BEST_SHA,
                fixed_image_size=224,frontend_dtype='bfloat16',source_epoch=10)
    # Names below match HF Qwen3VL checkpoint.
    wanted = {'visual.patch_embed.proj.weight':'patch.weight','visual.patch_embed.proj.bias':'patch.bias',
              'visual.pos_embed.weight':'position.weight','visual.merger.norm.weight':'merger_norm.weight',
              'visual.merger.norm.bias':'merger_norm.bias','visual.merger.linear_fc1.weight':'merger_fc1.weight',
              'visual.merger.linear_fc1.bias':'merger_fc1.bias','visual.merger.linear_fc2.weight':'merger_fc2.weight',
              'visual.merger.linear_fc2.bias':'merger_fc2.bias'}
    front, provenance = {}, {}
    for shard in sorted(args.qwen.glob('*.safetensors')):
        with safe_open(shard,framework='pt',device='cpu') as handle:
            for key in handle.keys():
                match = next((short for short in wanted if key.endswith(short)),None)
                if match:
                    front['frontend.'+wanted[match]] = handle.get_tensor(key).to(torch.bfloat16)
                    provenance[key] = {'shard':shard.name,'selection':'full tensor'}
                elif key.endswith('embed_tokens.weight'):
                    tensor = handle.get_slice(key)
                    front['frontend.tokens.weight'] = torch.cat([tensor[int(i):int(i)+1] for i in token_ids]).to(torch.bfloat16)
                    provenance[key] = {'shard':shard.name,'rows':token_ids.tolist()}
    if len(front)!=len(wanted)+1:
        raise RuntimeError(f'Unexpected frontend tensor count {len(front)}')
    front['frontend.token_ids'] = token_ids
    original = torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    state, excluded = convert_student(original)
    state.update(front)
    model = OpticalRetrieval(meta)
    model.load_state_dict(state,strict=True)
    args.output.mkdir(parents=True,exist_ok=True)
    processor.save_pretrained(args.output/'processor')
    torch.save({'metadata':meta,'state_dict':model.state_dict()},args.output/'best.pt')
    write_json(args.output/'architecture.json',model.audit())
    write_json(args.output/'conversion.json',dict(source_checkpoint=str(args.checkpoint),source_sha256=BEST_SHA,
               selected_qwen_tensors=provenance,discarded_checkpoint_tensors=excluded,
               optimizer_discarded=True,full_transformer_loaded=False,
               note='Language output projection/residual gate only reconstruct discarded final hidden states.'))
    if args.teacher_cache:
        cache=torch.load(args.teacher_cache,map_location='cpu',weights_only=True)
        if cache['identity']['prompt']!=INSTRUCTION or len(cache['identity']['ids'])!=1920:
            raise ValueError('Unexpected teacher cache identity')
        torch.save({'ids':cache['identity']['ids'][:1440],
                    'manifest_sha256':cache['identity']['manifest_sha256'],
                    'vectors':cache['square'][:1440,:64].clone(),
                    'source_sha256':sha256(args.teacher_cache)},args.output/'train_targets.pt')
    write_json(args.output/'manifest.json',{'schema':1,'files':{str(f.relative_to(args.output)).replace('\\','/'):sha256(f)
               for f in sorted(args.output.rglob('*')) if f.is_file()}})
    print(json.dumps({'output':str(args.output),'size_bytes':sum(f.stat().st_size for f in args.output.rglob('*') if f.is_file()),'audit':model.audit()},indent=2))


if __name__=='__main__':
    main()
