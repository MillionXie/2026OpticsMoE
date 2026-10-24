"""CPU-only binding of the exact archived LGVQ frontend tensor fingerprints.

Reads ONLY native patch/position/embedding tensors from safetensors. No full
Qwen model or Transformer is instantiated. Reproduces original prompt
tokenization and embedding lookup, comparing to original numerical assets.
Video decoding/patch CNN/quality CNN and performance/cost remain separate.
"""
from __future__ import annotations
import argparse
import hashlib
import inspect
import io
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3];sys.path.insert(0,str(HERE));sys.path.insert(0,str(REPO))
from audit_lgvq_front_metadata import hashed_records

METADATA_PIN='1f89ca0e480be4142069f1e14f43df99f47499a20ea2aaa6e3a13672d168a5a6'
PROMPT_PIN='bdce3439754e84ffbcbec601fbd7392f04b3545191d8d57cca639e9a6a8ddb73'
TEMPORAL_PT='5303b574b200e14bf943af21c60a246720be453b9cf8847cd93c0eaaa243a77c'


def native_key(keys,component,name):
    if component=='vision':
        suffix='visual.'+name
        matches=[k for k in keys if k==suffix or k.endswith('.'+suffix)]
    elif component=='language':matches=[k for k in keys if k.endswith('.embed_tokens.'+name)]
    else:raise ValueError('Unknown frontend component')
    if len(matches)!=1:raise ValueError('Ambiguous/missing original tensor: '+component+'/'+name)
    return matches[0]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('metadata','spatial-prompt','temporal-release','model','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve original binding evidence')
    if sha(a.metadata)!=METADATA_PIN or sha(a.spatial_prompt)!=PROMPT_PIN:
        raise ValueError('Wrong recovered original metadata/prompt')
    original=json.loads(a.metadata.read_text(encoding='utf8'))
    vision=original['records']['qwen_vision']['metadata']
    language=original['records']['qwen_spatial_language']['metadata']
    list(hashed_records(vision));list(hashed_records(language))
    if vision['qwen_front_pair_identity']!=language['qwen_front_pair_identity']:
        raise ValueError('Original vision and language source pair differ')
    import torch
    from torch.nn import functional as F
    from safetensors import safe_open
    from transformers import AutoConfig,AutoTokenizer
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
    from experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54 import cache_qwen_front as front
    torch.set_num_threads(8)
    if torch.cuda.is_initialized():raise ValueError('CPU-only asset binding must not inherit CUDA')
    file_checks=[]
    for item in language['qwen_source_identity']['checkpoint_artifacts']['files']:
        path=a.model/item['path']
        if path.stat().st_size!=item['size_bytes']:raise ValueError('Original model file size differs')
        actual=sha(path) if 'sha256' in item else None
        if actual is not None and actual!=item['sha256']:raise ValueError('Original config/tokenizer identity differs')
        file_checks.append(dict(path=item['path'],bytes=path.stat().st_size,sha256=actual,
            large_weight_file_not_fully_rehashed_in_this_binding=actual is None))
    config=AutoConfig.from_pretrained(str(a.model),local_files_only=True)
    for component,actual in (('qwen_vision_front_fingerprint',front._config_mapping(config.vision_config)),
                              ('qwen_language_embedding_fingerprint',front._config_mapping(config.text_config))):
        expected=(vision if component.startswith('qwen_vision') else language)[component]
        key='vision_config' if component.startswith('qwen_vision') else 'language_config'
        if actual!=expected[key]:raise ValueError('Original component config differs: '+component)
    method_sha=hashlib.sha256(inspect.getsource(Qwen3VLVisionModel.fast_pos_embed_interpolate).encode('utf8')).hexdigest()
    if method_sha!=vision['qwen_vision_front_fingerprint']['position_interpolation_code_sha256']:
        raise ValueError('Position interpolation implementation differs')
    checks=[];matrix=None
    with safe_open(str(a.model/'model.safetensors'),framework='pt',device='cpu') as handle:
        keys=list(handle.keys())
        for component,reference in (('vision',vision['qwen_vision_front_fingerprint']),
                                     ('language',language['qwen_language_embedding_fingerprint'])):
            for descriptor in reference['tensors']:
                key=native_key(keys,component,descriptor['name']);value=handle.get_tensor(key)
                actual=front._tensor_descriptor(descriptor['name'],value)
                if actual!=descriptor:raise ValueError('Original frontend parameter bytes differ: '+key)
                checks.append(dict(component=component,safetensors_key=key,**actual))
                if component=='language':matrix=value
    tokenizer=AutoTokenizer.from_pretrained(str(a.model),local_files_only=True)
    cache=torch.load(a.spatial_prompt,map_location='cpu',weights_only=True)
    prompt_checks=[]
    for target in ('spatial','temporal'):
        rendered=front.render_prompt(tokenizer,target_name=target,prompt=front.TARGET_PROMPTS[target])
        cpu=tokenizer([rendered],padding=True,return_tensors='pt',add_special_tokens=False)
        with torch.inference_mode():tokens=F.embedding(cpu['input_ids'],matrix).to(matrix.dtype).half().contiguous()
        if target=='spatial':
            reference=cache['language_tokens'];mask=cache['attention_mask']
            if rendered!=cache['rendered_chat_template'] or not torch.equal(cpu['input_ids'],cache['input_ids']):
                raise ValueError('Original spatial chat/tokenization differs')
            source=dict(prompt_sha256=PROMPT_PIN,reference_kind='original prompt cache also matches formal .6710 field')
        else:
            with zipfile.ZipFile(a.temporal_release) as z:
                release=json.loads(z.read('release.json'))
                if release['checkpoint_sha256']!=TEMPORAL_PT or release['target']!='temporal':
                    raise ValueError('Wrong .8044 temporal numerical reference')
                field=release['fields'][0];raw=z.read(field['file'])
                if hashlib.sha256(raw).hexdigest()!=field['sha256']:raise ValueError('Original temporal field content differs')
                data=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
                reference=data['language_tokens'];mask=data['language_mask']
            source=dict(release=str(a.temporal_release),field=field['file'],field_sha256=field['sha256'],
                reference_kind='original .8044 formal frozen field, not used as an online timing input')
        if (reference.dtype!=torch.float16 or tuple(reference.shape)!=tuple(tokens.shape)
                or not torch.equal(reference,tokens) or not torch.equal(mask,cpu['attention_mask'].bool())):
            raise ValueError('Original '+target+' prompt embedding/mask differs')
        prompt_checks.append(dict(target=target,prompt=front.TARGET_PROMPTS[target],rendered_chat_template=rendered,
            input_ids=cpu['input_ids'].tolist(),attention_mask=cpu['attention_mask'].tolist(),
            shape=list(tokens.shape),dtype=str(tokens.dtype),bitwise_equal=True,reference=source))
    files=[Path(__file__),HERE/'torch_metadata.py',HERE/'audit_lgvq_front_metadata.py',Path(front.__file__)]
    report=dict(status='FRONT_PARAMETER_AND_LANGUAGE_BINDING_ONLY_NO_VIDEO_NN_TIMING',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        sources={str(f.relative_to(REPO)):sha(f) for f in files},
        archived_metadata_sha256=METADATA_PIN,model_candidate=str(a.model),original_artifact_checks=file_checks,
        vision_position_method_sha256=method_sha,matched_front_tensors=checks,matched_prompt_checks=prompt_checks,
        historical_whole_checkpoint_revision_not_proved=True,used_frontend_parameter_bytes_match_original=True,
        no_full_model_instantiated=True,no_transformer_forward=True,no_video_decode=True,
        no_gpu_inference=True,cuda_initialized=torch.cuda.is_initialized(),created_unix=time.time(),
        remaining='Live original video patch/position/quality/FrameStem numerical contracts and formal200x3 timing remain pending')
    with a.output.open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps(dict(status=report['status'],matched_front_tensors=len(checks),
        spatial_language_bitwise=True,temporal_language_bitwise=True,cuda_initialized=report['cuda_initialized'])))


if __name__=='__main__':main()
