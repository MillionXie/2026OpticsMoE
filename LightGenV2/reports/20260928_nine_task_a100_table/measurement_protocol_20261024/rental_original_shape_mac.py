"""DS/Kimi dense MAC from original public source hashes and sealed input shapes.

No model imports, neural/GPU calls, rental connection or new measurements.
Public source is parsed, never executed. It must match the ORIGINAL runtime SHA.
DS eager MLA uses absorbed compressed attention; Kimi FA2 pads V to Q/K width.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
TASK=REPO/'LightGenV2/tasks/t06_video_quality_assessment'
ROOT=TASK/'runs/simulation/lgvq_mps_e2e_a100_20261024/rental_80gb_20261025_01'
METADATA=TASK/'runs/simulation/a100_rental_original_metadata_20261025_01'
HEADS={'deepseek':'93e6f0de60192b0edf4e13de4af29697eb1f3ad6144567807ac0c2d3e2e395a0',
       'kimi':'e6bf55b3b7ff318d37c973b67c4a6ce1062e4760ac91c2bdc748628988022586'}
SOURCE={
 'kimi':('https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct/resolve/398eede0903cd983a2bfa0cc634e9ac1d843f375/modeling_kimi_vl.py','74e8f18afa5bf8f580ef5be5f15b51506f1e4f604df5a84f508a7796b8947c0b'),
 'DSmodel':('https://raw.githubusercontent.com/deepseek-ai/DeepSeek-VL2/main/deepseek_vl2/models/modeling_deepseek.py','9c7309b9ba7e8c2548ec9e11d6f8fc23256103ed0b58925ea9429e014995578f'),
 'DSconfig':('https://raw.githubusercontent.com/deepseek-ai/DeepSeek-VL2/main/deepseek_vl2/models/configuration_deepseek.py','6ab21f29a4722e26fa28c8e0d4277591689a598df17cf6c712330e8f62b3fc7c'),
 'DSvl':('https://raw.githubusercontent.com/deepseek-ai/DeepSeek-VL2/main/deepseek_vl2/models/modeling_deepseek_vl_v2.py','ffb3d419ef869e77587402a890807efd8c4c719a773e630b0234fe662e2d9610'),
 'DSvision':('https://raw.githubusercontent.com/deepseek-ai/DeepSeek-VL2/main/deepseek_vl2/models/siglip_vit.py','4512ee84bb47d0648e3cefd126dc46f5d0bd038729e478e2331184b5de64e537')}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def class_node(tree,name):return next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name)
def method(tree,owner,name):return next(n for n in class_node(tree,owner).body if isinstance(n,ast.FunctionDef) and n.name==name)


def defaults(tree,name):
    node=class_node(tree,name);result={}
    for n in node.body:
        if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name) and n.value is not None:
            try:result[n.target.id]=ast.literal_eval(n.value)
            except (ValueError,TypeError):pass
    init=method(tree,name,'__init__');args=init.args.args
    for arg,value in zip(args[-len(init.args.defaults):],init.args.defaults):
        try:result[arg.arg]=ast.literal_eval(value)
        except (ValueError,TypeError):pass
    return result


def fetch_source_facts():
    trees={};receipts={}
    for key,(url,pin) in SOURCE.items():
        value=urllib.request.urlopen(url,timeout=30).read()
        if hashlib.sha256(value).hexdigest()!=pin:raise ValueError('Public source differs from sealed original runtime: '+key)
        trees[key]=ast.parse(value);receipts[key]=dict(url=url,sha256=pin,bytes=len(value),parsed_only_never_imported=True)
    ds=ast.unparse(method(trees['DSmodel'],'DeepseekV2Attention','forward'))
    kimi=ast.unparse(method(trees['kimi'],'DeepseekV3FlashAttention2','forward'))
    if not all(v in ds for v in ('q_absorb','out_absorb','compressed_kv.unsqueeze(-3).mT',"torch.einsum('bhql,blc->bhqc'")):
        raise ValueError('Original absorbed MLA arithmetic not present')
    if 'value_states = F.pad(value_states, [0, self.q_head_dim - self.v_head_dim])' not in kimi:
        raise ValueError('Original FA2 V-padding graph changed')
    return dict(public_sources=receipts,DS_language_defaults=defaults(trees['DSconfig'],'DeepseekV2Config'),
        DS_vision_defaults=defaults(trees['DSvl'],'VisionEncoderConfig'),DS_projector_defaults=defaults(trees['DSvl'],'MlpProjectorConfig'),
        absorbed_MLA_forward_AST_sha256=hashlib.sha256(ds.encode()).hexdigest(),
        FA2_V_padding_forward_AST_sha256=hashlib.sha256(kimi.encode()).hexdigest(),
        no_source_module_import_or_execution=True)


def language_mac(c,length,absorbed):
    h=c['hidden_size'];heads=c['num_attention_heads'];r=c['kv_lora_rank'];nope=c['qk_nope_head_dim'];rope=c['qk_rope_head_dim'];v=c['v_head_dim'];q=nope+rope;layers=c['num_hidden_layers']
    if c['q_lora_rank'] is not None or c.get('ep_size',1)!=1 or length<1:raise ValueError('Exact original no-Q-LoRA single-device B1 graph required')
    terms=dict(language_Q_projection=layers*length*h*heads*q,language_compressed_KV_A=layers*length*h*(r+rope),
        language_output_projection=layers*length*heads*v*h)
    if absorbed:
        terms.update(language_Q_absorb=layers*heads*length*nope*r,
            language_compressed_QK=layers*heads*length*length*(rope+r),
            language_compressed_AV=layers*heads*length*length*r,
            language_output_absorb=layers*heads*length*r*v)
        padding=0
    else:
        terms.update(language_expanded_KV_B=layers*length*r*heads*(nope+v),
            language_QK=layers*heads*length*length*q,language_AV_actual_padded=layers*heads*length*length*q)
        padding=layers*heads*length*length*(q-v)
    moe=sum(i>=c['first_k_dense_replace'] and i%c['moe_layer_freq']==0 for i in range(layers));dense=layers-moe
    terms.update(language_dense_SwiGLU=dense*length*3*h*c['intermediate_size'],
        language_routed_expert_SwiGLU=moe*length*c['num_experts_per_tok']*3*h*c['moe_intermediate_size'],
        language_shared_expert_SwiGLU=moe*length*3*h*c['moe_intermediate_size']*c['n_shared_experts'],
        language_router=moe*length*h*c['n_routed_experts'],language_weighted_expert_sum=moe*length*c['num_experts_per_tok']*h,
        native_full_vocabulary_last_vector=h*c['vocab_size'],trained_five_quality_rows=h*5,quality_score_dot=5)
    return terms,padding


def request_mac(model,config,evidence,facts):
    values=evidence['small_cpu_values'];shapes=evidence['tensors'];ids=values['input_ids'];mask=values['attention_mask']
    length=evidence['effective_input_tokens']
    if (len(ids)!=1 or len(mask)!=1 or len(ids[0])!=length or len(mask[0])!=length or not all(mask[0])
            or shapes['input_ids']['shape']!=[1,length]):raise ValueError('Actual unpadded B1 original sequence required')
    if hashlib.sha256(json.dumps(values,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=evidence['small_values_sha256']:
        raise ValueError('Original processor ID/mask/tiling evidence changed')
    if model=='deepseek':
        c=dict(facts['DS_language_defaults'],**config['language_config']);vision=dict(facts['DS_vision_defaults'],**config['vision_config']);project=dict(facts['DS_projector_defaults'],**config['projector_config'])
        if (shapes['images']['shape']!=[1,8,3,384,384] or values['images_spatial_crop']!=[[[1,1]]*4]
                or vision['class_token'] or not vision['ignore_head'] or project['token_pooling']
                or project['projector_type']!='downsample_mlp_gelu' or project['depth']!=2 or project['mlp_ratio']!=1):raise ValueError('Original eight real tiles/projector graph differs')
        chunks=[(384//vision['patch_size'])**2]*8;n=sum(chunks);width=vision['width'];depth=vision['layers'];inter=int(width*vision['mlp_ratio'])
        per_tile=((384//vision['patch_size']+project['downsample_ratio']-1)//project['downsample_ratio'])**2
        projector=8*per_tile*(project['input_dim']*project['downsample_ratio']**2*project['n_embed']+project['n_embed']**2)
        terms,padding=language_mac(c,length,True)
    elif model=='kimi':
        c=config['text_config'];vision=config['vision_config'];width=vision['hidden_size'];depth=vision['num_hidden_layers'];inter=vision['intermediate_size']
        if shapes['pixel_values']['shape']!=[4096,3,14,14] or shapes['image_grid_hws']['shape']!=[4,2]:raise ValueError('Original four448 grid shapes differ')
        # Original prepare_batch asserts all four grids equal32x32. Values were
        # not duplicated by CPU evidence; this is SOURCE-derived, not a new read.
        chunks=[32*32]*4;n=sum(chunks);merge=vision['merge_kernel_size'][0]*vision['merge_kernel_size'][1];grouped=width*merge
        projector=(n//merge)*(grouped*grouped+grouped*c['hidden_size'])
        terms,padding=language_mac(c,length,False)
    else:raise ValueError('Only the original full DS/Kimi models')
    patch=vision['patch_size']
    terms.update(vision_patch_embedding=n*3*patch*patch*width,
        vision_QKV_output=depth*n*4*width*width,vision_two_linear_MLP=depth*n*2*width*inter,
        vision_QK_AV=depth*2*width*sum(k*k for k in chunks),original_multimodal_projector=projector)
    return dict(sample_id=evidence['sample_id'],actual_language_length=length,vision_attention_chunks=chunks,
        MAC=sum(terms.values()),MAC_terms=terms,FA2_zero_padded_value_dense_MAC=padding,
        logical_unpadded_MAC=sum(terms.values())-padding)


def build():
    facts=fetch_source_facts();groups=[]
    for model in ('deepseek','kimi'):
        config_path=METADATA/model/'config.json';config=read(config_path)
        if sha(METADATA/model/'best_checkpoint.pt')!=HEADS[model]:raise ValueError('Original five-row head metadata changed')
        first_signature=None
        for cap in (100,50):
            repeats=[];refs=[];totals=Counter()
            for repeat in (1,2,3):
                folder=ROOT/f'{model}_cap{cap}_repeat{repeat:02d}';report=read(folder/'report.json');inputs=read(folder/'input_audit.json')
                if ((report['model'],report['cap'],report['repeat'],report['batch'])!=(model,cap,repeat,1)
                        or report['status']!='complete' or report['checkpoint_sha256']!=HEADS[model]
                        or report['assets']['config.json']!=sha(config_path) or report['precision']!='native'
                        or report['gpu_uuid']!='GPU-6d1916d1-fbd3-524b-3d98-7fb38512c388' or report['power_limit_w']!=400
                        or report['explicit_warmup_forwards']!=0 or report['measured_videos']!=200):raise ValueError('Original qualified rental model/point identity differs')
                runtime=report['runtime'];pins=runtime['official_runtime_source_sha256']
                expected={'modeling_kimi_vl.py':SOURCE['kimi'][1]} if model=='kimi' else {name:SOURCE[key][1] for name,key in [('modeling_deepseek.py','DSmodel'),('modeling_deepseek_vl_v2.py','DSvl'),('configuration_deepseek.py','DSconfig'),('siglip_vit.py','DSvision')]}
                if any(pins[k]!=v for k,v in expected.items()):raise ValueError('Source-backed original runtime differs')
                if runtime['actual_attention']!=({'vision_class':'Block','text_class':'DeepseekV2Attention'} if model=='deepseek' else {'vision':'flash_attention_2','text_class':'DeepseekV3FlashAttention2'}):raise ValueError('Wrong actual attention implementation')
                if sha(folder/'input_audit.json')!=report['input_audit_sha256']:raise ValueError('Sealed input evidence changed')
                with (folder/'calls.csv').open(encoding='utf-8-sig',newline='') as f:calls=list(csv.DictReader(f))
                records=[request_mac(model,config,e,facts) for e in inputs['samples']]
                if len(records)!=200 or [r['sample_id'] for r in records]!=[r['sample_id'] for r in calls]:raise ValueError('Original first200 order differs')
                signature=[(r['sample_id'],r['actual_language_length'],r['MAC']) for r in records]
                if first_signature is None:first_signature=signature
                if first_signature!=signature:raise ValueError('Original cap/repeats have unequal workloads')
                bound={}
                for name in ('rental_vlm_batch_followup.py','rental_precision_recheck.py'):
                    path='LightGenV2/tasks/t06_video_quality_assessment/'+name;blob=subprocess.check_output(['git','show',report['git_commit']+':'+path],cwd=REPO)
                    if hashlib.sha256(blob).hexdigest()!=report['sources'][path]:raise ValueError('Original full native/quality graph differs')
                    bound[path]=report['sources'][path]
                for r in records:totals.update(r['MAC_terms'])
                repeats.append(dict(repeat=repeat,MAC200=sum(r['MAC'] for r in records),records=records))
                refs.append(dict(run=folder.relative_to(REPO).as_posix(),report_sha256=sha(folder/'report.json'),
                    input_audit_sha256=sha(folder/'input_audit.json'),calls_sha256=sha(folder/'calls.csv'),graph_sources=bound,
                    seconds200=report['cost']['continuous_wall_s'],joules200=report['cost']['estimated_system_energy_j']))
            op=2*sum(r['MAC200'] for r in repeats);seconds=sum(r['seconds200'] for r in refs);energy=sum(r['joules200'] for r in refs)
            groups.append(dict(model=model,cap=cap,calls=600,MAC_OP600=op,TOP_per_video=op/600/1e12,
                equivalent_TOP_per_input_frame=op/2400/1e12,TOPS=op/1e12/seconds,TOPS_per_W=op/1e12/energy,
                FA2_zero_padding_MAC600=sum(v['FA2_zero_padded_value_dense_MAC'] for r in repeats for v in r['records']),
                MAC_terms600=dict(totals),cost_references=refs,repeat_results=repeats,config_sha256=sha(config_path),head_sha256=HEADS[model]))
    return dict(status='ORIGINAL_RUNTIME_AND_SEALED_INPUT_SHAPES_DENSE_MAC_DERIVED',groups=groups,source_facts=facts,
        GPU_forwards=0,new_neural_forwards=0,new_timing_or_power_used=False,no_rental_connection=True,
        convention='1 real dense MAC=2OP; dense attention without causal sparsity discount; actual Kimi V padding retained, unpadded alternative explicit',
        DS_attention='original eager absorbed MLA, no fictitious expanded KV_B forward',
        Kimi_grids='source-derived from original prepare_batch assert32x32 four grids plus sealed4096 patch/4x2 tensor shapes; not newly observed grid values',
        non_MAC_not_in_subtotal=['embedding_lookup','bias','norm','softmax','activation','RoPE elementwise/trig/cache','position interpolation','pooling','router topk/sort','layout','tile newline concatenation'],
        complete_scalar_OP_or_hardware_instruction_count_claim=False,derivation_source_sha256=sha(__file__),
        derivation_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve prior formula receipt')
    value=build()
    with a.output.open('x',encoding='utf8') as f:json.dump(value,f,indent=2)
    print(json.dumps([{k:r[k] for k in ('model','cap','TOP_per_video','TOPS','TOPS_per_W')} for r in value['groups']]))
