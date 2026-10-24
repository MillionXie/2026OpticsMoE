"""CPU arithmetic from original Qwen graph/config and sealed request shapes.

No model import, model forward, CUDA, new timing or new inference. This is a
declared dense-NN MAC convention, NOT a complete scalar/instruction OP count.
GQA attention uses query heads; all main/deepstack mergers execute and count.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
from math import prod
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
sys.path.insert(0,str(HERE))
from measure_abo_text_qwen import sha,write

TRANSFORMERS_QWEN3VL_SOURCE_SHA='685b10cbc48556343df4e159199c5884d02db9ad227640f0f4fbe75ddf0b2a99'


def native_mac(config,language_length,vision_chunks):
    """One B1 original request, actual chunks = grid_h*grid_w for each time slice.

    cu_seqlens in original Qwen3VL VisionModel repeats each spatial area by
    grid_t; non-Flash SDPA processes each chunk independently, not joint frames.
    """
    if not isinstance(language_length,int) or language_length<1 or not vision_chunks or min(vision_chunks)<1:
        raise ValueError('Actual nonempty B1 length and vision chunks required')
    text=config['text_config'];vision=config['vision_config'];length=language_length;n=sum(vision_chunks)
    h=text['hidden_size'];heads=text['num_attention_heads'];kv=text['num_key_value_heads'];dim=text['head_dim']
    v=vision['hidden_size'];inter=vision['intermediate_size'];merge=vision['spatial_merge_size']**2
    if n%merge or heads%kv:raise ValueError('Actual spatial merger/GQA shape invalid')
    grouped=v*merge;merged=n//merge;count=1+len(vision['deepstack_visual_indexes'])
    return dict(
        patch_embedding=n*vision['in_channels']*vision['temporal_patch_size']*vision['patch_size']**2*v,
        vision_QKV_and_output=vision['depth']*n*4*v*v,
        vision_two_linear_MLP=vision['depth']*n*2*v*inter,
        vision_QK_and_AV=vision['depth']*2*v*sum(s*s for s in vision_chunks),
        actual_main_and_deepstack_mergers=count*merged*(grouped*grouped+grouped*vision['out_hidden_size']),
        language_QKVO=text['num_hidden_layers']*length*(h*(heads+2*kv)*dim+heads*dim*h),
        language_SwiGLU=text['num_hidden_layers']*length*3*h*text['intermediate_size'],
        language_QK_and_AV=text['num_hidden_layers']*2*heads*dim*length*length,
        language_rotary_position_matrix=3*(dim//2)*length)


def openmoji_standard_head_mac(instruction_length):
    """Actual original standard SharedGridReadout, source-backed dimensions.

    PositionReadout performs full padded64 einsum, not only valid instruction.
    Native language transformer still uses the entire observed chat sequence.
    """
    if not 0<instruction_length<=64:raise ValueError('Original untruncated instruction span required')
    w=192;area=14*14
    return dict(image_adapter=area*1024*w,text_adapter=instruction_length*2048*w,
        full_padded64_position_linear=64*w,language_pool=w*w,post_film=w*2*w,
        coordinate_projection=area*2*w,
        two_conditioned_conv_groups=2*(w*2*w+area*w*9+area*w*w),
        decoder_pre_depthwise_pointwise=area*w*9+area*w*w,
        category_edit_grid_heads=6*6*w*(17+1),task_head=w*4)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--openmoji-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();root=args.openmoji_root.resolve()
    if args.output.exists():raise FileExistsError('Preserve prior formula receipts')
    references=[];signatures=[];by_repeat=[]
    for repeat in (1,2,3):
        folder=root/f'qwen_repeat{repeat:02d}';identity=json.loads((folder/'identity.json').read_text())
        summary=json.loads((folder/'summary.json').read_text());config_path=folder/'original_native_config.json';config=json.loads(config_path.read_text())
        sealed={r['file']:r['sha256'] for r in json.loads((folder/'files_manifest.json').read_text())}
        for name in ('original_native_config.json','calls.csv','identity.json','summary.json'):
            if sha(folder/name)!=sealed[name]:raise ValueError('Sealed shape/config/cost identity changed')
        if (identity['native_weight_sha256']!='7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0'
                or identity['selected_body_sha256']!='0be775796d5f64df3fab6359eb354ec6666dfc29329484a620b619a6f7ec7cf8'
                or identity['original_config_sha256']!='520930847865400eabf39c9420cf131bf787ec18cf41a92c73973923654034c4'):
            raise ValueError('Not the adopted original native model and standard-head resolved configuration')
        with (folder/'calls.csv').open(encoding='utf-8-sig',newline='') as stream:calls=list(csv.DictReader(stream))
        if len(calls)!=200:raise ValueError('Existing actual200 input shape record required')
        subtotal=Counter();records=[]
        for i,row in enumerate(calls):
            inputs=json.loads(row['native_inputs_json']);length=sum(inputs['attention_mask'][0]);n=inputs['shapes']['pixel_values'][0]
            counts=json.loads(row['native_layer_counts_json']);lang=json.loads(row['language_feature_shape_json'])
            if (int(row['index'])!=i or counts!={'vision':24,'language':28} or n!=196
                    or inputs['shapes']['image_grid_thw']!=[1,3] or len(inputs['input_ids'][0])!=length
                    or json.loads(row['visual_feature_shape_json'])!=[196,1024] or lang[1]!=2048):
                raise ValueError('Not the exact recorded B1 fixed224 original native workload')
            terms=native_mac(config,length,[n]);head=openmoji_standard_head_mac(lang[0])
            terms.update({'head.'+k:v for k,v in head.items()});subtotal.update(terms)
            records.append(dict(index=i,sample_id=row['sample_id'],native_sequence_length=length,
                actual_premerger_tokens=n,actual_instruction_span=lang[0],MAC=sum(terms.values()),terms=terms))
        by_repeat.append(dict(repeat=repeat,MAC200=sum(subtotal.values()),MAC_terms200=dict(subtotal),records=records))
        signatures.append([(r['sample_id'],r['MAC']) for r in records])
        references.append(dict(run=folder.relative_to(REPO).as_posix(),source_commit=identity['source_commit'],
            config_sha256=sha(config_path),calls_sha256=sha(folder/'calls.csv'),summary_sha256=sha(folder/'summary.json'),
            seconds200=summary['continuous_wall_s'],joules200=summary['system_energy_j']))
    if signatures[1:]!=signatures[:1]*2:raise ValueError('Cost trials do not share actual network input shapes')
    # Original head source/version identity was recorded by each cost run.
    bound_sources={}
    for name,digest in identity['sources'].items():
        if name in ('LightGenV2/tasks/t04_semantic_interaction/qwen_shared.py',
                    'LightGenV2/tasks/t04_semantic_interaction/shared_readout.py',
                    'LightGenV2/tasks/t04_semantic_interaction/embedding_model.py',
                    'experiments/qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing/modeling.py',
                    'experiments/qwen3_vl_2b_synthetic_instruction_four_stage_optical_editing/modeling.py'):
            blob=subprocess.check_output(['git','show',identity['source_commit']+':'+name],cwd=REPO)
            if hashlib.sha256(blob).hexdigest()!=digest:raise ValueError('Recorded original head source differs')
            bound_sources[name]=digest
    if len(bound_sources)!=5:raise ValueError('Complete original source-backed head formulas missing')
    op=2*sum(r['MAC200'] for r in by_repeat);seconds=sum(r['seconds200'] for r in references);energy=sum(r['joules200'] for r in references)
    value=dict(status='ORIGINAL_NETWORK_AND_SEALED_SHAPES_DENSE_MAC_DERIVED',task='07 OpenMoji Qwen',
        calls=600,MAC_OP600=op,TOP_per_input=op/600/1e12,TOPS=op/1e12/seconds,TOPS_per_W=op/1e12/energy,
        GPU_forwards=0,new_neural_forwards=0,new_timing_or_power_used=False,repeat_results=by_repeat,cost_references=references,
        graph_sources=bound_sources,transformers_qwen3vl_runtime_source_sha256=TRANSFORMERS_QWEN3VL_SOURCE_SHA,
        shared_head_variant='standard, width192, padded position64, two conditional conv groups, grid6',
        convention='1 real dense MAC=2OP; causal mask not a sparsity discount; all main+3 deepstack mergers included',
        non_MAC_not_in_dense_subtotal=['bias','norm','activation','softmax','embedding_lookup','position_interpolation',
            'vision_rotary_outer_and_trig','rotary_elementwise','pooling','FiLM_elementwise','layout','grid_composition'],
        complete_scalar_OP_claim=False,hardware_peak_or_optical_capacity_claim=False,
        derivation_source_sha256=sha(Path(__file__)),derivation_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
    with args.output.open('x',encoding='utf8') as f:json.dump(value,f,indent=2)
    print(json.dumps({k:value[k] for k in ('status','calls','TOP_per_input','TOPS','TOPS_per_W','GPU_forwards')}))


if __name__=='__main__':main()
