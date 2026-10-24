"""LSP/SALICON original full frozen-Vision/decoder MAC, no neural forwards.

The original teacher still calls the entire visual module after capturing its
last pre-merger block. All main/deepstack mergers count even though its decoder
uses pre-merger features. There is no Language Transformer or vocabulary head.
"""
import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import subprocess
from native_shape_mac import native_mac,TRANSFORMERS_QWEN3VL_SOURCE_SHA

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
CONFIG=REPO/'LightGenV2/tasks/t08_abo_image_text_retrieval/runs/simulation/a100_title_operations_20261024_01/audit_original/model_config.json'
CONFIG_SHA='9172f55b0b9cce70b7f67b10c58a408ccf3ec15c587e6efd4d5f41631237fded'
WEIGHT_SHA='c73fa9caeddeb3ff831d46c085a7a5708343248ca777e90f2d486964464509c1'
GROUPS={
 'LSP':('t02_keypoint_detection','a100_input_inclusive_20261024_02','0a4569f288f6de424b1b412452fa804a96e58d8f7e8655efaa23f461ab8cc735'),
 'SALICON':('t03_saliency','a100_input_inclusive_20261024_01','689d086b5a390c399df109d07b73185575cd2245998b7bd427f6ccea5b0fe019')}
GRAPH_SOURCES=['experiments/qwen3_vl_embedding_2b_fss1000_vision_optical_saliency/modeling.py',
 'experiments/qwen3_vl_embedding_2b_lsp_pose_optical_moe16/modeling.py',
 'LightGenV2/tasks/t03_saliency/aligned_baseline.py','experiments/vision2_hybrid_dense/modeling.py']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def pose_head(settings,area=196):
    if (settings['pose_head_mode']!='deconv' or settings['vision_hidden_size']!=1024
            or settings['pose_projection_dim']!=128 or settings['pose_decoder_channels']!=[128,128]
            or settings['heatmap_size']!=56):raise ValueError('Not the original1102990-parameter deconv head')
    terms=dict(token_projection=area*1024*128,body_two_3x3=2*area*128*128*9,
        upsample14_to28=14**2*128*128*16,upsample28_to56=28**2*128*128*16,
        final_refine=56**2*128*128*9,joint_predictor=56**2*128*14)
    # Original token LN, projection bias, five conv GN groups and final bias.
    params=2*1024+1024*128+128+2*(128*128*9+256)+2*(128*128*16+256)+128*128*9+256+128*14+14
    if params!=1102990:raise ValueError('Head parameter arithmetic differs')
    return terms


def saliency_head(area=196):
    terms=dict(aligned_input_adapter=area*1024*192,decoder_token_projection=area*192*128,
        decoder_two_residual_blocks=2*area*(128*9+128*128))
    height=14;current=128
    for after in (96,64,32,16):
        height*=2;terms[f'decoder_upsample{height}']=height**2*(current*9+current*after);current=after
    terms.update(final_refine=224**2*(16*9+16*16),classifier=224**2*16)
    return terms


def request_terms(config,task,settings):
    all_terms=native_mac(config,1,[196])
    terms={k:v for k,v in all_terms.items() if not k.startswith('language_')}
    terms.update({'head.'+k:v for k,v in (pose_head(settings) if task=='LSP' else saliency_head()).items()})
    return terms


def derive():
    if sha(CONFIG)!=CONFIG_SHA:raise ValueError('Original frozen-Vision config differs')
    config=read(CONFIG);groups=[]
    for task,(directory,run,head_sha) in GROUPS.items():
        repeats=[];refs=[];signatures=[];source_bindings={}
        for repeat in (1,2,3):
            folder=REPO/'LightGenV2/tasks'/directory/'runs/simulation'/run/f'qwen_repeat{repeat:02d}'
            sealed={r['file']:r['sha256'] for r in read(folder/'files_manifest.json')}
            for name in ('identity.json','samples.csv','summary.json'):
                if sha(folder/name)!=sealed[name]:raise ValueError('Sealed native/head input or cost changed')
            identity=read(folder/'identity.json');summary=read(folder/'summary.json');settings=identity['resolved_settings']
            if (identity['system']!='qwen' or identity['checkpoint_sha256']!=head_sha
                    or identity['native_model_assets']['config.json']!=CONFIG_SHA
                    or identity['native_model_assets']['model.safetensors']!=WEIGHT_SHA
                    or identity['repeat']!=repeat or identity['explicit_warmup_forwards']!=0
                    or summary['status']!='complete' or summary['valid_samples']!=200
                    or settings['vision_depth']!=24 or settings['vision_hidden_size']!=1024):
                raise ValueError('Not the qualified original full-Vision and trained-head graph')
            # Bind every decoder/backbone definition to the actual recorded Git,
            # not to an assumed currently edited class. No imports execute it.
            for name in GRAPH_SOURCES:
                blob=subprocess.check_output(['git','show',identity['source_commit']+':'+name],cwd=REPO)
                if hashlib.sha256(blob).hexdigest()!=sha(REPO/name):raise ValueError('Read original recorded graph before changing formulas')
                source_bindings[name]=hashlib.sha256(blob).hexdigest()
            with (folder/'samples.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
            if len(rows)!=200:raise ValueError('All original200 shapes required')
            total=Counter();records=[]
            for i,row in enumerate(rows):
                if (int(row['index'])!=i or ast.literal_eval(row['pixel_values_shape'])!=[196,1536]
                        or ast.literal_eval(row['image_grid_thw'])!=[[1,14,14]]
                        or ast.literal_eval(row['processed_size'])!=[224,224]):raise ValueError('Original fixed224 native processor shape differs')
                terms=request_terms(config,task,settings);total.update(terms)
                records.append(dict(index=i,sample_id=row['sample_id'],MAC=sum(terms.values()),MAC_terms=terms))
            signatures.append([(r['sample_id'],r['MAC']) for r in records]);repeats.append(dict(repeat=repeat,MAC200=sum(total.values()),MAC_terms200=dict(total),records=records))
            refs.append(dict(run=folder.relative_to(REPO).as_posix(),source_commit=identity['source_commit'],
                identity_sha256=sha(folder/'identity.json'),samples_sha256=sha(folder/'samples.csv'),summary_sha256=sha(folder/'summary.json'),
                seconds200=summary['actual_software_wall_seconds'],joules200=summary['cost']['sample_j']*200))
        if signatures[1:]!=signatures[:1]*2:raise ValueError('Three original shape/order signatures differ')
        op=2*sum(r['MAC200'] for r in repeats);seconds=sum(r['seconds200'] for r in refs);energy=sum(r['joules200'] for r in refs)
        groups.append(dict(task=task,calls=600,MAC_OP600=op,TOP_per_input=op/600/1e12,TOPS=op/1e12/seconds,
            TOPS_per_W=op/1e12/energy,repeat_results=repeats,cost_references=refs,original_graph_sources=source_bindings,
            trained_head_sha256=head_sha,all_native24_vision_blocks_and_four_mergers=True,language_or_vocabulary_projection_executed=False))
    return dict(status='ORIGINAL_FULL_VISION_AND_HEAD_SAVED_SHAPES_DENSE_MAC_DERIVED',groups=groups,
        config_sha256=CONFIG_SHA,weight_sha256=WEIGHT_SHA,transformers_runtime_source_sha256=TRANSFORMERS_QWEN3VL_SOURCE_SHA,
        GPU_forwards=0,new_neural_forwards=0,new_timing_or_power_used=False,
        convention='1 real dense MAC=2OP; dense padded convolution convention; transposed-conv input-area*kernel-area*Cin*Cout (not output-area); no hardware-instruction claim',
        non_MAC_not_in_subtotal=['bias','norm','activation','pooling','position interpolation','RoPE/trig/elementwise','bilinear interpolation','sigmoid/normalization','heatmap argmax/coordinate decode','layout'],
        complete_scalar_OP_claim=False,derivation_source_sha256=sha(__file__),
        derivation_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve previous derivation')
    result=derive()
    with a.output.open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
    print(json.dumps([{k:g[k] for k in ('task','TOP_per_input','TOPS','TOPS_per_W')} for g in result['groups']]))
