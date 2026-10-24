"""Original image-query graph MAC from sealed 200x3 inputs; zero forwards."""
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

HERE=Path(__file__).resolve().parent; REPO=HERE.parents[3]
sys.path.insert(0,str(HERE))
from native_shape_mac import native_mac,TRANSFORMERS_QWEN3VL_SOURCE_SHA
from measure_abo_image_qwen import MODEL_CONFIG_SHA,MODEL_WEIGHTS_SHA,ORIGINAL_SHA,FEATURE_PATH,FEATURE_SHA,sha


def request_terms(config, row):
    """Actual dynamic image grid, whole input sequence, original 100x2048 dot."""
    values={k:ast.literal_eval(row[k]) for k in ('input_ids','attention_mask','input_shapes','image_grid_thw')}
    ids,mask,shapes,grid=(values[k] for k in ('input_ids','attention_mask','input_shapes','image_grid_thw'))
    if (len(ids)!=1 or len(mask)!=1 or len(ids[0])!=len(mask[0])
            or not mask[0] or set(mask[0])!={1} or len(grid)!=1 or len(grid[0])!=3
            or shapes['input_ids']!=[1,len(ids[0])] or shapes['attention_mask']!=shapes['input_ids']
            or shapes['image_grid_thw']!=[1,3] or len(ids[0])!=int(row['valid_input_tokens'])):
        raise ValueError('Exact unpadded original B1 sequence/grid required')
    t,h,w=grid[0]
    if min(t,h,w)<=0 or t!=1:
        raise ValueError('Original single-image request only')
    vision=config['vision_config']
    flattened=vision['in_channels']*vision['temporal_patch_size']*vision['patch_size']**2
    if shapes['pixel_values']!=[t*h*w,flattened]:
        raise ValueError('Recorded patch shape disagrees with original config/grid')
    terms=native_mac(config,len(ids[0]),[h*w]*t)
    # forward_base_hidden calls the base .model, NOT model.lm_head. Original
    # MRL descriptor is a slice/normalization, then GPU FP32 cosine dot100.
    terms['original_FP32_100_title_search']=100*config['text_config']['hidden_size']
    return terms, dict(sequence_length=len(ids[0]),grid=grid[0],premerger_tokens=t*h*w)


def derive(config_path):
    config_path=Path(config_path)
    if sha(config_path)!=MODEL_CONFIG_SHA:
        raise ValueError('Wrong original Embedding2B config')
    config=json.loads(config_path.read_text())
    task=REPO/'LightGenV2/tasks/t08_abo_image_text_retrieval/runs/simulation'
    folders=[task/'a100_image_input_inclusive_20261024_02'/f'qwen_repeat{n:02d}' for n in (1,2)]
    folders.append(task/'a100_image_input_inclusive_20261024_03/qwen_repeat03')
    results=[];references=[];signatures=[];checked_sources=set()
    for repeat,folder in enumerate(folders,1):
        sealed={x['file']:x['sha256'] for x in json.loads((folder/'files_manifest.json').read_text())}
        for name in ('identity.json','calls.csv','report.json','retrieval_outputs.npz'):
            if sha(folder/name)!=sealed[name]:raise ValueError('Sealed raw shape/output/cost changed')
        report=json.loads((folder/'report.json').read_text());identity=json.loads((folder/'identity.json').read_text())
        if (report['status']!='complete' or report['measured_calls']!=200
                or report['model_config_sha256']!=MODEL_CONFIG_SHA or report['model_weights_sha256']!=MODEL_WEIGHTS_SHA
                or report['original_source_sha256']!=ORIGINAL_SHA or report['descriptor_dimensions']!=2048
                or report['ranking_stable'] or report['score_device']!='cuda' or report['score_dtype']!='FP32'
                or report['repeat']!=repeat or report['sources']!=identity['sources']
                or report['sources'][FEATURE_PATH]!=FEATURE_SHA):
            raise ValueError('Not the original qualified native image-query graph')
        for name,digest in report['sources'].items():
            key=(report['source_commit'],name,digest)
            if key not in checked_sources:
                blob=subprocess.check_output(['git','show',report['source_commit']+':'+name],cwd=REPO)
                if hashlib.sha256(blob).hexdigest()!=digest:raise ValueError('Original graph not in recorded Git')
                checked_sources.add(key)
        with (folder/'calls.csv').open(encoding='utf-8-sig',newline='') as stream:
            calls=list(csv.DictReader(stream))
        if len(calls)!=200:raise ValueError('All original 200 records required')
        total=Counter();records=[]
        for i,row in enumerate(calls):
            if int(row['index'])!=i:raise ValueError('Original input order changed')
            terms,shape=request_terms(config,row);total.update(terms)
            records.append(dict(index=i,sample_id=row['sample_id'],**shape,MAC=sum(terms.values()),terms=terms))
        signatures.append([(x['sample_id'],x['MAC']) for x in records])
        results.append(dict(repeat=repeat,MAC200=sum(total.values()),MAC_terms200=dict(total),records=records))
        references.append(dict(run=folder.relative_to(REPO).as_posix(),source_commit=report['source_commit'],
            report_sha256=sha(folder/'report.json'),calls_sha256=sha(folder/'calls.csv'),
            seconds200=report['continuous_wall_s'],joules200=report['total_system_energy_j']))
    if signatures[1:]!=signatures[:1]*2:raise ValueError('Three trials have different input shapes')
    op=2*sum(x['MAC200'] for x in results)
    return dict(status='ORIGINAL_NETWORK_AND_SEALED_SHAPES_DENSE_MAC_DERIVED',task='02 image-to-title Qwen',
        calls=600,MAC_OP600=op,TOP_per_input=op/600/1e12,
        TOPS=op/1e12/sum(x['seconds200'] for x in references),
        TOPS_per_W=op/1e12/sum(x['joules200'] for x in references),
        GPU_forwards=0,new_neural_forwards=0,new_timing_or_power_used=False,
        repeat_results=results,cost_references=references,model_config_sha256=MODEL_CONFIG_SHA,
        model_weights_sha256=MODEL_WEIGHTS_SHA,original_features_sha256=FEATURE_SHA,
        transformers_qwen3vl_runtime_source_sha256=TRANSFORMERS_QWEN3VL_SOURCE_SHA,
        model_lm_head_executed=False,offline_gallery_inference_in_query_OP=False,
        convention='1 real dense MAC=2OP; original complete native vision/language and all 4 mergers; causal masking not sparse compute',
        non_MAC_not_in_dense_subtotal=['bias','norm','activation','softmax','embedding_lookup','position_interpolation',
            'vision_rotary_outer_and_trig','rotary_elementwise','MRL_slice_and_normalization','argsort','CPU_return','layout'],
        complete_scalar_OP_claim=False,hardware_peak_or_optical_capacity_claim=False,
        derivation_source_sha256=sha(__file__),
        derivation_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Preserve prior formula receipt')
    result=derive(args.config)
    with args.output.open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:result[k] for k in ('status','calls','TOP_per_input','TOPS','TOPS_per_W','GPU_forwards')}))


if __name__=='__main__':main()
