"""Original Qwen video network MAC from sealed CPU processor records only.

No Torch, model, GPU, new timing, or inference. Includes the original complete
vocabulary projection of the LAST pooled hidden vector, then five quality rows.
This is a declared dense-MAC subtotal, not a complete scalar operation count.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
sys.path.insert(0,str(HERE))
from native_shape_mac import native_mac,sha
TASK=REPO/'LightGenV2/tasks/t06_video_quality_assessment'
ROOT=TASK/'runs/simulation/lgvq_mps_e2e_a100_20261024'
CONFIG_SHA='bec4b3d446efa05807365c9e1cec03ac590836879d02f3a6da879971154bdd3b'
WEIGHT_SHA='7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0'
HEAD_SHA='e102adc6a7b33335c8f49ae1cbeea3302a1a10017d282da8d6091a640276d0d2'
MANIFEST_SHA='607c50d20662a47795c23cd2038081b30368ed108ba9be8a1bb8a9d250f8e7fc'


def quality_head_mac(config):
    text=config['text_config'];h=text['hidden_size']
    return dict(native_full_vocabulary_last_pooled_vector=h*text['vocab_size'],
        trained_five_quality_rows=h*5,quality_probability_score_dot=5)


def request_mac(config,evidence):
    small=evidence['small_cpu_values'];shapes=evidence['tensors']
    ids=small['input_ids'];mask=small['attention_mask'];grid=small['video_grid_thw']
    if (len(ids)!=1 or len(mask)!=1 or len(ids[0])!=len(mask[0])
            or any(v!=1 for v in mask[0]) or len(grid)!=1 or len(grid[0])!=3):
        raise ValueError('Actual unpadded B1 original video IDs/masks/grid required')
    t,h,w=grid[0];n=t*h*w;length=len(ids[0]);vision=config['vision_config']
    patch_width=vision['in_channels']*vision['temporal_patch_size']*vision['patch_size']**2
    if (min(t,h,w)<1 or shapes['pixel_values_videos']['shape']!=[n,patch_width]
            or shapes['input_ids']['shape']!=[1,length] or shapes['attention_mask']['shape']!=[1,length]
            or shapes['video_grid_thw']['shape']!=[1,3] or evidence['effective_input_tokens']!=length):
        raise ValueError('Actual processor tensor shapes/token count differ')
    signature=hashlib.sha256(json.dumps(small,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if signature!=evidence['small_values_sha256']:raise ValueError('Sealed actual CPU metadata changed')
    terms=native_mac(config,length,[h*w]*t);terms.update(quality_head_mac(config))
    return dict(sample_id=evidence['sample_id'],actual_sequence_length=length,actual_grid=grid,
        actual_attention_chunks=[h*w]*t,small_values_sha256=signature,MAC=sum(terms.values()),MAC_terms=terms)


def group_paths():
    return {
        'laboratory40GB_cap100':[ROOT/'qwen100_repaired01/qwen_cap100_repeat01',
            ROOT/'qwen100_remaining01/qwen_cap100_repeat02',ROOT/'qwen100_remaining01/qwen_cap100_repeat03'],
        'laboratory40GB_cap50':[ROOT/'lab_remaining01/qwen50'/f'qwen_cap50_repeat{i:02d}' for i in (1,2,3)],
        'supplementary80GB_cap100':[ROOT/'rental_qwen_anchor_20261025_02'/f'qwen_cap100_repeat{i:02d}' for i in (1,2,3)]}


def build(config_path,selected_groups=None):
    if sha(config_path)!=CONFIG_SHA:raise ValueError('Exact original Instruct config bytes required')
    config=json.loads(config_path.read_text());groups=[];shared_signature=None;available=group_paths()
    if selected_groups is not None and (not selected_groups or len(set(selected_groups))!=len(selected_groups) or not set(selected_groups)<=available.keys()):
        raise ValueError('Explicit existing canonical groups only')
    for label in (list(available) if selected_groups is None else selected_groups):
        folders=available[label]
        refs=[];repeats=[];subtotals=Counter()
        for repeat,folder in enumerate(folders,1):
            report=json.loads((folder/'report.json').read_text());inputs=json.loads((folder/'input_audit.json').read_text())
            cap=50 if label.endswith('cap50') else 100;rental=label.startswith('supplementary')
            if (report['status']!='complete' or report['model']!='qwen' or report['cap']!=cap or report['repeat']!=repeat
                    or report['checkpoint_sha256']!=HEAD_SHA or report['manifest_sha256']!=MANIFEST_SHA
                    or report['assets']['config.json']!=CONFIG_SHA or report['assets']['model.safetensors']!=WEIGHT_SHA
                    or report['measured_videos']!=200 or report['explicit_warmup_forwards']!=0
                    or report['gpu_uuid']!=('GPU-6d1916d1-fbd3-524b-3d98-7fb38512c388' if rental else 'GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55')
                    or report['power_limit_w']!=(400 if rental else 250)):
                raise ValueError('Qualified original model/input/platform/cap identity differs')
            if sha(folder/'input_audit.json')!=report['input_audit_sha256']:raise ValueError('Sealed actual processor evidence changed')
            with (folder/'calls.csv').open(encoding='utf-8-sig',newline='') as f:calls=list(csv.DictReader(f))
            records=[request_mac(config,item) for item in inputs['samples']]
            if len(records)!=200 or [r['sample_id'] for r in records]!=[r['sample_id'] for r in calls]:
                raise ValueError('Actual original first200 order differs')
            signature=[(r['sample_id'],r['small_values_sha256'],r['MAC']) for r in records]
            if shared_signature is None:shared_signature=signature
            if signature!=shared_signature:raise ValueError('Three repeats/platforms have different actual workloads')
            source_pins={}
            for name in ('quality_token_batch_benchmark_200.py','quality_token_common.py','evaluate_temporal_baseline_batches.py'):
                path='LightGenV2/tasks/t06_video_quality_assessment/'+name
                blob=subprocess.check_output(['git','show',report['git_commit']+':'+path],cwd=REPO)
                digest=hashlib.sha256(blob).hexdigest()
                if digest!=report['sources'][path]:raise ValueError('Original complete backbone/vocabulary/quality graph differs')
                source_pins[path]=digest
            mac=sum(r['MAC'] for r in records)
            for record in records:subtotals.update(record['MAC_terms'])
            repeats.append(dict(repeat=repeat,MAC200=mac,records=records))
            refs.append(dict(path=folder.relative_to(REPO).as_posix(),report_sha256=sha(folder/'report.json'),
                calls_sha256=sha(folder/'calls.csv'),input_audit_sha256=sha(folder/'input_audit.json'),
                source_commit=report['git_commit'],graph_sources=source_pins,cap=cap,gpu_uuid=report['gpu_uuid'],
                power_limit_w=report['power_limit_w'],seconds200=report['cost']['continuous_wall_s'],
                joules200=report['cost']['estimated_system_energy_j']))
        ops=2*sum(r['MAC200'] for r in repeats);seconds=sum(r['seconds200'] for r in refs);energy=sum(r['joules200'] for r in refs)
        groups.append(dict(group=label,calls=600,MAC_terms600=dict(subtotals),MAC_OP600=ops,TOP_per_video=ops/600/1e12,
            equivalent_TOP_per_input_frame=ops/2400/1e12,TOPS=ops/1e12/seconds,TOPS_per_W=ops/1e12/energy,
            cost_references=refs,repeat_results=repeats,not_replacing_40GB_anchor=rental))
    return dict(status='ORIGINAL_NETWORK_AND_SEALED_VIDEO_SHAPES_DENSE_MAC_DERIVED',groups=groups,
        config_sha256=CONFIG_SHA,native_weight_sha256=WEIGHT_SHA,quality_head_sha256=HEAD_SHA,
        GPU_forwards=0,new_neural_forwards=0,new_timing_or_power_used=False,
        original_vocabulary_projection='once per pooled last vector, NOT every language token; five quality rows also execute',
        convention='1 real dense MAC=2OP; all24Vision/28Language, all4mergers; query-head QKAV; per-time-slice vision attention',
        non_MAC_not_in_dense_subtotal=['decode','resize','processor','tokenizer','embedding_lookup','norm','bias','activation',
            'softmax','rotary_elementwise','vision_position_interpolation','pooling','layout'],complete_scalar_OP_claim=False,
        derivation_source_sha256=sha(Path(__file__)),derivation_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--groups',nargs='+',choices=tuple(group_paths()),help='Explicit groups present on this CPU; does not infer or drop failed points')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve immutable formula receipt')
    value=build(a.config,a.groups)
    with a.output.open('x',encoding='utf8') as f:json.dump(value,f,indent=2)
    print(json.dumps([dict(group=g['group'],TOP_per_video=g['TOP_per_video'],TOPS=g['TOPS'],TOPS_per_W=g['TOPS_per_W']) for g in value['groups']]))
