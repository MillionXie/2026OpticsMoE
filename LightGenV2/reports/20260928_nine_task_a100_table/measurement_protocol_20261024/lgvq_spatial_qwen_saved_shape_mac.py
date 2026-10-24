"""Original spatial quality-head graph and sealed600 native shapes, no forwards."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import subprocess
from lgvq_qwen_saved_shape_mac import request_mac,CONFIG_SHA,WEIGHT_SHA,MANIFEST_SHA

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
TASK=REPO/'LightGenV2/tasks/t06_video_quality_assessment'
ROOT=TASK/'runs/simulation/lgvq_spatial_input_a100_20261024_01'
CONFIG=TASK/'runs/simulation/a100_qwen_saved_graph_metadata_20261025_01/original_native_config.json'
HEAD_SHA='9155ec7a45a19e14ac87661021b44599317d5efb16475df77707d70e866ad913'
REFERENCE_SHA='9858954de368171e4736607cd8f2340b871191799e10d3068b3576b73cdd20e6'
PROMPT='Please evaluate the spatial quality of this video and rate it using one of the following five levels: Excellent, Good, Fair, Poor, or Bad.'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def validate(report,repeat):
    if (report['status']!='complete' or report['system']!='qwen' or report['task']!='LGVQ spatial'
            or report['repeat']!=repeat or report['batch']!=1 or report['measured_videos']!=200
            or report['checkpoint_sha256']!=HEAD_SHA or report['manifest_sha256']!=MANIFEST_SHA
            or report['assets']['config.json']!=CONFIG_SHA or report['assets']['model.safetensors']!=WEIGHT_SHA
            or report['reference_sha256']!=REFERENCE_SHA or report['prompt']!=PROMPT
            or report['explicit_warmup_forwards']!=0 or not report['original_full_vocabulary_projection']
            or report['processed_frame_size']!=[448,448] or report['quality_readout_dtypes']!=['torch.float32']
            or report['gpu_uuid']!='GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55' or report['power_limit_w']!=250):
        raise ValueError('Original qualified spatial full-native/quality-head scope differs')


def derive(config_path=CONFIG):
    if sha(config_path)!=CONFIG_SHA:raise ValueError('Exact original Instruct config required')
    config=read(config_path);repeats=[];refs=[];signatures=[];total=Counter()
    for repeat in (1,2,3):
        folder=ROOT/f'qwen_repeat{repeat:02d}';report=read(folder/'report.json');validate(report,repeat)
        sealed={r['file']:r['sha256'] for r in read(folder/'files_manifest.json')}
        for name in ('report.json','input_audit.json','calls.csv'):
            if sha(folder/name)!=sealed[name]:raise ValueError('Sealed native input or cost differs')
        if sha(folder/'input_audit.json')!=report['input_audit_sha256']:raise ValueError('Original processor evidence differs')
        records=[request_mac(config,r) for r in read(folder/'input_audit.json')['samples']]
        with (folder/'calls.csv').open(encoding='utf-8-sig',newline='') as f:calls=list(csv.DictReader(f))
        if (len(records)!=200 or [r['sample_id'] for r in records]!=[r['sample_id'] for r in calls]
                or sum(r['actual_sequence_length'] for r in records)!=report['effective_input_tokens200']):
            raise ValueError('Original first200 IDs/order/effective tokens differ')
        bindings={}
        for name in ('quality_token_batch_benchmark_200.py','quality_token_common.py','evaluate_temporal_baseline_batches.py','quality_token_dataset_once.py'):
            path=(TASK/name).relative_to(REPO).as_posix();blob=subprocess.check_output(['git','show',report['source_commit']+':'+path],cwd=REPO)
            if hashlib.sha256(blob).hexdigest()!=report['sources'][path]:raise ValueError('Original backbone/vocabulary/quality graph differs')
            bindings[path]=report['sources'][path]
        signatures.append([(r['sample_id'],r['small_values_sha256'],r['MAC']) for r in records])
        for r in records:total.update(r['MAC_terms'])
        repeats.append(dict(repeat=repeat,MAC200=sum(r['MAC'] for r in records),records=records))
        refs.append(dict(path=folder.relative_to(REPO).as_posix(),report_sha256=sha(folder/'report.json'),
            input_audit_sha256=sha(folder/'input_audit.json'),calls_sha256=sha(folder/'calls.csv'),graph_sources=bindings,
            source_commit=report['source_commit'],seconds200=report['cost']['continuous_wall_s'],joules200=report['cost']['estimated_system_energy_j']))
    if signatures[1:]!=signatures[:1]*2:raise ValueError('Original three spatial workloads differ')
    op=2*sum(r['MAC200'] for r in repeats);seconds=sum(r['seconds200'] for r in refs);energy=sum(r['joules200'] for r in refs)
    return dict(status='ORIGINAL_SPATIAL_NETWORK_AND_SAVED_SHAPES_DENSE_MAC_DERIVED',calls=600,MAC_OP600=op,
        TOP_per_input=op/600/1e12,equivalent_TOP_per_input_frame=op/2400/1e12,TOPS=op/1e12/seconds,TOPS_per_W=op/1e12/energy,
        MAC_terms600=dict(total),repeat_results=repeats,cost_references=refs,config_sha256=CONFIG_SHA,head_sha256=HEAD_SHA,
        GPU_forwards=0,new_neural_forwards=0,new_timing_or_power_used=False,
        original_full_vocabulary='once per last pooled vector, then trained five quality rows; NOT every language token',
        convention='1 real dense MAC=2OP; original24Vision/28Language/GQA/all4mergers/per-time-slice Vision attention',
        complete_scalar_OP_claim=False,non_MAC_not_in_subtotal=['decode','processor','embedding lookup','norm','bias','activation','softmax','RoPE/trig/elementwise','position interpolation','pooling','layout'],
        formula_source_sha256=sha(HERE/'lgvq_qwen_saved_shape_mac.py'),derivation_source_sha256=sha(__file__),
        derivation_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path,default=CONFIG);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve prior derivation')
    v=derive(a.config)
    with a.output.open('x',encoding='utf8') as f:json.dump(v,f,indent=2)
    print(json.dumps({k:v[k] for k in ('TOP_per_input','TOPS','TOPS_per_W','GPU_forwards')}))
