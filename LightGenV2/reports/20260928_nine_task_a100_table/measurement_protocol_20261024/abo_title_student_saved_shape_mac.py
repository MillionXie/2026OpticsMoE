"""Original .86 title NN MAC from sealed600 lengths, PT shapes and call ledger.

No Torch/model load or new neural/GPU call. B1 means electronic sequence rows
are the actual valid length, not the224-row maximum optical aperture. Preserve
actual residual execution separately from the selected-stage projection.
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
from measure_abo_text_qwen import sha
from operation_ledger import linear_mac,convolution_mac

TASK=REPO/'LightGenV2/tasks/t08_abo_image_text_retrieval'
BODY_SHA='cc977b83286a8e90398ebd30064428886bc3c06f1eb0557e470060f7ff5c1cae'
META=TASK/'runs/simulation/a100_title_saved_shape_metadata_20261025_01/metadata.json'
META_SHA='4354b0a5c49683700546fa9b321b73e00bb4a3098fa51c4fc1c07d85a2da2da2'
RUNS=[TASK/'runs/simulation'/name/f'ours_repeat{i:02d}' for i,name in (
    (1,'a100_title_student_input_inclusive_20261024_02'),
    (2,'a100_title_student_input_inclusive_20261025_04'),
    (3,'a100_title_student_input_inclusive_20261025_04'))]
ROPE=TASK/'runs/simulation/a100_title_operations_20261024_01/operation_counts.json'
CONFIG=ROPE.parent/'audit_original/model_config.json'
GRAPH_FILES=[
    'experiments/qwen3_vl_embedding_2b_caltech101_balanced_optical_fusion_ablation/modeling.py',
    'experiments/qwen3_vl_embedding_2b_caltech101_electronic_retrieval/electronic_blocks.py',
    'experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust/optical_blocks.py',
    'experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/optics/moe.py',
    'experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/optics/replacement.py',
    'experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/features.py']


def expected_counts():
    return {'language.input_adapter':1,'language.output_adapter':2,
        'language.parallel_residual1':1,'language.parallel_residual2':1,
        'language.amplitude_linear_norm_softplus':2,
        'language.expert_ccd_output_linear':1,'language.global_ccd_output_linear':1,
        'language.router_simulation':1,'language.expert_global_simulation':2,
        'head.retrieval_readout':1,'output.cpu_normalize_cosine_stable_rank':1}


def validate_counts(counts):
    if any(counts[k]!=v for k,v in expected_counts().items()):
        raise ValueError('Actual original title module occurrences differ')
    if any(v for k,v in counts.items() if k.startswith(('vision.','bridge.','frontend.patch'))):
        raise ValueError('Not the adopted pure-title online graph')


def call_mac(length,language,readout,rope_mac):
    if not isinstance(length,int) or not 0<length<=224 or rope_mac!=192*length:
        raise ValueError('Actual B1 length/native position shape differs')
    def linear(name,tokens=length):return linear_mac([1,tokens,language[name]['shape'][1]],language[name]['shape'])
    serial={
        'language.input_adapter':linear('core.input_adapter.weight'),
        'language.output_adapter_twice':2*linear('core.output_adapter.weight'),
        'language.amplitude_adapter_twice':2*linear('core.optical_branch.core.input_adapter.weight'),
        'language.expert_ccd_output_adapter':linear('core.optical_branch.expert_output_adapter.weight'),
        'language.global_ccd_output_adapter':linear('core.optical_branch.core.output_adapter.weight'),
        'head.retrieval_projection':linear_mac([1,384],readout['projection.weight']['shape']),
        'CPU_gallery_dot2400x64':2400*64}
    residual={}
    for i in (0,1):
        prefix=f'core.blocks.{i}.'
        weight=language[prefix+'token_depthwise.weight']['shape']
        # Original causal F.pad adds four left rows, Conv1d outputs L rows.
        residual[prefix+'token_depthwise']=convolution_mac([1,192,length+4],weight,[1,192,length],groups=192)
        for name in ('token_pointwise.weight','mlp.0.weight','mlp.3.weight'):
            residual[prefix+name]=linear(prefix+name)
    return dict(selected_serial_MAC=sum(serial.values()),parallel_residual_MAC=sum(residual.values()),
        native_position_construction_MAC=rope_mac,
        actual_MAC=sum(serial.values())+sum(residual.values())+rope_mac,
        selected_components=serial,parallel_components=residual)


def original_rope_shapes(config):
    if config['text_config']['head_dim']!=128:raise ValueError('Native head/position dimensions changed')
    data=json.loads(ROPE.read_text());result=Counter()
    for row in data['records']:
        args=row['inputs']
        if (row['operator']=='aten.matmul.default' and len(args)==2
                and args[0].get('shape')==[3,1,64,1]
                and args[1].get('shape',[])[:3]==[3,1,1]):
            length=args[1]['shape'][3]
            if row['mac']!=192*length or row['calls']!=1:raise ValueError('Original native position contraction changed')
            result[row['sample_index']]+=row['mac']
    if set(result)!=set(range(200)):raise ValueError('Original200 native position records are incomplete')
    return result


def bind_source_bytes(blob,local,pin):
    # Existing CPU auditors bind original runtime bytes to its published Git
    # blob. A Windows checkout may contain CRLF; no other edit is accepted.
    if (hashlib.sha256(blob).hexdigest()!=pin
            or local.replace(b'\r\n',b'\n')!=blob.replace(b'\r\n',b'\n')):
        raise ValueError('Not the exact original published source plus checkout newlines')


def derive():
    if sha(META)!=META_SHA:raise ValueError('Preserve original checked PT metadata')
    meta=json.loads(META.read_text());language=meta['language_optical'];readout=meta['retrieval_readout']
    if meta['checkpoint_sha256']!=BODY_SHA or not meta['no_Torch_import_or_GPU_or_neural_forwards']:
        raise ValueError('Wrong adopted body metadata')
    # All matrix weights in this original live language/head graph are accounted.
    expected={'core.input_adapter.weight','core.output_adapter.weight','core.optical_branch.core.input_adapter.weight',
        'core.optical_branch.core.output_adapter.weight','core.optical_branch.expert_output_adapter.weight'}
    expected.update(f'core.blocks.{i}.{suffix}' for i in (0,1) for suffix in
        ('token_depthwise.weight','token_pointwise.weight','mlp.0.weight','mlp.3.weight'))
    actual={k for k,v in language.items() if k.endswith('.weight') and len(v['shape'])>=2}
    if actual!=expected or readout['projection.weight']['shape']!=[64,384]:raise ValueError('Unaccounted original matrix weights')
    rope=original_rope_shapes(json.loads(CONFIG.read_text()));results=[];refs=[];native_tokens=0
    for folder in RUNS:
        sealed={x['file']:x['sha256'] for x in json.loads((folder/'files_manifest.json').read_text())}
        for name in ('calls.csv','stages.csv','identity.json','summary.json','stage_ledger.json'):
            if sha(folder/name)!=sealed[name]:raise ValueError('Sealed actual cost/shape data changed')
        identity=json.loads((folder/'identity.json').read_text())
        summary=json.loads((folder/'summary.json').read_text())
        if (identity['body_sha256']!=BODY_SHA or summary['status']!='complete' or summary['valid_samples']!=200
                or sha(CONFIG)!=identity['model_config_sha256']):
            raise ValueError('Original model/config/qualified cost binding differs')
        # The fixed run selection has already passed both full raw audits.
        # Bind the original sealed summary here, not a host-specific audit-file
        # name; the index builder separately replays original cost audits.
        if (folder/'local_independent_audit01.json').exists():
            audit=json.loads((folder/'local_independent_audit01.json').read_text())
            if audit['status']!='passed' or audit['calls']!=200 or audit['report_sha256']!=sealed['summary.json']:
                raise ValueError('Existing independent cost audit differs')
        settings=identity['resolved_settings']
        for key,value in dict(inference_batch_size=1,electronic_width=192,electronic_layers=2,
                electronic_expansion=.5,electronic_token_mixer_enabled=True,
                electronic_language_token_mixer_type='depthwise_conv1d',electronic_language_token_mixer_kernel_size=5,
                student_language_mode='optical_moe',native_pre_attention_enabled=False,transformer_residual_enabled=False).items():
            if settings[key]!=value:raise ValueError('Original neural graph setting differs: '+key)
        for name in GRAPH_FILES:
            blob=subprocess.check_output(['git','show',identity['source_commit']+':'+name],cwd=REPO)
            bind_source_bytes(blob,(REPO/name).read_bytes(),identity['sources'][name])
        with (folder/'stages.csv').open(encoding='utf-8-sig',newline='') as f:stages=list(csv.DictReader(f))
        stage_counts={i:Counter() for i in range(200)}
        for row in stages:stage_counts[int(row['sample_index'])][row['stage']]+=1
        with (folder/'calls.csv').open(encoding='utf-8-sig',newline='') as f:calls=list(csv.DictReader(f))
        if len(calls)!=200:raise ValueError('Wrong qualified call count')
        per_call=[]
        for i,row in enumerate(calls):
            validate_counts(stage_counts[i]);ids=json.loads(row['input_ids']);mask=json.loads(row['attention_mask'])
            length=int(row['valid_input_tokens'])
            if (int(row['index'])!=i or json.loads(row['input_ids_shape'])!=[1,length]
                    or len(ids)!=1 or len(ids[0])!=length or mask!=[[1]*length]
                    or json.loads(row['detector_feature_shape'])!=[1,384]
                    or json.loads(row['embedding_shape'])!=[1,64]):raise ValueError('Actual B1 input/head shapes differ')
            per_call.append(dict(index=i,product_id=row['product_id'],length=length,**call_mac(length,language,readout,rope[i])))
            native_tokens+=length
        results.append(per_call)
        refs.append(dict(run=folder.relative_to(REPO).as_posix(),summary_sha256=sealed['summary.json'],
            calls_sha256=sealed['calls.csv'],stage_shapes_or_occurrences_sha256=sealed['stages.csv'],
            seconds200=(summary['charged_electronic_ms_total']+summary['optical_ms_total'])/1000,
            joules200=121.746*summary['optical_ms_total']/1000+104.230*summary['charged_electronic_ms_total']/1000))
    if results[1:]!=[results[0],results[0]]:raise ValueError('Original shape workload differs across fresh rounds')
    total={k:sum(c[k] for rows in results for c in rows) for k in
        ('actual_MAC','selected_serial_MAC','parallel_residual_MAC','native_position_construction_MAC')}
    seconds=sum(x['seconds200'] for x in refs);joules=sum(x['joules200'] for x in refs)
    return dict(status='ORIGINAL_GRAPH_SEALED600_LENGTHS_AND_PT_SHAPES_MAC_DERIVED',calls=600,
        checkpoint_sha256=BODY_SHA,metadata_sha256=META_SHA,metadata_path=META.relative_to(REPO).as_posix(),
        original_config_sha256=sha(CONFIG),original_native_position_shape_receipt_sha256=sha(ROPE),
        original_graph_sources={k:identity['sources'][k] for k in GRAPH_FILES},cost_references=refs,
        graph_binding='Exact original runtime Git blob SHA; current checkout only CRLF/LF differences permitted',
        actual_NN_MAC600=total['actual_MAC'],charged_serial_NN_MAC600=total['selected_serial_MAC'],
        parallel_residual_NN_MAC600=total['parallel_residual_MAC'],
        native_position_MAC600_not_in_selected_timing_spans=total['native_position_construction_MAC'],
        actual_NN_TOP_per_input=2*total['actual_MAC']/600/1e12,
        charged_serial_NN_TOP_per_input=2*total['selected_serial_MAC']/600/1e12,
        actual_NN_MAC_convention_TOPS=2*total['actual_MAC']/1e12/seconds,
        actual_NN_MAC_convention_TOPS_per_W=2*total['actual_MAC']/1e12/joules,
        charged_serial_NN_MAC_convention_TOPS=2*total['selected_serial_MAC']/1e12/seconds,
        charged_serial_NN_MAC_convention_TOPS_per_W=2*total['selected_serial_MAC']/1e12/joules,
        actual_native_tokens600=native_tokens,per_call_first_repeat=results[0],
        no_GPU_or_neural_forwards=True,no_new_time_or_power=True,one_real_MAC_equals_two_OP=True,
        online_vision_or_merger_calls=0,offline_gallery_inference_included=False,
        non_MAC=['token lookup','norm/activation/bias','mean/max pooling','normalization/fusion/RMS',
            'phase/complex propagation','mask/layout/transfer','CPU sorting'],
        physical_optical_dense_equivalent_OP=None,FFT_simulation_not_counted_as_physical_OP=True,
        complete_scalar_OP_claim=False,source_sha256=sha(Path(__file__)),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve earlier derived receipt')
    result=derive()
    with a.output.open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:result[k] for k in ('status','calls','actual_NN_TOP_per_input',
        'charged_serial_NN_TOP_per_input','actual_NN_MAC_convention_TOPS','actual_NN_MAC_convention_TOPS_per_W')}))


if __name__=='__main__':main()
