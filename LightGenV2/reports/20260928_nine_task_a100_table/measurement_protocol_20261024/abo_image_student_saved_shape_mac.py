"""Original .8058 image student NN MAC from sealed600 shapes and PT metadata.

No Torch/model/forward/GPU. Original student disables auxiliary DeepStack;
only the main native merger executes. Separate actual NN, selected serial,
parallel residual, native-position construction, optics and simulation FFT.
"""
import argparse
import ast
from collections import Counter
import csv
import json
from pathlib import Path
import subprocess
from abo_title_student_saved_shape_mac import GRAPH_FILES,bind_source_bytes
from operation_ledger import linear_mac,convolution_mac
from measure_abo_text_qwen import sha

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
TASK=REPO/'LightGenV2/tasks/t08_abo_image_text_retrieval'
ROOT=TASK/'runs/simulation/a100_image_student_input_inclusive_20261025_04'
BODY_SHA='a2aa9a93028410dfb780df7d320a9be65d07b6e7ec87ede08b53c4bddcbaf7af'
META=TASK/'runs/simulation/a100_image_saved_shape_metadata_20261025_01/metadata.json'
META_SHA='9c0c061990ae5a9abfae09b43924c692dcdaebfabea66ed8591a207971dbcd20'
CONFIG=TASK/'runs/simulation/a100_title_operations_20261024_01/audit_original/model_config.json'
CONFIG_SHA='9172f55b0b9cce70b7f67b10c58a408ccf3ec15c587e6efd4d5f41631237fded'


def state_matrices(state):return {k for k,v in state.items() if k.endswith('.weight') and len(v['shape'])>=2}


def validate_state(state):
    expected={'core.input_adapter.weight','core.output_adapter.weight','core.optical_branch.core.input_adapter.weight',
        'core.optical_branch.core.output_adapter.weight','core.optical_branch.expert_output_adapter.weight'}
    expected.update(f'core.blocks.{i}.{name}' for i in (0,1) for name in ('token_depthwise.weight','token_pointwise.weight','mlp.0.weight','mlp.3.weight'))
    if state_matrices(state)!=expected:raise ValueError('Unaccounted original matrix weights')


def modality_mac(state,length,vision):
    validate_state(state)
    def linear(name):return linear_mac([1,length,state[name]['shape'][1]],state[name]['shape'])
    serial=dict(input_adapter=linear('core.input_adapter.weight'),
        output_adapter=linear('core.output_adapter.weight')*(1 if vision else 2),
        amplitude_adapter_twice=2*linear('core.optical_branch.core.input_adapter.weight'),
        expert_ccd_output_adapter=linear('core.optical_branch.expert_output_adapter.weight'),
        global_ccd_output_adapter=linear('core.optical_branch.core.output_adapter.weight'))
    residual={}
    for i in (0,1):
        prefix=f'core.blocks.{i}.';w=state[prefix+'token_depthwise.weight']['shape']
        if vision:
            if length!=196 or w!=[192,1,3,3]:raise ValueError('Original fixed14x14 spatial depthwise mixer required')
            residual[prefix+'token_depthwise']=convolution_mac([1,192,16,16],w,[1,192,14,14],groups=192)
        else:
            if w!=[192,1,5]:raise ValueError('Original five-tap causal mixer required')
            residual[prefix+'token_depthwise']=convolution_mac([1,192,length+4],w,[1,192,length],groups=192)
        for name in ('token_pointwise.weight','mlp.0.weight','mlp.3.weight'):residual[prefix+name]=linear(prefix+name)
    return serial,residual


def call_mac(meta,config,length):
    if not isinstance(length,int) or not 0<length<=224:raise ValueError('Actual B1 language length required')
    serial={};residual={}
    for name,rows,vision in [('vision',196,True),('language',length,False)]:
        s,r=modality_mac(meta[name+'_optical'],rows,vision)
        serial.update({name+'.'+k:v for k,v in s.items()});residual.update({name+'.'+k:v for k,v in r.items()})
    v=config['vision_config'];text=config['text_config'];grouped=v['hidden_size']*v['spatial_merge_size']**2
    if (v['hidden_size']!=1024 or v['spatial_merge_size']!=2 or text['head_dim']!=128
            or meta['retrieval_readout']['projection.weight']['shape']!=[64,384]):raise ValueError('Original frozen bridge/readout dimensions differ')
    serial.update(native_patch_embedding=196*3*v['temporal_patch_size']*v['patch_size']**2*v['hidden_size'],
        main_native_merger_only=49*(grouped*grouped+grouped*v['out_hidden_size']),
        retrieval_projection=384*64,CPU_100title_search=100*64)
    position=3*(text['head_dim']//2)*length
    return dict(actual_MAC=sum(serial.values())+sum(residual.values())+position,
        selected_serial_MAC=sum(serial.values()),parallel_residual_MAC=sum(residual.values()),
        native_position_MAC_not_in_selected_spans=position,selected_components=serial,parallel_components=residual)


def validate_counts(counts):
    expected={'frontend.patch_embedding':1,'bridge.native_merger':1,'head.retrieval_readout':1,
        'output.cpu_original_cosine_nonstable_rank':1}
    for mode in ('vision','language'):
        expected.update({mode+'.'+k:v for k,v in dict(input_adapter=1,output_adapter=1 if mode=='vision' else 2,
            amplitude_linear_norm_softplus=2,expert_ccd_output_linear=1,global_ccd_output_linear=1,
            parallel_residual1=1,parallel_residual2=1,router_simulation=1,expert_global_simulation=2).items()})
    if any(counts[k]!=v for k,v in expected.items()) or any(v for k,v in counts.items() if k.startswith('bridge.deepstack')):
        raise ValueError('Original actual six-pass/one-main-merger module occurrences differ')


def derive(config_path=CONFIG):
    if sha(META)!=META_SHA or sha(config_path)!=CONFIG_SHA:raise ValueError('Original checked metadata/config differs')
    meta=json.loads(META.read_text());config=json.loads(Path(config_path).read_text())
    if (meta['checkpoint_sha256']!=BODY_SHA or meta['epoch']!=25 or meta['weight_variant']!='ema'
            or not meta['no_Torch_import_or_GPU_or_neural_forwards']):raise ValueError('Original adopted EMA metadata differs')
    repeats=[];refs=[];signatures=[]
    for repeat in (1,2,3):
        folder=ROOT/f'ours_repeat{repeat:02d}';identity=json.loads((folder/'identity.json').read_text());summary=json.loads((folder/'summary.json').read_text())
        sealed={r['file']:r['sha256'] for r in json.loads((folder/'files_manifest.json').read_text())}
        for name in ('identity.json','summary.json','calls.csv','stages.csv'):
            if sha(folder/name)!=sealed[name]:raise ValueError('Sealed original shape/cost differs')
        settings=identity['resolved_settings']
        if (identity['body_sha256']!=BODY_SHA or identity['model_config_sha256']!=CONFIG_SHA
                or summary['status']!='complete' or summary['valid_samples']!=200 or summary['repeat']!=repeat):raise ValueError('Qualified adopted cost differs')
        for key,value in dict(inference_batch_size=1,electronic_width=192,electronic_layers=2,electronic_expansion=2.,
                electronic_deepstack_enabled=False,student_language_mode='optical_moe',native_pre_attention_enabled=False,
                transformer_residual_enabled=False).items():
            if settings[key]!=value:raise ValueError('Original graph setting differs: '+key)
        for name in GRAPH_FILES:
            blob=subprocess.check_output(['git','show',identity['source_commit']+':'+name],cwd=REPO)
            bind_source_bytes(blob,(REPO/name).read_bytes(),identity['sources'][name])
        counts={i:Counter() for i in range(200)}
        with (folder/'stages.csv').open(encoding='utf-8-sig',newline='') as f:
            for row in csv.DictReader(f):counts[int(row['sample_index'])][row['stage']]+=1
        with (folder/'calls.csv').open(encoding='utf-8-sig',newline='') as f:calls=list(csv.DictReader(f))
        if len(calls)!=200:raise ValueError('All original200 records required')
        records=[]
        for i,row in enumerate(calls):
            validate_counts(counts[i]);shapes=ast.literal_eval(row['input_shapes']);ids=json.loads(row['input_ids']);mask=json.loads(row['attention_mask']);length=int(row['valid_input_tokens'])
            if (int(row['index'])!=i or shapes['pixel_values']!=[196,1536] or shapes['image_grid_thw']!=[1,3]
                    or shapes['input_ids']!=[1,length] or ids!=[ids[0]] or len(ids[0])!=length or mask!=[[1]*length]
                    or json.loads(row['embedding_shape'])!=[1,64] or json.loads(row['detector_feature_shape'])!=[1,384]):
                raise ValueError('Actual B1 fixed224/valid language/head shapes differ')
            records.append(dict(index=i,sample_id=row['sample_id'],length=length,**call_mac(meta,config,length)))
        signatures.append(records);repeats.append(dict(repeat=repeat,records=records))
        refs.append(dict(run=folder.relative_to(REPO).as_posix(),summary_sha256=sealed['summary.json'],calls_sha256=sealed['calls.csv'],
            stage_occurrences_sha256=sealed['stages.csv'],seconds200=(summary['charged_electronic_ms_total']+summary['optical_ms_total'])/1000,
            joules200=121.746*summary['optical_ms_total']/1000+104.230*summary['charged_electronic_ms_total']/1000))
    if signatures[1:]!=[signatures[0],signatures[0]]:raise ValueError('Original fresh workloads differ')
    total={k:sum(r[k] for rows in signatures for r in rows) for k in ('actual_MAC','selected_serial_MAC','parallel_residual_MAC','native_position_MAC_not_in_selected_spans')}
    seconds=sum(r['seconds200'] for r in refs);energy=sum(r['joules200'] for r in refs);op=total['actual_MAC']*2
    return dict(status='ORIGINAL_IMAGE_STUDENT_GRAPH_AND_SEALED600_SHAPES_MAC_DERIVED',calls=600,
        checkpoint_sha256=BODY_SHA,metadata_sha256=META_SHA,metadata_path=META.relative_to(REPO).as_posix(),
        actual_NN_MAC600=total['actual_MAC'],charged_serial_NN_MAC600=total['selected_serial_MAC'],parallel_residual_NN_MAC600=total['parallel_residual_MAC'],
        native_position_MAC600_not_in_selected_timing_spans=total['native_position_MAC_not_in_selected_spans'],
        actual_NN_TOP_per_input=op/600/1e12,charged_serial_NN_TOP_per_input=2*total['selected_serial_MAC']/600/1e12,
        actual_NN_MAC_convention_TOPS=op/1e12/seconds,actual_NN_MAC_convention_TOPS_per_W=op/1e12/energy,
        per_call_first_repeat=signatures[0],cost_references=refs,
        original_graph_sources={k:identity['sources'][k] for k in GRAPH_FILES},
        grid_scope='14x14 from original fixed224 square processor plus saved196 patches/[1,3] grid shape; grid values not duplicated in calls',
        native_mergers_executed_per_query=1,auxiliary_deepstack_mergers_executed=0,offline_gallery_inference_included=False,
        no_GPU_or_neural_forwards=True,no_new_time_or_power=True,physical_optical_dense_equivalent_OP=None,
        FFT_simulation_not_counted_as_physical_OP=True,complete_scalar_OP_claim=False,
        non_MAC=['embedding lookup','norm/activation/bias','fusion/RMS/compatibility diagnostics','position interpolation/vision rotary outer/trig','CCD pooling','normalization/sorting','phase/complex optical propagation','layout'],
        source_sha256=sha(__file__),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path,default=CONFIG);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve prior derivation')
    v=derive(a.config)
    with a.output.open('x',encoding='utf8') as f:json.dump(v,f,indent=2)
    print(json.dumps({k:v[k] for k in ('actual_NN_TOP_per_input','charged_serial_NN_TOP_per_input','actual_NN_MAC_convention_TOPS','actual_NN_MAC_convention_TOPS_per_W')}))
