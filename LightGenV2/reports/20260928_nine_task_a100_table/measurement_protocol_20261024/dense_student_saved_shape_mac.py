"""Original LSP/SALICON optical-student shape metadata, never Torch loading."""
import argparse
import ast
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
from torch_metadata import MetadataUnpickler,first_pickle
from dense_vision_qwen_saved_shape_mac import CONFIG,CONFIG_SHA,WEIGHT_SHA
from native_shape_mac import TRANSFORMERS_QWEN3VL_SOURCE_SHA

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
GROUPS={
 'LSP':('t02_keypoint_detection','a100_input_inclusive_20261024_02','head',
        '495b9c2c4e3df15d3715f1ce8f2faea7cb9156275b31103ec684f4e96a328518'),
 'SALICON':('t03_saliency','a100_input_inclusive_20261024_01','saliency_head',
        '036bc8caedcde4d6dabce276b1a2e4af15d960d88620098b19e1c198849b8bfe')}
METADATA_SHA={'LSP':'d770140a1ff4a4b530c4e0eb6a87a6d34733fee82cb39eb00727564e221544cd',
              'SALICON':'c8a551141a08444b521351644559e9515c5fc13636e44cf1a4aa5112fa18cea4'}


def inert_posix_path(*parts):return dict(__inert_posix_path_parts__=list(parts))


class DenseMetadataUnpickler(MetadataUnpickler):
    def find_class(self,module,name):
        # These two original metadata values are inert descriptions, not imports
        # of Torch or checkpoint/model code. Unknown globals still hard-fail.
        if (module,name)==('pathlib','PosixPath'):return inert_posix_path
        if (module,name)==('torch.torch_version','TorchVersion'):return str
        return super().find_class(module,name)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def extract_metadata(task):
    directory,run,head_key,pin=GROUPS[task]
    identity=read(REPO/'LightGenV2/tasks'/directory/'runs/simulation'/run/'ours_repeat01/identity.json')
    # Original recorded server path is only a DATA alias; no old source imported.
    marker='/2026OpticsMoE/'
    checkpoint=REPO/identity['checkpoint'].replace('\\','/').split(marker,1)[1]
    if identity['checkpoint_sha256']!=pin or sha(checkpoint)!=pin:raise ValueError('Original approved checkpoint differs')
    with checkpoint.open('rb') as f:raw=first_pickle(f)
    payload=DenseMetadataUnpickler(io.BytesIO(raw)).load()
    if not isinstance(payload,dict):raise ValueError('Original tensor metadata mapping required')
    return dict(task=task,checkpoint_path=checkpoint.relative_to(REPO).as_posix(),checkpoint_sha256=pin,
        checkpoint_bytes=checkpoint.stat().st_size,metadata_pickle_sha256=hashlib.sha256(raw).hexdigest(),metadata_pickle_bytes=len(raw),
        state_maps=dict(core=payload['core'],head=payload[head_key]),epoch=payload.get('epoch'),
        architecture=payload.get('architecture',payload.get('checkpoint_architecture')),
        no_Torch_import_or_model_forward=True,metadata_only_not_tensor_content_or_numerical_proof=True,
        source_sha256=sha(__file__),reader_source_sha256=sha(HERE/'torch_metadata.py'))


def nn_terms(task,maps):
    """Actual one-merger Linear/Conv NN subtotal, not physical propagation."""
    used={group:set() for group in maps}
    def weight(group,name,area,shape):
        value=maps[group][name]
        if value['shape']!=list(shape):raise ValueError('Original matrix/kernel differs: '+name)
        used[group].add(name);count=area
        for n in shape:count*=n
        return count
    serial=dict(native_patch196=196*1536*1024,native_one_main_merger49=49*(4096**2+4096*2048))
    serial['input_adapter']=weight('core','hybrid.input_adapter.weight',196,(192,1024))
    serial['output_adapter_executed_even_if_head_uses_latent']=weight('core','hybrid.output_adapter.weight',196,(1024,192))
    serial['amplitude_adapter_twice']=2*weight('core','hybrid.optical_branch.core.input_adapter.weight',196,(224,192))
    serial['expert_CCD_output']=weight('core','hybrid.optical_branch.expert_output_adapter.weight',196,(192,224))
    serial['global_CCD_output']=weight('core','hybrid.optical_branch.core.output_adapter.weight',196,(192,224))
    residual={}
    for i in (0,1):
        prefix=f'hybrid.blocks.{i}.'
        for name,shape in [('token_depthwise.weight',(192,1,3,3)),('token_pointwise.weight',(192,192)),
                           ('mlp.0.weight',(384,192)),('mlp.3.weight',(192,384))]:
            residual[prefix+name]=weight('core',prefix+name,196,shape)
        if task=='SALICON':residual[prefix+'expanded_CFFN_DW3']=weight('core',prefix+'mlp.1.0.conv.weight',196,(384,1,3,3))
    current=160 if task=='LSP' else 128
    serial['head.token_projection']=weight('head','token_projection.weight',196,(current,192))
    for i in (0,1):
        serial[f'head.body{i}.DW']=weight('head',f'body.{i}.block.0.weight',196,(current,1,3,3))
        serial[f'head.body{i}.pointwise']=weight('head',f'body.{i}.block.1.weight',196,(current,current,1,1))
    height=14
    for i,after in enumerate((128,96) if task=='LSP' else (96,64,32,16)):
        height*=2
        serial[f'head.up{height}.DW']=weight('head',f'upsample.{i}.block.0.weight',height**2,(current,1,3,3))
        serial[f'head.up{height}.pointwise']=weight('head',f'upsample.{i}.block.1.weight',height**2,(after,current,1,1))
        current=after
    serial['head.refine.DW']=weight('head','refine.block.0.weight',height**2,(current,1,3,3))
    serial['head.refine.pointwise']=weight('head','refine.block.1.weight',height**2,(current,current,1,1))
    name,outputs=('predictor',14) if task=='LSP' else ('classifier',1)
    serial['head.output']=weight('head',name+'.weight',height**2,(outputs,current,1,1))
    for group,values in maps.items():
        expected={k for k,v in values.items() if isinstance(v,dict) and v.get('__tensor_descriptor__') and k.endswith('weight') and len(v['shape'])>=2}
        if expected!=used[group]:raise ValueError('Unaccounted/invented NN matrix: '+str(expected^used[group]))
    return dict(actual_NN_MAC=sum(serial.values())+sum(residual.values()),charged_serial_NN_MAC=sum(serial.values()),
        parallel_residual_NN_MAC=sum(residual.values()),serial_NN_terms=serial,parallel_NN_terms=residual,
        other_executed_digital_dense_MAC={'GPU_router_dense_mask_contraction':4*478**2})


def derive():
    if sha(CONFIG)!=CONFIG_SHA:raise ValueError('Exact original native frontend config required')
    config=read(CONFIG)
    if config['vision_config']['hidden_size']!=1024:raise ValueError('Original patch/one-merger frontend differs')
    groups=[]
    for task,(directory,run,_,body_sha) in GROUPS.items():
        task_root=REPO/'LightGenV2/tasks'/directory
        meta_path=task_root/'runs/simulation/a100_dense_saved_shape_metadata_20261025_01/metadata_server_01.json'
        if sha(meta_path)!=METADATA_SHA[task]:raise ValueError('Original independent tensor descriptors changed')
        meta=read(meta_path)
        if meta['checkpoint_sha256']!=body_sha or not meta['no_Torch_import_or_model_forward']:raise ValueError('Original metadata scope differs')
        terms=nn_terms(task,meta['state_maps']);repeats=[];refs=[];signatures=[];sources={}
        for repeat in (1,2,3):
            folder=task_root/'runs/simulation'/run/f'ours_repeat{repeat:02d}';identity=read(folder/'identity.json');summary=read(folder/'summary.json')
            sealed={r['file']:r['sha256'] for r in read(folder/'files_manifest.json')}
            for name in ('identity.json','samples.csv','stages.csv','summary.json','stage_ledger.json'):
                if sha(folder/name)!=sealed[name]:raise ValueError('Sealed original input/stages/cost differs')
            settings=identity['resolved_settings']
            if (identity['system']!='ours' or identity['checkpoint_sha256']!=body_sha or identity['explicit_warmup_forwards']!=0
                    or identity['repeat']!=repeat or summary['status']!='complete' or summary['valid_samples']!=200
                    or settings['electronic_width']!=192 or settings['electronic_layers']!=2
                    or settings['native_pre_attention_enabled'] or settings['transformer_residual_enabled']
                    or identity['native_model_assets']['config.json']!=CONFIG_SHA
                    or identity['native_model_assets']['model.safetensors']!=WEIGHT_SHA):raise ValueError('Original dense student graph differs')
            ledger=read(folder/'stage_ledger.json')
            names={r['source_file'].replace('\\','/').split('/2026OpticsMoE/',1)[1] for r in ledger if '/2026OpticsMoE/' in r['source_file'].replace('\\','/')}
            names.update(['LightGenV2/tasks/t02_keypoint_detection/modeling.py',
                'experiments/qwen3_vl_embedding_2b_lsp_pose_optical_router/modeling.py'])
            if task=='SALICON':names.add('LightGenV2/tasks/t03_saliency/lightweight_residual.py')
            for name in sorted(names):
                blob=subprocess.check_output(['git','show',identity['source_commit']+':'+name],cwd=REPO)
                if (blob!=subprocess.check_output(['git','show','HEAD:'+name],cwd=REPO)
                        or blob!=(REPO/name).read_bytes().replace(b'\r\n',b'\n')):raise ValueError('Original graph definition changed: '+name)
                pin=hashlib.sha256(blob).hexdigest()
                for row in ledger:
                    if row['source_file'].replace('\\','/').endswith('/'+name) and row['source_sha256']!=pin:raise ValueError('Original runtime graph SHA differs')
                sources[name]=pin
            counts={i:Counter() for i in range(200)}
            with (folder/'stages.csv').open(encoding='utf-8-sig',newline='') as f:
                for row in csv.DictReader(f):counts[int(row['sample_index'])][row['stage']]+=1
            with (folder/'samples.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
            if len(rows)!=200:raise ValueError('All original200 calls required')
            records=[]
            for i,row in enumerate(rows):
                if (int(row['index'])!=i or ast.literal_eval(row['pixel_values_shape'])!=[196,1536]
                        or ast.literal_eval(row['image_grid_thw'])!=[[1,14,14]] or any(k.startswith('bridge.deepstack_merger') for k in counts[i])):
                    raise ValueError('Original native fixed224/one-merger shape differs')
                for name,n in [('frontend.patch_embedding',1),('bridge.native_merger',1),('vision.input_adapter_norm',1),('vision.output_adapter',1),
                    ('vision.amplitude_linear_norm_softplus',2),('vision.expert_ccd_output_linear',1),('vision.global_ccd_output_linear',1),
                    ('vision.parallel_residual1',1),('vision.parallel_residual2',1),('vision.router_simulation',1),('vision.expert_global_simulation',2)]:
                    if counts[i][name]!=n:raise ValueError('Original actual adapters/three-pass/residual calls differ')
                records.append(dict(index=i,sample_id=row['sample_id'],MAC=terms['actual_NN_MAC']))
            signatures.append(records);repeats.append(dict(repeat=repeat,MAC200=sum(r['MAC'] for r in records),records=records))
            refs.append(dict(run=folder.relative_to(REPO).as_posix(),identity_sha256=sealed['identity.json'],samples_sha256=sealed['samples.csv'],
                stages_sha256=sealed['stages.csv'],summary_sha256=sealed['summary.json'],
                seconds200=(summary['charged_electronic_ms_total']+summary['optical_ms_total'])/1000,
                joules200=summary['cost']['sample_j']*200))
        if signatures[1:]!=[signatures[0],signatures[0]]:raise ValueError('Original repeated workloads differ')
        op=2*sum(r['MAC200'] for r in repeats);seconds=sum(r['seconds200'] for r in refs);energy=sum(r['joules200'] for r in refs)
        groups.append(dict(task=task,calls=600,actual_NN_MAC_OP600=op,charged_serial_NN_OP600=1200*terms['charged_serial_NN_MAC'],
            parallel_residual_NN_OP600=1200*terms['parallel_residual_NN_MAC'],TOP_per_input=op/600/1e12,TOPS=op/1e12/seconds,TOPS_per_W=op/1e12/energy,
            per_query_MAC=terms,other_executed_digital_dense_MAC600=600*sum(terms['other_executed_digital_dense_MAC'].values()),
            repeat_results=repeats,cost_references=refs,original_graph_sources=sources,checkpoint_sha256=body_sha,
            metadata_path=meta_path.relative_to(REPO).as_posix(),metadata_sha256=METADATA_SHA[task],native_mergers_executed_per_query=1,auxiliary_deepstack_mergers_executed=0))
    return dict(status='ORIGINAL_DENSE_STUDENT_GRAPH_AND_SAVED600_SHAPES_NN_MAC_DERIVED',groups=groups,
        model_or_GPU_forwards=0,new_time_or_power=False,complete_scalar_OP_claim=False,physical_optical_dense_equivalent_OP=None,
        FFT_not_physical_optical_OP=True,config_sha256=CONFIG_SHA,transformers_runtime_source_sha256=TRANSFORMERS_QWEN3VL_SOURCE_SHA,
        convention='1 real dense MAC=2OP; actual Linear/Conv NN subtotal; dense padded kernels; only one real main merger; digital router mask contraction separate',
        non_MAC=['norm/bias/activation','native position interpolation/rotary construction','pooling','bilinear resize','routing scalar diagnostics/topk/softmax','fusion','phase/complex propagation','layout','argmax/coordinate decode','map softmax'],
        no_native_Transformer_or_Language_or_LM_head=True,source_sha256=sha(__file__),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--extract-metadata',choices=tuple(GROUPS))
    p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve prior metadata')
    value=extract_metadata(a.extract_metadata) if a.extract_metadata else derive();a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf8') as f:json.dump(value,f,indent=2)
    print(json.dumps(dict(file=str(a.output),sha256=sha(a.output),bytes=a.output.stat().st_size,
        results=[{k:g[k] for k in ('task','TOP_per_input','TOPS','TOPS_per_W')} for g in value.get('groups',[])])))
