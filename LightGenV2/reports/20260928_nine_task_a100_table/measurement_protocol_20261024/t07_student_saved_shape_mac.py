"""Original .8225 compact graph and sealed600 actual shapes; no Torch/forwards."""
import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
ROOT=REPO/'LightGenV2/tasks/t07_abo_image_retrieval/runs/simulation/a100_input_inclusive_20261024_01'
META=ROOT.parent/'a100_standalone_saved_shape_metadata_20261025_01/metadata.json'
META_SHA='70fc3a68e92fafabea6be14756074de458910ff18dc7a113f9491832c63c4463'
BODY_SHA='0b1ff5b682b833f8c4fd25e6330147e993e974272d7d2812fa679082ef0e87bc'
GRAPH={
 'LightGenV2/tasks/t07_abo_image_retrieval/standalone/frontend.py':'6759ee3d8893388590d5952f0819b64c88db5380f4b8c964174bbd6b08de53b1',
 'LightGenV2/tasks/t07_abo_image_retrieval/standalone/model.py':'bd89044e8e516d3935c0315680d47205285c62e7cc1302549f39c66b532a4bd3',
 'LightGenV2/tasks/t07_abo_image_retrieval/standalone/optics.py':'ff07716e09337520a5f338be6097522226bb8f9d42b1e39feb3e2c15eab3d67f'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def nn_terms(state,length):
    """Dense padded real Conv/Linear MAC; non-NN router contraction separate."""
    def linear(name,rows,shape):
        if state[name]['shape']!=list(shape):raise ValueError('Original weight shape differs: '+name)
        return rows*shape[0]*shape[1]
    serial={'frontend_patch':196*1024*3*2*16*16}
    if state['frontend.patch.weight']['shape']!=[1024,3,2,16,16]:raise ValueError('Original compact patch shape differs')
    serial['one_main_merger']=linear('frontend.merger_fc1.weight',49,(4096,4096))+linear('frontend.merger_fc2.weight',49,(2048,4096))
    residual={}
    for mode,rows,hidden,kernel in [('vision',196,1024,25),('language',length,2048,5)]:
        serial[mode+'_input']=linear(mode+'.input_adapter.weight',rows,(192,hidden))
        serial[mode+'_amplitude_twice']=2*linear(mode+'.optics.input_adapter.weight',rows,(224,192))
        serial[mode+'_expert_readout']=linear(mode+'.optics.expert_output.weight',rows,(192,224))
        serial[mode+'_global_readout']=linear(mode+'.optics.global_output.weight',rows,(192,224))
        if mode=='vision':serial['vision_output_once']=linear('vision.output_adapter.weight',rows,(1024,192))
        elif 'language.output_adapter.weight' in state:raise ValueError('Unexecuted language output adapter introduced')
        for block in (0,1):
            prefix=f'{mode}.blocks.{block}.'
            expected=[192,1,5,5] if mode=='vision' else [192,1,5]
            if state[prefix+'token_depthwise.weight']['shape']!=expected:raise ValueError('Original kernel5 residual required')
            residual[prefix+'depthwise']=rows*192*kernel
            residual[prefix+'pointwise']=linear(prefix+'token_pointwise.weight',rows,(192,192))
            residual[prefix+'MLP']=linear(prefix+'mlp.0.weight',rows,(384,192))+linear(prefix+'mlp.3.weight',rows,(192,384))
    serial['spatial2x2_readout']=linear('readout.projection.weight',1,(64,1152))
    serial['GPU_gallery1600_descriptor64_dot']=1600*64
    # Executed digital einsum is not an NN module nor physical optical OP.
    other={'GPU_router_two_dense_mask_contractions':2*4*478*478}
    if any(k.startswith('late_rgb.') for k in state):raise ValueError('Later RGB bypass is not approved .8225 graph')
    return dict(actual_NN_MAC=sum(serial.values())+sum(residual.values()),charged_serial_NN_MAC=sum(serial.values()),
        parallel_residual_NN_MAC=sum(residual.values()),serial_NN_terms=serial,parallel_NN_terms=residual,
        other_executed_digital_dense_MAC=other)


def derive():
    if sha(META)!=META_SHA:raise ValueError('Exact original checkpoint metadata required')
    metadata=read(META)
    if metadata['checkpoint_sha256']!=BODY_SHA or not metadata['no_torch_import_or_model_forward']:raise ValueError('Original metadata provenance differs')
    payload=metadata['payload'];meta=payload['metadata'];state=payload['state_dict'];length=len(meta['template_ids'])
    if (length!=77 or meta['template_ids'].count(meta['image_token_id'])!=49 or meta.get('late_rgb_adapter') is not None
            or meta['electronic_context_kernels']!={'vision':5,'language':5} or meta['retrieval_head']!='spatial2x2_64'):
        raise ValueError('Original fixed77/one-merger/no-bypass graph differs')
    terms=nn_terms(state,length);refs=[];signatures=[]
    for repeat in (1,2,3):
        folder=ROOT/f'ours_repeat{repeat:02d}';identity=read(folder/'identity.json');summary=read(folder/'summary.json')
        sealed={r['file']:r['sha256'] for r in read(folder/'files_manifest.json')}
        for name in ('identity.json','summary.json','samples.csv','stages.csv','stage_ledger.json'):
            if sha(folder/name)!=sealed[name]:raise ValueError('Original sealed shape/cost evidence changed')
        if (identity['checkpoint_sha256']!=BODY_SHA or identity['metadata']!=meta or identity['gallery_count']!=1600
                or identity['explicit_warmup']!=0 or identity['repeat']!=repeat or identity['mps_percent']!=100
                or summary['status']!='complete' or summary['valid_samples']!=200):raise ValueError('Original qualified .8225 scope differs')
        ledger=read(folder/'stage_ledger.json')
        for path,expected in GRAPH.items():
            records=[r for r in ledger if r['source_file'].replace('\\','/').endswith('/'+path)]
            blob=subprocess.check_output(['git','show',identity['source_commit']+':'+path],cwd=REPO)
            if not records or any(r['source_sha256']!=expected for r in records) or hashlib.sha256(blob).hexdigest()!=expected:
                raise ValueError('Original graph Git/source binding differs')
        with (folder/'samples.csv').open(encoding='utf-8-sig',newline='') as f:samples=list(csv.DictReader(f))
        counts={i:Counter() for i in range(200)}
        with (folder/'stages.csv').open(encoding='utf-8-sig',newline='') as f:
            for r in csv.DictReader(f):counts[int(r['sample_index'])][r['stage']]+=1
        if len(samples)!=200:raise ValueError('All200 original calls required')
        for i,r in enumerate(samples):
            if (ast.literal_eval(r['input_ids_shape'])!=[1,length] or ast.literal_eval(r['pixel_values_shape'])!=[196,1536]
                    or ast.literal_eval(r['image_grid_thw'])!=[[1,14,14]] or int(r['valid_input_tokens'])!=length
                    or counts[i]['bridge.native_merger']!=1 or counts[i]['vision.output_adapter']!=1
                    or counts[i]['language.output_adapter']!=0):raise ValueError('Actual compact input/merger/output calls differ')
            for mode in ('vision','language'):
                for name,n in [('encode_linear_norm_softplus',2),('ccd_output_linear',2),('parallel_residual1',1),('parallel_residual2',1),('router_energy_top2',1),('router_fft',1),('expert_global_fft',2)]:
                    if counts[i][mode+'.'+name]!=n:raise ValueError('Original six-pass/adapter/residual occurrences differ')
        signatures.append([(r['sample_id'],r['input_ids_shape'],r['pixel_values_shape']) for r in samples])
        refs.append(dict(run=folder.relative_to(REPO).as_posix(),summary_sha256=sealed['summary.json'],samples_sha256=sealed['samples.csv'],
            stages_sha256=sealed['stages.csv'],identity_sha256=sealed['identity.json'],seconds200=(summary['charged_electronic_ms_total']+summary['optical_ms_total'])/1000,
            joules200=summary['phase_accounted_energy_j_total']))
    if signatures[1:]!=[signatures[0],signatures[0]]:raise ValueError('Original three shape workloads differ')
    op=terms['actual_NN_MAC']*1200;seconds=sum(r['seconds200'] for r in refs);energy=sum(r['joules200'] for r in refs)
    return dict(status='ORIGINAL_T07_COMPACT8225_GRAPH_AND_SEALED600_SHAPES_NN_MAC_DERIVED',calls=600,
        actual_NN_MAC600=600*terms['actual_NN_MAC'],charged_serial_NN_MAC600=600*terms['charged_serial_NN_MAC'],
        parallel_residual_NN_MAC600=600*terms['parallel_residual_NN_MAC'],TOP_per_input=op/600/1e12,TOPS=op/1e12/seconds,TOPS_per_W=op/1e12/energy,
        per_query_MAC=terms,other_executed_digital_dense_MAC600=600*sum(terms['other_executed_digital_dense_MAC'].values()),
        actual_fixed_sequence_length=length,grid=[1,14,14],cost_references=refs,checkpoint_sha256=BODY_SHA,metadata_sha256=META_SHA,
        original_graph_sources=GRAPH,model_or_GPU_forwards=0,new_time_or_power=False,complete_scalar_OP_claim=False,
        physical_optical_dense_equivalent_OP=None,FFT_not_physical_optical_OP=True,
        convention='1 real dense MAC=2OP; Linear/Conv/search NN subtotal; full padded depthwise kernel; router digital mask contraction separately supplied',
        non_MAC=['position interpolation','embedding lookup','normalization','pooling','activation/bias','routing topk/softmax/scalar diagnostics','fusion','phase/complex propagation','layout','sorting'],
        no_RoPE_or_Language_transformer_or_LM_head=True,one_main_merger=True,no_late_RGB_bypass=True,
        source_sha256=sha(__file__),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve prior receipts')
    value=derive()
    with a.output.open('x') as f:json.dump(value,f,indent=2)
    print(json.dumps({k:value[k] for k in ('TOP_per_input','TOPS','TOPS_per_W','model_or_GPU_forwards')}))
