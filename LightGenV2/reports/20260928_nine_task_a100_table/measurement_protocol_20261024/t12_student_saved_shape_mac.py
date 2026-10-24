"""Original5b4f saved-shape NN MAC rules; no Torch/model/forward import.

The passed CPU proof contains BOTH gold and observed forwards. Split and compare
them, never count both as one query. Two CPU examples validate the formula only;
formal first200 totals require all three later qualified cost shape streams.
"""
import argparse
from collections import Counter
import copy
import csv
import hashlib
import json
from math import prod
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
PROOF=REPO/'LightGenV2/tasks/t12_text_to_image/runs/simulation/a100_live_frontend_contract_20261025_06/original_live_CPU_contract/contract.json'
PROOF_SHA='a37514afe55c0774d411e672a0dd66641c30bbd6ad49fbf8cf2728ab47ccd1c0'
BODY_SHA='5b4f9a37f19ce95cf23e4b874badc8e88955559553d47929fb80cd3cbf527cac'
GRAPH_NAMES=[
 'LightGenV2/tasks/t12_text_to_image/'+n+'.py'
 for n in ('qwen_mini_small','audited_unified','sealed_editor','small_fullframe','modeling')]
GRAPH_NAMES += [
 'experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust/optical_blocks.py',
 'experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval/router.py']
FIELDS=('module','kind','input_shape','output_shape','weight_shape','groups')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def clean(record):
    r={k:copy.deepcopy(record[k]) for k in FIELDS}
    r['module']=r['module'].removeprefix('model.')
    return r


def split_cpu_forwards(rows):
    if len(rows)!=112:raise ValueError('Original gold+observed112 module calls required')
    gold,observed=([clean(r) for r in rows[:56]],[clean(r) for r in rows[56:]])
    if gold!=observed:raise ValueError('Original and observed module shapes differ')
    counts=Counter(r['module'] for r in gold)
    repeated={n for n,c in counts.items() if c==2}
    if (len(counts)!=54 or repeated!={'text.optical.core.input_adapter','editor.bottleneck.optical.core.input_adapter'}
            or any(c not in (1,2) for c in counts.values())):
        raise ValueError('Original56-call graph/adapter multiplicities differ')
    return gold


def expected_shapes(template,length):
    if not isinstance(length,int) or not 1<=length<=64:raise ValueError('Original valid prompt length1..64 required')
    result=copy.deepcopy(template)
    for r in result:
        name=r['module']
        if name.startswith('text.') and name!='text.condition_projection':
            axis=1 if name=='text.input_projection' or name.startswith('text.layers.') else 0
            r['input_shape'][axis]=length;r['output_shape'][axis]=length
    return result


def module_mac(r):
    x,w,y=r['input_shape'],r['weight_shape'],r['output_shape']
    if any(not isinstance(n,int) or n<=0 for s in (x,w,y) for n in s):raise ValueError('Invalid original shape')
    if r['kind']=='Linear':
        if len(w)!=2 or x[-1]!=w[1] or y!=x[:-1]+[w[0]]:raise ValueError('Linear dimensions differ')
        return prod(x[:-1])*w[0]*w[1]
    if r['kind']=='Conv2d':
        g=r['groups']
        if (len(x)!=4 or len(w)!=4 or len(y)!=4 or not isinstance(g,int) or g<1
                or x[0]!=y[0] or x[1]!=w[1]*g or y[1]!=w[0] or y[1]%g):
            raise ValueError('Original Conv2d dimensions/groups differ')
        return y[0]*prod(y[2:])*w[0]*prod(w[1:])
    raise ValueError('Unknown original NN module; do not silently omit')


def request_terms(records,template,length):
    actual=[clean(r) for r in records]
    if actual!=expected_shapes(template,length):raise ValueError('Actual original full module sequence/shape differs')
    serial=Counter();parallel=Counter()
    for r in actual:
        name=r['module']
        target=parallel if name.startswith(('text.layers.','editor.bottleneck.electronic1.','editor.bottleneck.electronic2.')) else serial
        target[name]+=module_mac(r)
    # Original QwenMiniBlock: full MHA, Q/K/V/O width512, two layers.
    # Heads cancel algebraically: heads*(512/heads)=512. Dense causal/padding
    # convention, not a claim that triangular hardware kernels execute zeros.
    for layer in (0,1):
        for name in ('q_proj','k_proj','v_proj','o_proj'):
            row=next(r for r in actual if r['module']==f'text.layers.{layer}.{name}')
            if row['weight_shape']!=[512,512]:raise ValueError('Original full-MHA width differs')
        parallel[f'text.layers.{layer}.functional_QK_AV']=2*length*length*512
    return dict(actual_NN_MAC=sum(serial.values())+sum(parallel.values()),
        charged_serial_NN_MAC=sum(serial.values()),parallel_residual_NN_MAC=sum(parallel.values()),
        serial_NN_terms=dict(serial),parallel_NN_terms=dict(parallel),
        other_executed_digital_dense_MAC={'two_router_dense_mask_contractions':2*4*478*478})


def proof_reference():
    if sha(PROOF)!=PROOF_SHA:raise ValueError('Exact passed original CPU proof required')
    p=read(PROOF)
    if p['status']!='passed' or not p['no_gpu'] or p['body_sha256']!=BODY_SHA:raise ValueError('Original proof identity differs')
    graph={name:p['sources'][name] for name in GRAPH_NAMES}
    for name,digest in graph.items():
        blob=subprocess.check_output(['git','show',p['source_commit']+':'+name],cwd=REPO)
        if hashlib.sha256(blob).hexdigest()!=digest:raise ValueError('Original graph Git bytes differ')
    by_index={}
    for check in p['checks']:
        if not all(check[k] for k in ('bitwise_equal','reference_bitwise_equal','native_valid_embeddings_bitwise_equal')):
            raise ValueError('Original full numerical proof required')
        index=check['test_index'];length=sum(check['attention_mask'][0])
        rows=[r for r in p['actual_NN_shapes'] if r['test_index']==index]
        one=split_cpu_forwards(rows)
        by_index[index]=(one,length)
    if set(by_index)!={200,2303}:raise ValueError('Original two-index CPU proof differs')
    template=by_index[200][0]
    samples=[dict(test_index=i,valid_tokens=n,**request_terms(rows,template,n)) for i,(rows,n) in by_index.items()]
    return p,template,samples,graph


def derive(cost_roots=()):
    p,template,samples,graph=proof_reference()
    value=dict(status='T12_ORIGINAL_TWO_CPU_INDEX_SHAPE_FORMULA_CHECKED_NOT_FORMAL200_COST',
        CPU_reference_samples=samples,original_NN_module_calls_per_query=56,
        gold_and_observed_not_double_counted=True,CPU_contract_sha256=PROOF_SHA,checkpoint_sha256=BODY_SHA,
        original_graph_sources=graph,formal_first200_costs_complete=False,formal600_NN_MAC=None,
        TOPS=None,TOPS_per_W=None,model_or_GPU_forwards=0,new_time_or_power=False,
        complete_scalar_OP_claim=False,physical_optical_dense_equivalent_OP=None,
        convention='1 real MAC=2OP; dense padded Conv and rectangular causal attention; algorithmic not hardware instruction count',
        non_MAC=['embedding lookup','RoPE elementwise position construction','RMS/group normalization','softmax',
                 'bias/activation','pooling/interpolation','routing topk/log scores','fusion','clamp/noise/layout/transfer'],
        FFT_not_physical_optical_OP=True,router_digital_dense_mask_MAC_separate=True,
        source_sha256=sha(__file__),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
    if cost_roots:
        if len(cost_roots)!=3:raise ValueError('Exactly three original fresh200 cost roots required')
        from audit_t12_cost import audit
        requests=[];refs=[];signatures=[]
        for repeat,folder in enumerate(map(Path,cost_roots),1):
            independent=audit(folder);identity=read(folder/'identity.json');summary=read(folder/'summary.json')
            if independent['status']!='passed' or identity['system']!='ours' or identity['repeat']!=repeat or identity['cpu_contract_sha256']!=PROOF_SHA:
                raise ValueError('Three qualified original Ours rounds/proof required')
            with (folder/'calls.csv').open(encoding='utf-8-sig',newline='') as f:calls=list(csv.DictReader(f))
            rows=read(folder/'nn_module_shapes.json');signatures.append([(r['sample_id'],r['input_ids'],r['attention_mask']) for r in calls])
            for i,call in enumerate(calls):
                mask=json.loads(call['attention_mask'].replace('True','true').replace('False','false'))
                requests.append(request_terms([r for r in rows if r['index']==i],template,sum(mask[0])))
            if len(calls)!=200 or len(rows)!=200*56:raise ValueError('All200 actual full original shape streams required')
            refs.append(dict(run=folder.relative_to(REPO).as_posix(),summary_sha256=sha(folder/'summary.json'),
                shapes_sha256=sha(folder/'nn_module_shapes.json'),calls_sha256=sha(folder/'calls.csv'),
                seconds200=summary['seconds200'],joules200=summary['joules200']))
        if signatures[1:]!=[signatures[0],signatures[0]]:raise ValueError('Three original input workloads differ')
        actual=sum(r['actual_NN_MAC'] for r in requests);op=2*actual
        value.update(status='T12_ORIGINAL_QUALIFIED600_SHAPES_NN_MAC_DERIVED',formal_first200_costs_complete=True,
            formal600_NN_MAC=actual,charged_serial_NN_MAC600=sum(r['charged_serial_NN_MAC'] for r in requests),
            parallel_residual_NN_MAC600=sum(r['parallel_residual_NN_MAC'] for r in requests),
            TOP_per_input=op/600/1e12,TOPS=op/1e12/sum(r['seconds200'] for r in refs),
            TOPS_per_W=op/1e12/sum(r['joules200'] for r in refs),cost_references=refs,requests=requests)
    return value


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--output',type=Path,required=True)
    a.add_argument('--cost-roots',type=Path,nargs=3);args=a.parse_args()
    if args.output.exists():raise FileExistsError('Preserve existing receipt')
    value=derive(args.cost_roots or ())
    with args.output.open('x',encoding='utf8') as f:json.dump(value,f,indent=2)
    print(json.dumps({k:value[k] for k in ('status','formal_first200_costs_complete','model_or_GPU_forwards')}))
