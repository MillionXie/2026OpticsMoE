"""CPU-only independent dense-MAC arithmetic audit; never imports Torch."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from math import prod
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3];sys.path.insert(0,str(HERE))
from abo_query_contract import MODEL_CONFIG_SHA


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def independent_record_mac(record):
    args=record['inputs'];op=record['operator'].split('.')[1]
    def shape(index):return tuple(args[index]['shape'])
    if op in ('mm','bmm','matmul','mv','dot','addmm','baddbmm','addmv'):
        a,b=(shape(1),shape(2)) if op in ('addmm','baddbmm','addmv') else (shape(0),shape(1))
        if len(a)==1:a=(1,*a)
        if len(b)==1:b=(*b,1)
        assert a[-1]==b[-2]
        la,lb=a[:-2],b[:-2];n=max(len(la),len(lb));la=(1,)*(n-len(la))+la;lb=(1,)*(n-len(lb))+lb
        assert all(x==y or x==1 or y==1 for x,y in zip(la,lb))
        return prod([y if x==1 else x for x,y in zip(la,lb)])*a[-2]*a[-1]*b[-1]
    if 'scaled_dot_product' in op:
        q,k,v=shape(0),shape(1),shape(2)
        assert len(q)==len(k)==len(v)==4 and q[0]==k[0]==v[0] and k[-2]==v[-2] and q[-1]==k[-1]
        assert k[1]==v[1] and q[1]%k[1]==0
        return q[0]*q[1]*q[2]*k[2]*(q[3]+v[3])
    if op=='linear':
        x,w=shape(0),shape(1);assert x[-1]==w[1]
        return prod(x[:-1])*w[0]*w[1]
    raise ValueError('Unsupported independent dense-MAC audit rule: '+op)


def audit(root,config_path):
    root=Path(root);data=read(root/'operation_counts.json')
    if sha(config_path)!=MODEL_CONFIG_SHA:raise ValueError('Original native config differs')
    config=read(config_path)['text_config']
    seal=root/'sealed_files_after_exit01.json'
    for item in read(seal):
        path=root/item['file']
        if sha(path)!=item['sha256'] or path.stat().st_size!=item['bytes']:raise ValueError('Sealed OP raw differs')
    for path,digest in data['sources'].items():
        original=subprocess.check_output(['git','show',data['source_commit']+':'+path.replace('\\','/')],cwd=REPO)
        if hashlib.sha256(original).hexdigest()!=digest:raise ValueError('Replay source not at original recorded commit')
    if (data['input_calls']!=200 or data['unique_queries']!=100 or data['gpu_uuid']!='GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55'
            or data['mps_percent']!=100 or data['power_limit_w']!=250 or data['complete_total_scalar_OP_claim']
            or data['physical_optical_OP_included'] or data['numeric_FFT_counted_as_physical_light']):
        raise ValueError('Wrong model workload or OP scope')
    checks=data['raw_reference_bitwise_matches']
    if [r['index'] for r in checks]!=list(range(200)) or not all(r['bitwise_equal'] for r in checks):
        raise ValueError('Missing original measured output match')
    lengths=[r['valid_input_tokens'] for r in checks]
    total=0;by_sample=Counter();unknown=set()
    for row in data['records']:
        if row['sample_index'] not in range(200) or row['charged'] is not True:raise ValueError('Unscoped input record')
        if row['mac'] is not None:
            observed=independent_record_mac(row)
            if observed!=row['mac'] or not isinstance(row['calls'],int) or row['calls']<1:raise ValueError('Actual shape count differs')
            total+=observed*row['calls'];by_sample[row['sample_index']]+=observed*row['calls']
        elif row['kind'].startswith(('UNREGISTERED_','UNSUPPORTED_')):unknown.add(row['operator'])
    if set(by_sample)!=set(range(200)) or sorted(unknown)!=data['unsupported_operators']:
        raise ValueError('Missing sample MAC or hidden unknown operators')
    h=config['hidden_size'];layers=config['num_hidden_layers'];width=config['intermediate_size']
    heads=config['num_attention_heads'];kv=config['num_key_value_heads'];d=config['head_dim']
    main=layers*(sum(lengths)*(h*(heads+2*kv)*d+heads*d*h+3*h*width)
        +2*heads*d*sum(s*s for s in lengths))+200*2400*h
    rotary=3*(d//2)*sum(lengths)
    if (total!=main+rotary or main!=data['independent_QKVO_SwiGLU_attention_search_main_mac']
            or total!=data['actual_executed_real_mac'] or total!=data['charged_serial_real_mac']
            or 2*total!=data['actual_executed_mac_op'] or sum(lengths)!=data['common_input_tokens']):
        raise ValueError('Independent native-network total differs')
    events=read(root/'process_audit.json');worker_ids=set()
    for event in events:
        if len(event['clients'])>1:raise ValueError('Foreign MPS clients in untimed replay')
        worker_ids.update(event['clients'])
        if any(int(p[0]) not in {*event['clients'],*event['servers']} for p in event['processes']):raise ValueError('Foreign GPU process')
    if len(worker_ids)!=1:raise ValueError('Missing sole original worker')
    if any(e['foreign_gpu_processes'] or e['foreign_mps_clients'] for e in read(root/'loading_process_audit.json')):
        raise ValueError('Foreign use during loading')
    cleanup=read(root/'post_worker_mps_release02.json')
    if (cleanup['status']!='released_after_worker_exit_no_gpu_replay' or cleanup['current_gpu_processes']
            or cleanup['private_mps_control_returncode']==0 or cleanup['gpu_forwards']!=0):raise ValueError('Missing resolved private-MPS cleanup')
    return dict(status='MAC_ARITHMETIC_VALIDATED_CLEANUP_FAULT_SEPARATELY_RESOLVED',
        operation_raw_sha256=sha(root/'operation_counts.json'),sealed_manifest_sha256=sha(seal),
        calls=200,unique_titles=100,MAC200=total,MAC_OP200=2*total,
        GMAC_per_query=total/200/1e9,TOP_MAC_convention_per_query=2*total/200/1e12,
        native_network_and_search_main_MAC=main,rotary_position_construction_MAC=rotary,
        non_MAC_and_unsupported_explicit=sorted(unknown),not_a_full_scalar_OP_claim=True,
        no_timing_or_energy_from_replay_used=True,raw_cleanup_fault_preserved=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','config','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();value=audit(a.root,a.config)
    with a.output.open('x',encoding='utf8') as f:json.dump(value,f,indent=2)
    print(json.dumps(value))
