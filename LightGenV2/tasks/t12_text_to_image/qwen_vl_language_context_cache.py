"""Record native last-four Qwen language-layer call metadata for adaptation.

The original 24-layer prefix is frozen. Its already verified language23 cache
is reused; rotary/mask arguments come from actual unmodified Qwen forwards,
never guessed from a text-only token index. No generation or optimization.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch

from .qwen_vl_hybrid_run import load_data, checked_features, load_qwen, raw_hidden, completed_teacher
from .qwen_vl_turbo_cache import save_with_digest
from .qwen_vl_turbo_data import sha256, write_json


def tree_cpu(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, tuple): return tuple(tree_cpu(v) for v in value)
    if value is None or isinstance(value, (int, bool)): return value
    raise ValueError('Unsupported native decoder argument type '+str(type(value)))


def tree_equal(a,b):
    if type(a) is not type(b): return False
    if isinstance(a,torch.Tensor): return a.shape==b.shape and torch.equal(a,b)
    if isinstance(a,tuple): return len(a)==len(b) and all(tree_equal(x,y) for x,y in zip(a,b))
    return a==b


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','features','parent','qwen','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--gpu-uuid',required=True)
    a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=a.gpu_uuid or a.output.exists():
        raise ValueError('Authorized GPU and an immutable context cache required')
    completed_teacher(a.parent)
    a.output.mkdir(parents=True)
    write_json(a.output/'execution.json',{'argv':sys.argv,'pid':os.getpid(),
        'source_commit':os.environ.get('T12_SOURCE_COMMIT'),'source_sha256':sha256(Path(__file__)),
        'started_unix':time.time(),'scope':'Frozen public frontend call-metadata capture; no learning or generation'})
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.8,0);device=torch.device('cuda:0')
    payload=load_data(a.data);features=checked_features(a.features,payload)
    state=torch.load(a.parent/'best_checkpoint.pt',map_location='cpu',weights_only=False)
    if state['config'].get('semantic_prefix',0): raise ValueError('Original fused readout parent required')
    groups={}
    for i,h in enumerate(payload['hidden']):groups.setdefault(len(h),[]).append(i)
    qwen,processor=load_qwen(a.qwen,device)
    captured={}
    allowed={'attention_mask','position_ids','past_key_values','cache_position','position_embeddings','use_cache'}
    def hook(module,args,kwargs):
        if set(kwargs)-allowed: raise ValueError('Unreviewed native decoder kwargs '+str(set(kwargs)-allowed))
        if kwargs.get('past_key_values') is not None or kwargs.get('use_cache',False): raise ValueError('No autoregressive KV cache')
        captured['hidden']=args[0].detach().cpu()
        captured['kwargs']={k:tree_cpu(v) for k,v in kwargs.items()}
    handle=qwen.model.language_model.layers[24].register_forward_pre_hook(hook,with_kwargs=True)
    contexts={};checks=[]
    try:
        with torch.inference_mode():
            for length,ids in sorted(groups.items()):
                # Same image grid/layout; verify two distinct inputs of every
                # available sequence length yield identical positional args.
                selected=[ids[0]]
                if len(ids)>1: selected.append(ids[-1])
                for i in selected:
                    captured.clear()
                    hidden=raw_hidden(qwen,processor,payload,i,state['visual_layout'],device).cpu()
                    prefix=features[i]['language_23']
                    if not torch.equal(captured['hidden'],prefix): raise ValueError('Frozen prefix differs from verified terminal cache')
                    difference=float((hidden.float()-payload['hidden'][i].float()).abs().max())
                    if difference>1.:raise ValueError('Original full frontend differs')
                    if length in contexts:
                        if contexts[length].keys()!=captured['kwargs'].keys() or any(not tree_equal(contexts[length][k],v) for k,v in captured['kwargs'].items()):
                            raise ValueError('Native call metadata is not determined by this verified layout and sequence length')
                    else:contexts[length]=captured['kwargs']
                    checks.append({'sample_id':payload['rows'][i]['sample_id'],'sequence_length':length,'prefix_exact':True,'full_hidden_max_abs':difference})
                write_json(a.output/'progress.json',{'verified_lengths':len(contexts),'total_lengths':len(groups)})
    finally:handle.remove()
    source_sha=json.loads((a.features/'report.json').read_text())['sha256']
    digest=save_with_digest({'schema':'native_qwen_last4_context_v1','contexts':contexts,
        'prefix_feature_sha256':source_sha,'source_cache_sha256':json.loads((a.data/'cache_complete.json').read_text())['sha256'],
        'native_layer_indices':[24,25,26,27],'checks':checks},a.output/'native_contexts.pt')
    write_json(a.output/'report.json',{'complete':True,'sha256':digest,'file':'native_contexts.pt',
        'count':len(payload['rows']),'sequence_lengths':sorted(contexts),'raw_checks':checks,
        'independent_TEST_opened':False,'prefix_feature_sha256':source_sha,
        'note':'Position/mask metadata only. No LoRA weights trained yet; no claim of generation improvement.'})


if __name__=='__main__':main()
