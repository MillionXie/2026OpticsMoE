"""TRAIN-only native-Qwen token-order probe, without image generation or targets.

Equal-length red/blue instructions control image positions within each order.
This measures information flow; nonzero sensitivity is not correctness evidence.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch

from .qwen_vl_hybrid_run import load_data,load_qwen
from .qwen_vl_turbo_data import sha256,write_json
from .qwen_vl_turbo_run import tensor_image


def relative_change(first,second):
    a,b=first.float(),second.float()
    if a.shape!=b.shape:raise ValueError('Matched native state shapes required')
    scale=((a.square().mean()+b.square().mean())*.5).sqrt().clamp_min(1e-8)
    return {'relative_RMS':float((a-b).square().mean().sqrt()/scale),
            'max_abs':float((a-b).abs().max())}


def probe_ids(rows):
    ids=[]
    for category in ('lamp','pillow'):
        seen=set()
        for i,row in enumerate(rows):
            if row['split']!='train' or row['category']!=category:continue
            product=row['source']['product_id']
            if product not in seen:seen.add(product);ids.append(i)
            if len(seen)==2:break
        if len(seen)!=2:raise ValueError('Two TRAIN source products/category required')
    return ids


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','qwen','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--gpu-uuid',required=True);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=a.gpu_uuid or a.output.exists():raise ValueError('Authorized single GPU and new run required')
    payload=load_data(a.data);ids=probe_ids(payload['rows']);a.output.mkdir(parents=True)
    write_json(a.output/'execution.json',{'argv':sys.argv,'pid':os.getpid(),'source_commit':os.environ.get('T12_SOURCE_COMMIT'),
        'source_sha256':sha256(Path(__file__)),'started_unix':time.time(),'TRAIN_only':True})
    torch.set_num_threads(4);device=torch.device('cuda:0');torch.cuda.set_per_process_memory_fraction(.8,0)
    qwen,processor=load_qwen(a.qwen,device)
    image_id=json.loads((a.qwen/'config.json').read_text())['image_token_id'];records=[]
    with torch.no_grad():
        for i in ids:
            row=payload['rows'][i];image=tensor_image(payload['reference'][i]);category=row['category']
            image.save(a.output/(str(i)+'_TRAIN_reference.png'))
            prompts=[f'Replace the {category} with a {color} {category}; keep the background unchanged.' for color in ('red','blue')]
            item={'sample_id':row['sample_id'],'prompts':prompts,'orders':{}}
            for order in ('image_first','text_first'):
                states=[];positions=[];lengths=[]
                for prompt in prompts:
                    content=[{'type':'image','image':image},{'type':'text','text':prompt}]
                    if order=='text_first':content=content[::-1]
                    text=processor.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                    inputs=processor(text=[text],images=[image],return_tensors='pt').to(device)
                    found=torch.where(inputs['input_ids'][0]==image_id)[0]
                    if len(found)!=64 or inputs['image_grid_thw'].tolist()!=[[1,16,16]]:raise ValueError('Native image grid changed')
                    hidden=qwen.model(**inputs,use_cache=False,return_dict=True).last_hidden_state[0]
                    states.append((hidden[found].cpu(),hidden[-1].cpu()));positions.append(found.cpu().tolist());lengths.append(hidden.shape[0])
                if positions[0]!=positions[1] or lengths[0]!=lengths[1]:raise ValueError('Red/blue token lengths or image positions differ; no controlled comparison')
                item['orders'][order]={'sequence_length':lengths[0],'image_positions':positions[0],
                    'visual_states':relative_change(states[0][0],states[1][0]),
                    'last_fused_state':relative_change(states[0][1],states[1][1])}
            records.append(item)
    write_json(a.output/'report.json',{'complete':True,'TRAIN_only':True,'model':'Unmodified public full Qwen24V/28L',
        'source_commit':os.environ.get('T12_SOURCE_COMMIT'),'source_sha256':sha256(Path(__file__)),
        'data_cache_sha256':json.loads((a.data/'cache_complete.json').read_text())['sha256'],
        'independent_TEST_opened':False,'GT_or_mask_input':False,'image_generation_performed':False,
        'records':records,'interpretation':'Controlled information-flow diagnostic only. Nonzero feature changes do not establish successful editing or image quality.'})


if __name__=='__main__':main()
