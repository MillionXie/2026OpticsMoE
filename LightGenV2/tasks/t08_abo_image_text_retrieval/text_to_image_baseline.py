"""Frozen text-only query -> ABO easy100 TEST image gallery. No fitting."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch
from torch.nn import functional as F

QUERY='Retrieve product images that match the following product description.'
DOCUMENT="Represent the user's input."


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def metrics(scores, query_labels, gallery_labels):
    order=scores.argsort(dim=1,descending=True,stable=True)
    relevant=gallery_labels[order].eq(query_labels[:,None])
    total=relevant.sum(1)
    if not bool((total>0).all()):raise ValueError('Query without relevant gallery image')
    rank=torch.arange(1,scores.shape[1]+1,dtype=torch.float64)[None]
    precision=relevant.cumsum(1)/rank
    first=torch.where(relevant,rank,torch.inf).amin(1)
    report={'query_count':len(scores),'gallery_count':scores.shape[1],
            'mrr':float((1/first).mean()),'map':float(((precision*relevant).sum(1)/total).mean())}
    for k in (1,5,10):
        report[f'hit_at_{k}']=float(relevant[:,:k].any(1).double().mean())
        report[f'recall_at_{k}']=float((relevant[:,:k].sum(1)/total.double()).mean())
    return report,order,first


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','data','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    from PIL import Image,ImageOps
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    def read(name):
        with (a.data/name).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
    titles,train,test=read('titles.csv'),read('train.csv'),read('test.csv')
    assert len(titles)==100 and len(test)==2400 and len(train)==4800
    products={r['product_id']:int(r['label']) for r in titles}
    assert len(products)==100 and len({r['title'] for r in titles})==100
    assert not {r['sample_id'] for r in train}&{r['sample_id'] for r in test}
    for r in test:
        assert products[r['product_id']]==int(r['label'])
        path=(a.data/r['image_path']).resolve()
        assert path.is_relative_to(a.data.resolve()) and path.is_file()
    assert all(sum(r['product_id']==pid for r in test)==24 for pid in products)
    a.output.mkdir(parents=True)
    start=time.time()
    report=dict(status='running',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        command=sys.argv,model=str(a.model.resolve()),model_config_sha256=digest(a.model/'config.json'),
        model_weight_sha256={f.name:digest(f) for f in sorted(a.model.glob('*.safetensors'))},
        data_sha256={n:digest(a.data/n) for n in ('titles.csv','train.csv','test.csv')},
        gallery_image_sha256={r['sample_id']:digest(a.data/r['image_path']) for r in test},
        query_instruction=QUERY,document_instruction=DOCUMENT,torch=torch.__version__,python=sys.version,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),gpu=torch.cuda.get_device_name(0),
        training=False,query_modality='text only; no image or product_id in model input',
        protocol='100 official titles ->2400 heldout images;24 exact-SKU positives/title;closed catalog',
        note='Single fixed prompt, no score-driven subset or prompt selection. No timing/power benchmark.')
    def save():
        (a.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    save();model=None
    try:
        processor=AutoProcessor.from_pretrained(str(a.model),local_files_only=True,min_pixels=50176,max_pixels=50176)
        model=Qwen3VLForConditionalGeneration.from_pretrained(str(a.model),local_files_only=True,
            dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval().requires_grad_(False)
        report['trainable_parameters']=sum(p.numel() for p in model.parameters() if p.requires_grad)
        def encode(text=None,image=None):
            instruction=QUERY if text is not None else DOCUMENT
            content=[{'type':'text','text':text}] if text is not None else [{'type':'image','image':image}]
            prompt=processor.apply_chat_template([{'role':'system','content':[{'type':'text','text':instruction}]},
                {'role':'user','content':content}],tokenize=False,add_generation_prompt=True)
            kw=dict(text=[prompt],padding=True,return_tensors='pt')
            if image is not None:kw['images']=[image]
            inp=processor(**kw)
            inp={k:v.to('cuda') for k,v in inp.items() if k in ('input_ids','attention_mask','pixel_values','image_grid_thw')}
            hidden=model.model(**inp,use_cache=False,return_dict=True).last_hidden_state
            pos=torch.arange(inp['attention_mask'].shape[1],device='cuda')[None].masked_fill(~inp['attention_mask'].bool(),-1).amax(1)
            return hidden[0,pos[0]].float().cpu()
        with torch.inference_mode():
            text_vectors=torch.stack([encode(text=r['title']) for r in titles])
            galleries={}
            for mode in ('dynamic','contain_white'):
                vectors=[]
                for i,r in enumerate(test):
                    with Image.open(a.data/r['image_path']) as f:image=ImageOps.exif_transpose(f).convert('RGB')
                    if mode=='contain_white':
                        small=ImageOps.contain(image,(224,224),method=Image.Resampling.BICUBIC)
                        image=Image.new('RGB',(224,224),'white');image.paste(small,((224-small.width)//2,(224-small.height)//2))
                    vectors.append(encode(image=image))
                    if (i+1)%100==0:print(mode,i+1,len(test),flush=True)
                galleries[mode]=torch.stack(vectors)
        qlabels=torch.tensor([int(r['label']) for r in titles]);glabels=torch.tensor([int(r['label']) for r in test])
        report['metrics']={}
        for mode,gallery in galleries.items():
            for dim in (64,2048):
                name=f'{mode}_{dim}'
                scores=F.normalize(text_vectors[:,:dim],dim=1)@F.normalize(gallery[:,:dim],dim=1).T
                result,order,first=metrics(scores,qlabels,glabels);report['metrics'][name]=result
                predictions=[dict(product_id=r['product_id'],title=r['title'],first_positive_rank=int(first[i]),
                    top10_sample_ids=[test[j]['sample_id'] for j in order[i,:10].tolist()]) for i,r in enumerate(titles)]
                (a.output/f'{name}_predictions.json').write_text(json.dumps(predictions,ensure_ascii=False,indent=2),encoding='utf-8')
        torch.save(dict(text_vectors=text_vectors,galleries=galleries,query_labels=qlabels,gallery_labels=glabels,
            query_product_ids=[r['product_id'] for r in titles],gallery_ids=[r['sample_id'] for r in test]),a.output/'embeddings.pt')
        report.update(status='complete',elapsed_seconds=time.time()-start);save();print(json.dumps(report['metrics'],indent=2))
    except BaseException as exc:
        report.update(status='failed_or_interrupted',error=repr(exc));save();raise
    finally:
        del model
        torch.cuda.empty_cache()


if __name__=='__main__':main()
