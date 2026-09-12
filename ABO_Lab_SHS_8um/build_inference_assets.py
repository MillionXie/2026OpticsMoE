"""Package fixed ABO inference assets; copy native tensor bytes without dtype conversion."""
import argparse,hashlib,json,struct,zipfile
from pathlib import Path

CHECKPOINT_SHA='a2aa9a93028410dfb780df7d320a9be65d07b6e7ec87ede08b53c4bddcbaf7af'

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();source=a.source;model=source/'original_inference/models/Qwen3-VL-Embedding-2B'
    ck=source/'original_inference/assets/best_checkpoint.pt'
    if digest(ck)!=CHECKPOINT_SHA:raise ValueError('Wrong reference checkpoint')
    weights=model/'model.safetensors';manifest={'files':{},'checkpoint_sha256':CHECKPOINT_SHA,'source_model_sha256':digest(weights),'native_tensor_copy':'exact bytes, no dtype conversion; no native transformer/lm_head needed by student'}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.out,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        def add(src,rel):
            z.write(src,rel);manifest['files'][rel]=digest(src)
        for folder,prefix in ((source/'original_a100/backend','runtime/backend'),(source/'original_a100/abo_dual','runtime/abo_dual'),(source/'original_inference/assets','assets')):
            for f in sorted(folder.rglob('*')):
                if f.is_file() and '__pycache__' not in f.parts:add(f,prefix+'/'+f.relative_to(folder).as_posix())
        for f in model.iterdir():
            if f.is_file() and f.suffix in ('.json','.txt','.jinja'):add(f,'models/Qwen3-VL-Embedding-2B/'+f.name)
        # Original native transformer blocks are replaced by this student's graph.
        # Read only their complement, retaining all embedding/position/merger/norm tensors.
        with weights.open('rb') as f:
            length=struct.unpack('<Q',f.read(8))[0];old=json.loads(f.read(length));base=8+length
            keys=sorted(k for k in old if k!='__metadata__' and '.layers.' not in k and '.blocks.' not in k and k!='lm_head.weight')
            header={};offset=0
            for k in keys:
                item=old[k];n=item['data_offsets'][1]-item['data_offsets'][0]
                header[k]=dict(dtype=item['dtype'],shape=item['shape'],data_offsets=[offset,offset+n]);offset+=n
            header['__metadata__']={'format':'pt'}
            raw=json.dumps(header,separators=(',',':')).encode();raw+=b' '*((-len(raw))%8)
            name='models/Qwen3-VL-Embedding-2B/native_student.safetensors';h=hashlib.sha256()
            with z.open(name,'w',force_zip64=True) as target:
                b=struct.pack('<Q',len(raw))+raw;target.write(b);h.update(b)
                for k in keys:
                    start,end=old[k]['data_offsets'];f.seek(base+start);remaining=end-start
                    while remaining:
                        b=f.read(min(1024**2,remaining))
                        if not b:raise EOFError(k)
                        target.write(b);h.update(b);remaining-=len(b)
            manifest['files'][name]=h.hexdigest();manifest['native_keys']=keys;manifest['native_payload_bytes']=offset
        z.writestr('INFERENCE_ASSET_MANIFEST.json',json.dumps(manifest,indent=2))
    print(json.dumps({'zip':str(a.out),'bytes':a.out.stat().st_size,'sha256':digest(a.out),'native_payload_bytes':offset}),flush=True)

if __name__=='__main__':main()
