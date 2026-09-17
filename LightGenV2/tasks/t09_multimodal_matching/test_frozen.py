"""Evaluate preselected validation-best weights on reserved image identities once."""
import argparse
import io
import json
import random
import subprocess
import sys
import platform
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from .prepare import RemoteZip, URL, COLORS, SHAPES, TEMPLATES, tokens, save, digest, member_from_block
from .model import OpticalOEO, TextEncoder
from .run import evaluate, state_sha
from .vision import frozen_features


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    selection={}
    for mode,tag in [('fixed','fixed'),('fixed_dense','dense'),('learned','learned')]:
        run=a.runs/f'clevr_visual_{tag}_s17_v1'
        for arch in ['moe','d2nn']:
            path=run/mode/arch/'best_checkpoint.pt'
            recorded=json.loads((path.parent/'result.json').read_text())
            assert digest(path.read_bytes())==recorded['best_checkpoint_sha256']
            selection[mode+'/'+arch]=dict(checkpoint=str(path),sha256=recorded['best_checkpoint_sha256'],epoch=recorded['epoch'])
    save(a.out/'locked_selection.json',selection)
    save(a.out/'metadata.json',dict(command=sys.argv,python=platform.python_version(),torch=torch.__version__,
         commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
         source_sha256=digest(Path(__file__).read_bytes()),
         visual_checkpoint_sha256=digest((a.runs/'clevr_visual_aux_s17_v1/best_checkpoint.pt').read_bytes()),
         selection='Frozen validation-NLL-best checkpoints; no test-based selection'))
    save(a.out/'status.json',dict(status='preparing_test',selection='validation NLL; locked before test download',commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    ids=json.loads((a.data/'test_reserved_ids.json').read_text())
    vocab=json.loads((a.data/'vocab.json').read_text())
    used={r['image_id'] for split in ['train','val'] for r in json.loads((a.data/f'{split}_questions.json').read_text())}
    archive=RemoteZip(URL)
    with zipfile.ZipFile(archive) as z:
        names={x.filename:x for x in z.infolist()}
        raw=z.read('CLEVR_v1.0/scenes/CLEVR_train_scenes.json')
    assert digest(raw)==json.loads((a.data/'manifest.json').read_text())['official_scene_sha256']
    scenes={int(x['image_index']):x for x in json.loads(raw)['scenes']}
    cache=a.out/'images';cache.mkdir()
    def fetch(idx):
        name=scenes[idx]['image_filename'];assert name not in used
        info=names['CLEVR_v1.0/images/train/'+name]
        block=archive.block(info.header_offset,min(archive.size,info.header_offset+info.compress_size+4096))
        payload=member_from_block(info,block,info.header_offset)
        (cache/name).write_bytes(payload)
        return np.asarray(Image.open(io.BytesIO(payload)).convert('RGB').resize((64,64),Image.Resampling.LANCZOS))
    images=[]
    with ThreadPoolExecutor(6) as pool:
        for i,img in enumerate(pool.map(fetch,ids)):
            images.append(img)
            if i%25==0:print('test images',i+1,flush=True)
    rows=[];combinations={(c,s) for c in COLORS for s in SHAPES}
    for local,idx in enumerate(ids):
        scene=scenes[idx];present={(o['color'],o['shape']) for o in scene['objects']}
        rng=random.Random(17*100000+idx);positive=sorted(present)
        pos=rng.sample(positive,min(3,len(positive)))
        while len(pos)<3:pos.append(rng.choice(positive))
        neg=rng.sample(sorted(combinations-present),3)
        for pair,(yes,no) in enumerate(zip(pos,neg)):
            for label,(color,shape) in [(1,yes),(0,no)]:
                rows.append(dict(image_local=local,image_id=scene['image_filename'],label=label,question=TEMPLATES[(idx+pair)%len(TEMPLATES)].format(color=color,shape=shape)))
    save(a.out/'test_questions.json',rows)
    save(a.out/'data_manifest.json',dict(parent_manifest_sha256=digest((a.data/'manifest.json').read_bytes()),images={f.name:digest(f.read_bytes()) for f in cache.iterdir()},questions_sha256=digest((a.out/'test_questions.json').read_bytes())))
    word_ids=torch.zeros(len(rows),32,dtype=torch.long,device='cuda')
    for i,r in enumerate(rows):
        words=tokens(r['question']);assert len(words)<=32 and all(w in vocab for w in words)
        word_ids[i,:len(words)]=torch.tensor([vocab[w] for w in words],device='cuda')
    features=frozen_features(a.runs/'clevr_visual_aux_s17_v1/best_checkpoint.pt',torch.tensor(np.stack(images),device='cuda'))
    data=dict(images=features,ids=word_ids,index=torch.tensor([r['image_local'] for r in rows],device='cuda'),labels=torch.tensor([r['label'] for r in rows],device='cuda'),rows=rows)
    results={}
    for key,record in selection.items():
        mode,arch=key.split('/');path=Path(record['checkpoint'])
        front=TextEncoder(len(vocab),mode).cuda()
        front.load_state_dict(torch.load(path.parent.parent/'frontend.pt',weights_only=False)['state']);front.eval()
        checkpoint=torch.load(path,weights_only=False)
        assert state_sha(front)==checkpoint['frontend_sha256']
        model=OpticalOEO(arch,17).cuda();model.load_state_dict(checkpoint['model']);model.eval()
        score,pred=evaluate(model,front,data,32)
        assert abs(float((pred.argmax(1)==np.array([r['label'] for r in rows])).mean())-score['accuracy'])<1e-6
        np.savez_compressed(a.out/(mode+'_'+arch+'_predictions.npz'),probabilities=pred,labels=np.array([r['label'] for r in rows]))
        results[key]=dict(**record,test=score);save(a.out/'results.json',results)
        print(key,score['accuracy'],flush=True)
        del model,front,checkpoint;torch.cuda.empty_cache()
    save(a.out/'status.json',dict(status='complete',test_images=len(images),test_questions=len(rows),test_used_for_selection=False))


if __name__=='__main__':main()
