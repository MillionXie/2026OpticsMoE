"""Test every declared validation-selected model; never select using test scores."""
import argparse
import io
import json
import os
import subprocess
import wave
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from .prepare import save, digest, tokens
from .audio_prepare import logmel, WORDS
from .model import OpticalOEO, TextEncoder
from .run import evaluate, state_sha
from .vision import frozen_features


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--kind',choices=['clevr','audio'],required=True)
    p.add_argument('--audio-runs',nargs='+',help='Completed raw-audio runs; both architectures per run')
    p.add_argument('--clevr-runs',nargs='+',help='Completed custom CLEVR runs; both architectures per run')
    for key in ['data','runs','out','cache']:
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(17)
    profiles=([(tag,arch,'clevr_visual_fixed_'+tag+'_s17_v1')
               for tag in ['long30','long30_phase05'] for arch in ['moe','d2nn']]
              if a.kind=='clevr' else [('audio','moe','audio_matching_moe_s17_v1'),
                                      ('audio','d2nn','audio_matching_d2nn_canonical_s17_v1')])
    selection={}
    if a.audio_runs:
        assert a.kind=='audio'
        profiles=[]
        for run in a.audio_runs:
            config=json.loads((a.runs/run/'metadata.json').read_text())['config']
            assert not config['vision_checkpoint'] and config['mode']=='fixed'
            architectures=['moe','d2nn'] if config['architecture']=='both' else [config['architecture']]
            profiles.extend((run,arch,run) for arch in architectures)
    if a.clevr_runs:
        assert a.kind=='clevr' and not a.audio_runs
        profiles=[]
        for run in a.clevr_runs:
            config=json.loads((a.runs/run/'metadata.json').read_text())['config']
            assert config['vision_checkpoint'] and config['mode']=='fixed'
            architectures=['moe','d2nn'] if config['architecture']=='both' else [config['architecture']]
            profiles.extend((run,arch,run) for arch in architectures)
    for tag,arch,run in profiles:
        path=a.runs/run/'fixed'/arch/'best_checkpoint.pt'
        result=json.loads((path.parent/'result.json').read_text())
        assert digest(path.read_bytes())==result['best_checkpoint_sha256']
        selection[tag+'/'+arch]=dict(checkpoint=str(path),sha256=result['best_checkpoint_sha256'],epoch=result['epoch'])
    save(a.out/'locked_selection.json',selection)
    front_path=a.runs/('clevr_visual_aux_s17_v1' if a.kind=='clevr' else 'audio_frontend_s17_v1')/'best_checkpoint.pt'
    if a.audio_runs:front_path=None
    save(a.out/'metadata.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
         cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),torch=torch.__version__,
         frontend_sha256=digest(front_path.read_bytes()) if front_path else None,data_manifest_sha256=digest((a.data/'manifest.json').read_bytes()),
         selection='All declared validation-NLL-selected checkpoints locked before test evaluation; no test selection'))
    if a.kind=='clevr':
        rows=json.loads((a.cache/'test_questions.json').read_text())
        manifest=json.loads((a.cache/'data_manifest.json').read_text())
        assert digest((a.cache/'test_questions.json').read_bytes())==manifest['questions_sha256']
        assert manifest['parent_manifest_sha256']==digest((a.data/'manifest.json').read_bytes())
        names={r['image_local']:r['image_id'] for r in rows};images=[]
        for i in range(len(names)):
            raw=(a.cache/'images'/names[i]).read_bytes()
            assert digest(raw)==manifest['images'][names[i]]
            images.append(np.asarray(Image.open(io.BytesIO(raw)).convert('RGB').resize((64,64),Image.Resampling.LANCZOS)))
    else:
        records=json.loads((a.data/'test_reserved_ids.json').read_text())
        used=[r for split in ['train','val'] for r in json.loads((a.data/(split+'_audio_records.json')).read_text())]
        assert not ({r['speaker'] for r in used}&{r['speaker'] for r in records})
        assert not ({r['sha256'] for r in used}&{r['sha256'] for r in records})
        assert digest(a.cache.read_bytes())==json.loads((a.data/'manifest.json').read_text())['archive_sha256']
        images=[];rows=[]
        with zipfile.ZipFile(a.cache) as z:
            for i,r in enumerate(records):
                raw=z.read(r['source_member']);assert digest(raw)==r['sha256']
                with wave.open(io.BytesIO(raw),'rb') as w:
                    assert w.getnchannels()==1 and w.getframerate()==16000 and w.getsampwidth()==2 and w.getnframes()<=16000
                    samples=np.frombuffer(w.readframes(w.getnframes()),dtype='<i2').copy()
                images.append(np.repeat(logmel(samples)[:,:,None],3,axis=2))
                target=WORDS.index(r['word']);offset=1+int(r['sha256'][:8],16)%7
                for label,query in [(1,target),(0,(target+offset)%8)]:
                    rows.append(dict(image_local=i,image_id=r['source_member'],speaker=r['speaker'],label=label,
                                     question='does the audio say '+WORDS[query]+' ?'))
    used_ids={r['image_id'] for split in ['train','val'] for r in json.loads((a.data/(split+'_questions.json')).read_text())}
    assert not used_ids & {r['image_id'] for r in rows}
    save(a.out/'test_questions.json',rows)
    vocab=json.loads((a.data/'vocab.json').read_text());ids=torch.zeros(len(rows),32,dtype=torch.long,device='cuda')
    for i,r in enumerate(rows):
        words=tokens(r['question']);assert len(words)<=32 and all(w in vocab for w in words)
        ids[i,:len(words)]=torch.tensor([vocab[w] for w in words],device='cuda')
    features=torch.tensor(np.stack(images),device='cuda')
    if front_path:features=frozen_features(front_path,features)
    save(a.out/'test_manifest.json',dict(images=len(images),questions=len(rows),
         questions_sha256=digest((a.out/'test_questions.json').read_bytes()),features_sha256=digest(features.cpu().numpy().tobytes())))
    labels=np.array([r['label'] for r in rows])
    data=dict(images=features,ids=ids,index=torch.tensor([r['image_local'] for r in rows],device='cuda'),
              labels=torch.tensor(labels,device='cuda'),rows=rows)
    results={}
    for key,record in selection.items():
        arch=key.split('/')[-1];path=Path(record['checkpoint'])
        front=TextEncoder(len(vocab),'fixed').cuda()
        front.load_state_dict(torch.load(path.parent.parent/'frontend.pt',weights_only=False)['state']);front.eval()
        checkpoint=torch.load(path,weights_only=False);assert state_sha(front)==checkpoint['frontend_sha256']
        model=OpticalOEO(arch,17,input_layout=checkpoint.get('input_layout','legacy'),oeo_activation=checkpoint.get('oeo_activation','relu')).cuda();model.load_state_dict(checkpoint['model']);model.eval()
        score,pred=evaluate(model,front,data,32)
        assert abs(float((pred.argmax(1)==labels).mean())-score['accuracy'])<1e-6
        nll=float(-np.log(np.maximum(pred[np.arange(len(labels)),labels],1e-30)).mean())
        assert abs(nll-score['nll'])<1e-5
        np.savez_compressed(a.out/(key.replace('/','_')+'_predictions.npz'),probabilities=pred,labels=labels)
        results[key]=dict(**record,test=score);save(a.out/'results.json',results)
        print(key,score,flush=True)
        del model,front,checkpoint;torch.cuda.empty_cache()
    save(a.out/'status.json',dict(status='complete',test_used_for_selection=False))


if __name__=='__main__':main()
