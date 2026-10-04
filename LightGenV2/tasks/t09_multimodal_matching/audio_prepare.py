"""Official mini Speech Commands -> speaker-disjoint audio/text matching pilot."""
import argparse
import hashlib
import io
import json
import re
import wave
import zipfile
from pathlib import Path
import numpy as np
import requests
import torch
from .prepare import save, digest, tokens

URL='https://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip'
LICENSE_URL='https://research.google/blog/launching-the-speech-commands-dataset/'
WORDS=['down','go','left','no','right','stop','up','yes']


def logmel(samples):
    x=torch.from_numpy(samples.astype(np.float32)/32768.)
    x=torch.nn.functional.pad(x,(0,16000-len(x)))
    power=torch.stft(x,n_fft=512,hop_length=160,win_length=400,
                    window=torch.hann_window(400),center=True,return_complex=True).abs().square()
    hz=torch.linspace(0,8000,257)
    m=torch.linspace(2595*np.log10(1+20/700),2595*np.log10(1+8000/700),66)
    edges=700*(10**(m/2595)-1)
    left=(hz[None]-edges[:-2,None])/(edges[1:-1,None]-edges[:-2,None])
    right=(edges[2:,None]-hz[None])/(edges[2:,None]-edges[1:-1,None])
    bank=torch.minimum(left,right).clamp_min(0)
    db=10*torch.log10((bank@power).clamp_min(1e-10))
    image=((db-db.max()).clamp(-80,0)+80)/80
    assert image.shape==(64,101) and torch.isfinite(image).all()
    return (image.numpy()*255).round().astype(np.uint8)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--archive',type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    archive=a.archive or a.out/'mini_speech_commands.zip'
    if not a.archive:
        with requests.get(URL,stream=True,timeout=120) as r:
            r.raise_for_status()
            with archive.open('wb') as f:
                for chunk in r.iter_content(1024*1024):f.write(chunk)
    split={k:[] for k in ['train','val','test']};seen={};duplicates=0
    with zipfile.ZipFile(archive) as z:
        docs={n:z.read(n).decode('utf8',errors='replace') for n in z.namelist() if n.endswith('README.md') and not n.startswith('__MACOSX')}
        save(a.out/'source_readme.json',docs)
        assert any('speech_commands_v0.01.tar.gz' in t for t in docs.values())
        # The excerpt README refers to the original release rather than restating its license.
        # Publisher page was inspected on 2026-09-17 through web browsing.
        # Do not require the training host to reach research.google on every run.
        save(a.out/'license_evidence.json',dict(source=LICENSE_URL,checked='2026-09-17',
             publisher='Google Research',publication_date='2017-08-24',license='CC BY 4.0',
             chain='Archive README identifies original Speech Commands v0.01; original publisher states CC BY 4.0',
             verification='Publisher page read externally; not downloaded by this training process'))
        for name in sorted(z.namelist()):
            if not name.endswith('.wav') or name.startswith('__MACOSX'):continue
            word=Path(name).parent.name
            if word not in WORDS:continue
            speaker=re.sub(r'_nohash_.*','',Path(name).name)
            bucket=(int(hashlib.sha1(speaker.encode()).hexdigest(),16)%((1<<27)-1))*100./((1<<27)-1)
            part='val' if bucket<10 else 'test' if bucket<20 else 'train'
            raw=z.read(name);sha=digest(raw)
            if sha in seen:
                assert seen[sha]==word, 'Conflicting identical audio labels'
                duplicates+=1;continue
            seen[sha]=word
            split[part].append(dict(source_member=name,speaker=speaker,word=word,sha256=sha))
        speakers={k:{x['speaker'] for x in v} for k,v in split.items()}
        assert not(speakers['train']&speakers['val'] or speakers['train']&speakers['test'] or speakers['val']&speakers['test'])
        save(a.out/'test_reserved_ids.json',split['test'])
        for part in ['train','val']:
            images=[];rows=[]
            for i,record in enumerate(split[part]):
                with wave.open(io.BytesIO(z.read(record['source_member'])),'rb') as w:
                    assert w.getnchannels()==1 and w.getframerate()==16000 and w.getsampwidth()==2 and w.getnframes()<=16000
                    samples=np.frombuffer(w.readframes(w.getnframes()),dtype='<i2').copy()
                image=logmel(samples);images.append(np.repeat(image[:,:,None],3,axis=2))
                target=WORDS.index(record['word'])
                # Balanced pilot: one positive and one deterministic negative per clip.
                offset=1+int(record['sha256'][:8],16)%7
                for label,query in [(1,target),(0,(target+offset)%8)]:
                    rows.append(dict(image_local=i,image_id=record['source_member'],speaker=record['speaker'],
                                     question='does the audio say '+WORDS[query]+' ?',label=label,
                                     audio_class=target,query_class=query))
                if i%1000==0:print(part,i,flush=True)
            np.savez_compressed(a.out/f'{part}_images.npz',images=np.stack(images))
            save(a.out/f'{part}_questions.json',rows)
            save(a.out/f'{part}_audio_records.json',split[part])
    train=json.loads((a.out/'train_questions.json').read_text())
    vocab={'<pad>':0,'<unk>':1}
    for word in sorted({w for row in train for w in tokens(row['question'])}):vocab[word]=len(vocab)
    save(a.out/'vocab.json',vocab)
    counts={k:{w:sum(r['word']==w for r in rows) for w in WORDS} for k,rows in split.items()}
    save(a.out/'manifest.json',dict(source=URL,license='CC BY 4.0',license_source=LICENSE_URL,
        license_evidence_sha256=digest((a.out/'license_evidence.json').read_bytes()),archive_sha256=digest(archive.read_bytes()),
        task='Derived balanced audio/text keyword matching, not official Speech Commands accuracy',
        split='speaker SHA1, validation <10%, test <20%, training remainder; no file-level random split',
        counts=counts,duplicate_waveforms_removed=duplicates,retained_test_decoded=False,
        audio='mono PCM16 16kHz; pad right to 16000; STFT512/hop160/window400 Hann; 64 HTK-mel triangles 20..8000Hz; relative 80dB range to uint8; 64x101 repeated channels',
        files={f.name:digest(f.read_bytes()) for f in a.out.iterdir() if f.is_file() and f.suffix in ['.json','.npz']}))
    print(json.dumps(counts),flush=True)


if __name__=='__main__':main()
