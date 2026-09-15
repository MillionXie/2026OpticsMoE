"""Fixed paired geographic splits and architecture-independent sampling/augmentation."""
import hashlib, json, os
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.transforms import ColorJitter, RandomHorizontalFlip
from experiments.qwen3_vl_embedding_2b_cifar10_four_layer_optical_router_classification.data import BalancedClassBatchSampler

ROOT=Path(__file__).resolve().parent
PLAN=json.loads((ROOT/'PLAN.json').read_text())
RECORDS=json.loads((ROOT/'SPLIT.json').read_text())['records']
DATA=Path(PLAN['data_root'])
STAGES={s['name']:s for s in PLAN['moe_stages']}

def signature(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def file_sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2),encoding='utf-8');os.replace(temp,path)

def verify_source():
    assert json.loads((ROOT/'USER_AUTHORIZATION.json').read_text())['approved'] and PLAN['training_authorized']
    manifest=json.loads((ROOT/'SOURCE_MANIFEST.json').read_text())
    for name,sha in manifest.items():
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or file_sha(path)!=sha:raise RuntimeError('Sealed source changed: '+name)
    assert signature(RECORDS)==PLAN['split_records_sha256']
    return signature(manifest)

class Images:
    def __init__(self,domain,partition):
        if partition=='test' and not (ROOT/'runs/SELECTION_SEAL.json').is_file():
            raise RuntimeError('Test access requires all four selected checkpoints sealed')
        self.indices=[i for i,r in enumerate(RECORDS) if r['domain']==domain and r['split']==partition]
        self.labels=[RECORDS[i]['label'] for i in self.indices]
        self.domain=0 if domain=='A' else 1
    def __len__(self):return len(self.indices)
    def image(self,i,aug_seed=None):
        with Image.open(DATA/RECORDS[self.indices[i]]['path']) as im: im=ImageOps.exif_transpose(im).convert('RGB')
        im=ImageOps.pad(im,(224,224),method=Image.Resampling.BICUBIC,color=(127,127,127))
        if aug_seed is not None:
            # Independent of model architecture, checkpoint RNG, and augmentation consumption elsewhere.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(aug_seed)
                im=RandomHorizontalFlip()(im);im=ColorJitter(.1,.1,.1,.02)(im)
        return im

def index_batches(stage,epoch,domains=None,steps=None):
    domains=domains or STAGES[stage]['domains'];steps=steps or STAGES[stage]['steps_per_epoch']
    ds=[Images(d,'train') for d in domains];paired=len(ds)==2
    samplers=[]
    for j,d in enumerate(ds):
        sampler=BalancedClassBatchSampler(d.labels,classes_per_batch=10,samples_per_class=3 if paired else 6,steps=steps,seed=1484+(391 if paired and j==1 else 0))
        sampler.set_epoch(epoch);samplers.append(sampler)
    for step,groups in enumerate(zip(*samplers),1):
        rows=[]
        for d,ids in zip(ds,groups):
            for i in ids:
                pos=len(rows);gid=d.indices[i]
                seed=int.from_bytes(hashlib.sha256(f'42|{stage}|{epoch}|{step}|{pos}|{gid}'.encode()).digest()[:8],'little')%(2**63-1)
                rows.append((d,i,[d.labels[i],d.domain,gid,seed]))
        yield rows

def train_batches(stage,epoch,domains=None,steps=None):
    for rows in index_batches(stage,epoch,domains,steps):
        yield [d.image(i,meta[3]) for d,i,meta in rows],torch.tensor([meta for _,_,meta in rows],dtype=torch.long)

def eval_batches(domain,partition,batch_size=60,limit=None):
    ds=Images(domain,partition)
    for start in range(0,min(len(ds),limit or len(ds)),batch_size):
        ids=range(start,min(start+batch_size,len(ds),limit or len(ds)))
        yield [ds.image(i) for i in ids],torch.tensor([[ds.labels[i],ds.domain,ds.indices[i]] for i in ids],dtype=torch.long)

def batch_hash(rows):return hashlib.sha256(np.asarray([meta for _,_,meta in rows],dtype=np.int64).tobytes()).hexdigest()
