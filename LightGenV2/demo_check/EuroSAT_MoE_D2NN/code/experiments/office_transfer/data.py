"""Independent natural-image domains; group duplicates before the split."""
import hashlib,json,math,random
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageOps
from scipy.fft import dctn
from torch.utils.data import Dataset,DataLoader
from experiments.vision_transfer.data import atomic_json
from experiments.qwen3_vl_embedding_2b_cifar10_four_layer_optical_router_classification.data import BalancedClassBatchSampler

PLAN=Path(__file__).with_name('plan.json')
def plan():return json.loads(PLAN.read_text(encoding='utf-8'))
def signature(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def phash(image):
    x=dctn(np.asarray(image.convert('L').resize((32,32)),dtype=float),norm='ortho')[:8,:8].flatten()
    bits=x>np.median(x[1:]);bits[0]=False
    return sum(int(v)<<i for i,v in enumerate(bits))

def prepare():
    p=plan();root=Path(p['data_root']);dest=root/'split10.json'
    if dest.exists():
        data=json.loads(dest.read_text());assert data['plan_sha256']==signature(p),'Existing split policy mismatch'
        assert data['split_sha256']==signature(data['records'])
        return data
    records=[];invalid=[]
    for d,domain in enumerate(p['domains']):
        for y,cls in enumerate(p['classes']):
            folder=root/domain/cls
            if not folder.is_dir():raise RuntimeError('Missing category: '+str(folder))
            for file in sorted(folder.iterdir()):
                if file.suffix.lower() not in ('.jpg','.jpeg','.png','.bmp'):continue
                try:
                    with Image.open(file) as im:
                        im=ImageOps.exif_transpose(im).convert('RGB');im.load()
                        pixel=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest();h=phash(im)
                    records.append(dict(path=file.relative_to(root).as_posix(),label=y,domain=d,pixel_sha256=pixel,phash=h,file_sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
                except Exception as e:invalid.append(dict(path=str(file),error=repr(e)))
    parent=list(range(len(records)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    edges=[]
    for i,a in enumerate(records):
        for j in range(i):
            b=records[j]
            if a['pixel_sha256']==b['pixel_sha256'] or (a['phash']^b['phash']).bit_count()<=p['near_duplicate_phash_distance']:
                parent[find(i)]=find(j);edges.append([i,j])
    groups={}
    for i in range(len(records)):groups.setdefault(find(i),[]).append(i)
    accepted=[];quarantine=[];rng=random.Random(p['split_seed']);stats={}
    for y,cls in enumerate(p['classes']):
        gs=[v for v in groups.values() if records[v[0]]['label']==y and len({records[k]['label'] for k in v})==1]
        rng.shuffle(gs);gs.sort(key=len,reverse=True)
        total=np.array([sum(records[k]['domain']==d for g in gs for k in g) for d in (0,1)])
        desired=np.asarray(p['split_ratios'])[:,None]*total[None,:];actual=np.zeros((3,2),int)
        for g in gs:
            counts=np.array([sum(records[k]['domain']==d for k in g) for d in (0,1)])
            scores=[]
            for s in range(3):
                candidate=actual.copy();candidate[s]+=counts
                scores.append(float((((candidate-desired)/np.maximum(desired,1))**2).sum()))
            s=int(np.argmin(scores));actual[s]+=counts
            group=signature(sorted(records[k]['path'] for k in g))[:16]
            for k in g:records[k].update(split=('train','validation','test')[s],duplicate_group=group);accepted.append(records[k])
        if (actual[1:]<5).any() or (actual[0]<15).any():raise RuntimeError('Insufficient per-domain samples for fixed class '+cls)
        stats[cls]={p['domains'][d]:dict(zip(('train','validation','test'),map(int,actual[:,d]))) for d in (0,1)}
    for g in groups.values():
        if len({records[k]['label'] for k in g})>1:quarantine.extend(records[k] for k in g)
    accepted.sort(key=lambda r:r['path'])
    split=dict(plan_sha256=signature(p),classes=p['classes'],domains=p['domains'],records=accepted,split_sha256=signature(accepted),statistics=stats,
        duplicate_edges=len(edges),duplicate_groups=sum(len(g)>1 for g in groups.values()),quarantined_label_conflicts=quarantine,invalid_images=invalid,
        note='Conservative pHash screening is not an exhaustive semantic duplicate guarantee; all detected groups remain in one split across both domains.')
    atomic_json(dest,split);return split

class Images(Dataset):
    def __init__(self,split,domain,partition,augment=False):
        self.root=Path(plan()['data_root']);self.rows=[r for r in split['records'] if r['domain']==domain and r['split']==partition]
        self.labels=[r['label'] for r in self.rows];self.augment=augment
    def __len__(self):return len(self.rows)
    def image(self,i):
        with Image.open(self.root/self.rows[i]['path']) as im:im=ImageOps.exif_transpose(im).convert('RGB')
        # Preserve object framing; identical deterministic letterboxing at validation/test.
        im=ImageOps.pad(im,(224,224),method=Image.Resampling.BICUBIC,color=(127,127,127))
        if self.augment:
            from torchvision.transforms import ColorJitter,RandomHorizontalFlip
            im=RandomHorizontalFlip()(im);im=ColorJitter(.1,.1,.1,.02)(im)
        return im
    def __getitem__(self,i):return self.image(i),self.labels[i]

def collate(rows):
    im,y=zip(*rows);return list(im),torch.tensor(y,dtype=torch.long)

def train_batches(a,b,epoch,seed,transfer):
    # Independent class-balanced index streams; there is no fabricated cross-domain image pairing.
    steps=math.ceil(max(len(a),len(b))/30) if transfer else math.ceil(len(a)/60)
    sa=BalancedClassBatchSampler(a.labels,classes_per_batch=10,samples_per_class=3 if transfer else 6,steps=steps,seed=seed)
    sa.set_epoch(epoch)
    if transfer:
        sb=BalancedClassBatchSampler(b.labels,classes_per_batch=10,samples_per_class=3,steps=steps,seed=seed+391)
        sb.set_epoch(epoch)
        for ia,ib in zip(sa,sb):
            images=[a.image(i) for i in ia]+[b.image(i) for i in ib]
            yield images,torch.tensor([[a.labels[i],0,i] for i in ia]+[[b.labels[i],1,i] for i in ib])
    else:
        domain=a.rows[0]['domain']
        for ids in sa:yield [a.image(i) for i in ids],torch.tensor([[a.labels[i],domain,i] for i in ids])
