import contextlib,copy,hashlib,json,math,re
from pathlib import Path
import torch
from experiments.vision_transfer import model as M,engine as E
from experiments.office_transfer.data import signature,prepare,Images,train_batches,atomic_json
from experiments.office_transfer.runtime import setup

ROOT=Path(__file__).resolve().parents[2]
def plan():return json.loads(Path(__file__).with_name('plan.json').read_text(encoding='utf-8'))
def authorize():
    auth=json.loads((ROOT/'MERGE_AUTHORIZATION.json').read_text(encoding='utf-8'))
    manifest=json.loads((ROOT/'MERGE_MANIFEST.json').read_text())
    if not auth['approved'] or auth['plan_sha256']!=signature(plan()) or auth['manifest_sha256']!=signature(manifest):raise RuntimeError('Expert merge authorization mismatch')
    for n,sha in manifest.items():
        f=(ROOT/n).resolve()
        if not f.is_relative_to(ROOT) or hashlib.sha256(f.read_bytes()).hexdigest()!=sha:raise RuntimeError('Expert merge source mismatch: '+n)

def grouped_weights(domain):
    if domain.ndim!=1 or not bool(((domain==0)|(domain==1)).all()):raise ValueError('Expected training domain labels 0/1')
    mask=(torch.arange(4,device=domain.device)[None,:]//2)==domain[:,None]
    return mask.float()/math.sqrt(2)

class MergeRouter(M.ReservedRouter):
    merge_mode='automatic'
    merge_domains=None
    def forward(self,fields):
        route=super().forward(fields)
        if self.merge_mode=='uniform':weights=torch.full_like(route['weights'],.5)
        elif self.merge_mode=='isolated':
            if self.merge_domains is None:raise RuntimeError('Isolated diagnostic/training mode needs explicit domain')
            d=self.merge_domains
            if isinstance(d,int):d=torch.full((len(fields),),d,device=fields.device,dtype=torch.long)
            weights=grouped_weights(d).to(route['weights'].device)
            if weights.shape!=route['weights'].shape:raise RuntimeError('Domain batch mismatch')
        elif self.merge_mode=='automatic':return route
        else:raise RuntimeError('Unknown route mode')
        route.update(weights=weights,selected_mask=weights>0,routing_mode='expert_merge_'+self.merge_mode)
        route['selected_indices']=torch.argsort(weights,dim=1,descending=True)[:,:2] if self.merge_mode=='isolated' else torch.arange(4,device=weights.device).expand(len(weights),-1)
        return route

def build(loaded,s):
    r,h=M.build(loaded,s,'moe');r.vision_surrogate.core.optical_branch.core.router.__class__=MergeRouter
    return r,h

@contextlib.contextmanager
def routing(r,mode='automatic',domains=None):
    router=r.vision_surrogate.core.optical_branch.core.router
    old=(router.merge_mode,router.merge_domains)
    router.merge_mode,router.merge_domains=mode,domains
    try:yield
    finally:router.merge_mode,router.merge_domains=old

def forward(loaded,r,h,inputs,mode='automatic',domains=None):
    with routing(r,mode,domains):return M.predict(loaded,r,h,inputs)

def evaluate(loaded,r,h,s,split,mode='automatic'):
    result={}
    for d,label in ((0,'A'),(1,'B')):
        with routing(r,mode,d if mode=='isolated' else None):result[label]=E.evaluate(loaded,r,h,Images(split,d,'validation'),s)
    result['mean']=(result['A']['accuracy']+result['B']['accuracy'])/2
    return result

def state_tensors(state):
    return {prefix+'.'+n:t for prefix in ('vision_optical','classification_head') for n,t in state[prefix].items()}
def expert_number(name):
    match=re.search(r'\.experts\.(\d+)\.raw_phase$',name)
    return int(match[1]) if match else None
def state_digest(state,shared_only=False):
    sha=hashlib.sha256()
    sha.update(state['backbone_metadata']['frozen_stem_sha256'].encode())
    for n,t in sorted(state_tensors(state).items()):
        if shared_only and expert_number(n) is not None:continue
        sha.update(n.encode());sha.update(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return sha.hexdigest()

def merge_states(shared,a,b):
    base=state_digest(shared,True)
    if base!=state_digest(a,True) or base!=state_digest(b,True):raise RuntimeError('Cannot merge experts with different shared electronics, head, router, global phase or visual stem')
    merged=copy.deepcopy(shared);copied=[]
    for prefix in ('vision_optical','classification_head'):
        for n,t in merged[prefix].items():
            i=expert_number(prefix+'.'+n)
            if i is not None:merged[prefix][n]=(a if i<2 else b)[prefix][n].clone();copied.append(i)
    if sorted(copied)!=[0,1,2,3]:raise RuntimeError('Expected exactly four expert phase tensors')
    merged.update(stage='merged',shared_sha256=base,origin_checkpoint_sha256={'shared':state_digest(shared),'A':state_digest(a),'B':state_digest(b)})
    return merged

def optimizer(r,h,stage,epoch,old=None):
    cfg=plan();scale=.2+.8*.5*(1+math.cos(math.pi*(epoch-1)/max(1,cfg['epochs'][stage]-1)))
    if stage=='shared':
        rates=dict(electronic=2e-4*scale,head=1e-3*scale)
        if epoch>cfg['shared_warmup_epochs']:rates['shared_phase']=1e-3*scale
    elif stage=='expert_A':rates=dict(expert_a=3e-3*scale)
    elif stage=='expert_B':rates=dict(expert_b=3e-3*scale)
    elif stage=='router':rates=dict(router=1e-3*scale)
    else:raise ValueError(stage)
    groups={}
    for n,p in M.named(r,h).items():
        g=M.group_of(n);p.grad=None;p.requires_grad_(n in r.transfer_eligible and rates.get(g,0)>0)
        if p.requires_grad:groups.setdefault(g,[]).append(p)
    if old is not None and set(groups)=={g['group_name'] for g in old.param_groups}:
        for g in old.param_groups:g['lr']=rates[g['group_name']]
        return old
    return torch.optim.AdamW([dict(params=ps,group_name=g,lr=rates[g],weight_decay=0.) for g,ps in groups.items()])

def load_checkpoint(path,split):
    s=torch.load(path,map_location='cpu',weights_only=False)
    if s['merge_plan_sha256']!=signature(plan()) or s['split_sha256']!=split['split_sha256']:raise RuntimeError('Checkpoint data/plan mismatch')
    return s
