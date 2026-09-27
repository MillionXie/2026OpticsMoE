"""TRAIN-only physical nearest-positive ranking, anchored to adapted linear head."""
import copy,json
from pathlib import Path
import torch
from torch.nn import functional as F
import layerwise as lab

def main():
    torch.set_num_threads(4);torch.manual_seed(42)
    root=Path(__file__).resolve().parent;out=root/'runs/rank_recovery_trainonly';out.mkdir(exist_ok=False)
    cache=torch.load(root/'runs/rank_cache_trainonly/cache.pt',weights_only=True)
    x=cache['features'].cuda();assert len(x)==1600
    fi=cache['fit'].cuda();vi=cache['validation'].cuda();y=cache['labels'].cuda()
    payload=torch.load(root/'runs/readout50_trainonly_aligned/best.pt',map_location='cpu',weights_only=True)
    base=torch.nn.Linear(1152,64).cuda();base.load_state_dict({k.removeprefix('readout.projection.'):v for k,v in payload['state_dict'].items() if k.startswith('readout.projection.')})
    with torch.no_grad():anchor=F.normalize(base(x),dim=-1)
    def score(z):return float(y[fi][(z[vi]@z[fi].T).argmax(1)].eq(y[vi]).float().mean())
    initial=score(anchor);assert abs(initial-.7875)<1e-5,initial
    positive=y[fi,None].eq(y[fi][None]);diag=torch.eye(len(fi),device='cuda',dtype=torch.bool);positive&=~diag
    negative=~y[fi,None].eq(y[fi][None]);best_score=initial;best=None;history=[]
    for lr in (1e-5,3e-5,1e-4):
        head=copy.deepcopy(base);opt=torch.optim.AdamW(head.parameters(),lr=lr,weight_decay=.01)
        for ep in range(1,101):
            opt.zero_grad();z=F.normalize(head(x[fi]),dim=-1);sim=z@z.T
            # Retrieval succeeds with one valid nearest positive; avoid forcing
            # every same-product view to be equally close as in mean SupCon.
            pos=torch.logsumexp((sim/.04).masked_fill(~positive,-1e4),dim=1)*.04
            neg=torch.logsumexp((sim/.04).masked_fill(~negative,-1e4),dim=1)*.04
            loss=F.softplus((neg-pos+.03)/.04).mean()+2*(1-(z*anchor[fi]).sum(1)).mean()
            loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),.5);opt.step()
            with torch.no_grad():val=score(F.normalize(head(x),dim=-1))
            row={'lr':lr,'epoch':ep,'validation_r1':val,'loss':float(loss.detach())};history.append(row)
            if ep%20==0:print(json.dumps(row),flush=True)
            if val>best_score:best_score=val;best={'state_dict':copy.deepcopy(head.state_dict()),'selection':row}
        torch.save(head.state_dict(),out/f'last_lr{lr}.pt')
    if best:torch.save(best,out/'best_head.pt')
    report={'status':'complete','baseline_validation_r1':initial,'best_validation_r1':best_score,'selection':best['selection'] if best else None,'history':history,'test_evaluated':False,'query_used_for_training_or_selection':False,'upstream_fixed':True,'parameters_added':0}
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='history'}),flush=True)
if __name__=='__main__':main()
