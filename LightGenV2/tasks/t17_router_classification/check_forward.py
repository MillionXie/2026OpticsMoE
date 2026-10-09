"""One synthetic GPU backward check; no dataset or test-set evaluation."""
import torch
from .model import RouterClassification

def main():
    for variant in ('optical','electronic','d2nn'):
        model=RouterClassification(variant).cuda()
        x=torch.rand(1,224,224,device='cuda')
        out=model(x)
        loss=torch.nn.functional.cross_entropy(out['logits'],torch.tensor([3],device='cuda'))
        loss.backward()
        assert torch.isfinite(loss)
        phases=[p for n,p in model.named_parameters() if 'phase' in n]
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in phases)
        assert any(p.grad.abs().sum()>0 for p in phases)
        print(variant,'finite optical gradients',float(loss.detach()),flush=True)
        del model,out,loss
        torch.cuda.empty_cache()

if __name__=='__main__': main()
