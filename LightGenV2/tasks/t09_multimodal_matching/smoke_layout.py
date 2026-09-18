"""Check raw audio field contracts and phase gradients before formal training."""
import argparse
import json
from pathlib import Path
import torch
from .model import TextEncoder, OpticalOEO, encode, enlarge_tiles, loss
from .run import load_data, state_sha, setseed
from .prepare import save,digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);setseed(17)
    vocab=json.loads((a.data/'vocab.json').read_text());data=load_data(a.data,'train',vocab,'cuda')
    images=data['images'][data['index'][:4]];ids=data['ids'][:4];y=data['labels'][:4]
    front=TextEncoder(len(vocab),'fixed').cuda();assert sum(p.numel() for p in front.parameters())==0
    records=[]
    for layout in ['legacy','two_band','interleaved','left_right']:
        amplitude=encode(images,front(ids),layout)
        assert amplitude.shape==(4,224,224)
        assert torch.isfinite(amplitude).all() and (amplitude>=0).all()
        assert torch.allclose(amplitude.square().sum((-2,-1)),torch.ones(4,device='cuda'),atol=1e-6)
        expanded=enlarge_tiles(amplitude,478,layout)
        if layout!='legacy':
            for x in [amplitude,expanded]:
                parts=(x[:,0::2],x[:,1::2]) if layout=='interleaved' else (x.chunk(2,dim=-1) if layout=='left_right' else x.chunk(2,dim=-2))
                for band in parts:
                    assert torch.allclose(band.square().sum((-2,-1)),torch.full((4,),.5,device='cuda'),atol=1e-6)
            if layout=='interleaved':
                reference=encode(images,front(ids),'two_band')
                assert torch.equal(amplitude[:,0::2],reference[:,:112])
                assert torch.equal(amplitude[:,1::2],reference[:,112:])
            if layout=='left_right':
                assert torch.equal(amplitude[:,:,:112],encode(images,front(ids),'left_right')[:,:,:112])
            changed=images.clone();changed[...,1]=0
            try:encode(changed,front(ids),layout)
            except AssertionError:pass
            else:raise AssertionError('RGB must not be silently treated as mono')
        else:assert torch.equal(amplitude,encode(images,front(ids)))
        for arch in ['moe','d2nn']:
            model=OpticalOEO(arch,17,input_layout=layout).cuda()
            before=state_sha(model);output=model(amplitude);cost=loss(output,y);cost.backward()
            gradients={n:float(p.grad.norm()) for n,p in model.named_parameters()}
            assert all(v>0 and v<float('inf') for v in gradients.values())
            if arch=='moe':assert all(float(g.norm())>0 for g in model.first_phase.grad)
            torch.optim.Adam(model.parameters(),lr=.001).step();assert before!=state_sha(model)
            records.append(dict(layout=layout,arch=arch,input_sha256=digest(amplitude.cpu().numpy().tobytes()),loss=float(cost),gradients=gradients))
            del model;torch.cuda.empty_cache()
    save(a.out/'verification.json',dict(passed=True,records=records,trainable_electronic_parameters=0))
    print('Layout, power conservation, zero electronic parameters, and all phase gradients PASS',flush=True)

if __name__=='__main__':main()
