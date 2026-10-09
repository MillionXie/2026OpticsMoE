"""Paired six-layer MoE/OEO training; fixed phase-leakage ratio only."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
HIST=HERE.parents[1]/'demo_check/reproduction'
sys.path.insert(0,str(HIST))
import kather2016_experiment as k
from kather2016_experiment import b,m,r,torch,np
from optical_reference.optics import PhaseLayer
from model import install_residual,coherent_modulation

original_build=b.build


def build(arch,depth,cfg):
    assert arch=='moe' and depth in (2,4,6)
    model=original_build(arch,depth,cfg)
    install_residual(model,PhaseLayer,cfg['residual_rho'])
    return model


b.build=build


def source_identity():
    return dict(historical=k.sources(),task={p.name:r.sha(p) for p in
        (Path(__file__),HERE/'model.py',HERE/'campaign.py',HERE/'prepare_mango.py')})


def config(rho):
    cfg=k.config('base')
    cfg.update(residual_rho=rho,residual_scope='expert_and_global_only',
        residual_formula='((1-rho)*exp(i*phi)+rho)*U',expert_vectorize=False)
    return cfg


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['smoke','train','evaluate'],required=True)
    parser.add_argument('--rho',type=float,choices=[0.,.3],required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--depth',type=int,choices=[2,4,6],default=6)
    parser.add_argument('--dataset',choices=['kather','mango_variety'],default='kather')
    parser.add_argument('--epochs',type=int,choices=[30,100],default=30)
    a=parser.parse_args()
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    assert visible.startswith('GPU-') and ',' not in visible
    a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    cfg=config(a.rho);src=source_identity()
    if a.epochs==100:
        cfg.update(epochs=100,minimum_epochs=100,patience=100)
    if a.dataset=='mango_variety':
        manifest=r.read(a.data.parent/'data_manifest.json')
        assert manifest['dataset']=='MangoLeafVarietyBD_raw_v2' and manifest['license']=='CC BY 4.0'
        cfg.update(dataset=manifest['dataset'],classes=manifest['classes'],
            deduplication=manifest['split_policy'],scope='Mango variety image-level single-seed optical ablation')
    r.save(a.out/'metadata.json',dict(command=sys.argv,pid=os.getpid(),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        config=cfg,depth=a.depth,sources=src,environment=m.environment(),data_sha256=r.sha(a.data),
        time=r.now(),test_read=a.phase=='evaluate',test_previously_observed=a.dataset=='kather' or a.epochs==100))
    if a.phase=='smoke':
        r.setseed(17)
        base=original_build('moe',a.depth,cfg)
        x=b.encode(k.load_data(a.data,'train')[0][:2])
        base.net.expert_bank.vectorize_homogeneous_d2nn=False
        with torch.no_grad(): reference=b.forward(base,x,'moe')[0]
        state={n:p.detach().clone() for n,p in base.named_parameters()}
        install_residual(base,PhaseLayer,0.)
        with torch.no_grad(): zero=b.forward(base,x,'moe')[0]
        assert torch.equal(reference,zero)
        assert all(torch.equal(p,state[n]) for n,p in base.named_parameters())
        install_residual(base,PhaseLayer,.3)
        prob,capture,_=b.forward(base,x,'moe')
        loss=-prob[:,0].clamp_min(1e-12).log().mean()
        loss.backward()
        gradients={n:float(p.grad.norm()) for n,p in base.named_parameters()}
        assert all(np.isfinite(v) and v>0 for v in gradients.values())
        ones=torch.ones(2,dtype=torch.complex64,device=x.device)
        phase=torch.tensor([0.,np.pi],device=x.device)
        assert torch.allclose(coherent_modulation(ones,phase,.3).abs().square(),
            torch.tensor([1.,.16],device=x.device),atol=1e-6)
        r.save(a.out/'smoke.json',dict(passed=True,rho0_exact_identity=True,
            initialization_unchanged=True,gradients=gradients,targets=5*a.depth))
    elif a.phase=='train':
        r.setseed(17)
        result=b.train('moe',a.depth,17,k.load_data(a.data,'train'),k.load_data(a.data,'val'),
            cfg,a.out,src)
        r.save(a.out/'result.json',result)
        r.setseed(17)
        model=build('moe',a.depth,cfg)
        ck=torch.load(a.out/result['name']/'best_checkpoint.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(ck['model'])
        diagnostics=[]
        hooks=[]
        for name,module in model.named_modules():
            if isinstance(module,PhaseLayer) and hasattr(module,'residual_rho'):
                def hook(mod,inputs,out,name=name):
                    diagnostics.append(dict(layer=name,input_power=float(inputs[0].abs().square().sum()),
                        output_power=float(out.abs().square().sum())))
                hooks.append(module.register_forward_hook(hook))
        with torch.no_grad(): b.forward(model,b.encode(k.load_data(a.data,'val')[0][:8]),'moe')
        for hook in hooks: hook.remove()
        r.save(a.out/'fixed_validation_power.json',dict(indices=list(range(8)),layers=diagnostics))
    else:
        assert a.checkpoint
        ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
        assert ck['config']==cfg and ck['sources']==src and ck['depth']==a.depth
        model=build('moe',a.depth,cfg);model.load_state_dict(ck['model'])
        vm,rows=b.evaluate(model,k.load_data(a.data,'val'),'moe',cfg['batch_size'])
        r.csvwrite(a.out/'val_predictions.csv',rows)
        tm,rows=b.evaluate(model,k.load_data(a.data,'test'),'moe',cfg['batch_size'])
        r.csvwrite(a.out/'test_predictions.csv',rows)
        r.save(a.out/'metrics.json',dict(val=vm,test=tm,checkpoint_sha256=r.sha(a.checkpoint),
            epoch=ck['epoch'],rho=a.rho,depth=a.depth,scope='single_seed_exploratory'))
    r.save(a.out/'status.json',dict(state='complete',phase=a.phase,time=r.now()))


if __name__=='__main__':
    main()
