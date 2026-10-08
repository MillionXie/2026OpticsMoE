"""Check exported computation, gradients and stored-checkpoint validation scores."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj)
    return obj


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--package',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--data',type=Path)
    p.add_argument('--run',type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    manifest=json.loads((a.package/'MANIFEST.json').read_text())['files']
    for name,row in manifest.items():
        assert hashlib.sha256((a.package/name).read_bytes()).hexdigest()==row['sha256']
    sys.path.insert(0,str(a.package.resolve()))
    exported=module('exported_models',a.package/'models.py')
    original=module('original_models',Path(__file__).with_name('models.py'))
    cfg=json.loads((a.package/'config.json').read_text())
    assert cfg['architectures']==['dynamic_four','full_d2nn']
    device='cuda' if torch.cuda.is_available() else 'cpu'
    images=torch.randint(1,255,(4,56,56,3),generator=torch.Generator().manual_seed(71),dtype=torch.uint8).to(device)
    labels=torch.tensor([0,1,2,3],device=device);results=[]
    for name in cfg['architectures']:
        old=original.PhaseOnly(name,cfg).to(device);new=exported.PhaseOnly(name,cfg).to(device)
        assert all(torch.equal(old.state_dict()[k],v) for k,v in new.state_dict().items())
        x=old(images);y=new(images)
        for key in x:
            assert (x[key] is None and y[key] is None) or torch.equal(x[key],y[key]),key
        original.objective(x,labels).backward();exported.objective(y,labels).backward()
        assert all(torch.equal(dict(old.named_parameters())[k].grad,v.grad) for k,v in new.named_parameters())
        row=dict(architecture=name,outputs_bitwise_equal=True,gradients_bitwise_equal=True)
        if a.data is not None:
            assert a.run is not None
            data=np.load(a.data,allow_pickle=False)
            checkpoint=torch.load(a.run/name/'best_checkpoint.pt',map_location='cpu',weights_only=False)
            new.load_state_dict(checkpoint['model']);new.eval();probs=[]
            with torch.no_grad():
                for start in range(0,len(data['validation_labels']),cfg['batch_size']):
                    batch=torch.from_numpy(data['validation_images'][start:start+cfg['batch_size']]).to(device)
                    probs.append(new(batch)['probabilities'].cpu().numpy())
            probs=np.concatenate(probs);target=data['validation_labels'];domain=data['validation_domains']
            metrics=dict(accuracy=float((probs.argmax(1)==target).mean()),
                         domain_accuracy={str(d):float((probs.argmax(1)[domain==d]==target[domain==d]).mean()) for d in (0,1)})
            summary=json.loads((a.run/name/'summary.json').read_text())
            for key,value in metrics.items():assert value==summary['validation'][key]
            row['checkpoint_validation']=metrics
        results.append(row)
        del old,new,x,y
    report=dict(passed=True,device=device,torch=torch.__version__,manifest_files=len(manifest),results=results)
    (a.out/'verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
