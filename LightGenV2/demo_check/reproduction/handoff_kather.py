"""Git-independent single-model train/evaluate entry for the laboratory package."""
import argparse,json
from pathlib import Path
import kather_followup as f
from kather_followup import b,m,k,r,torch,np

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['train','evaluate'],required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--config',type=Path);p.add_argument('--arch',choices=f.SPEC['architectures']);p.add_argument('--depth',type=int,choices=[2,4,6],default=6);p.add_argument('--seed',type=int,default=17);p.add_argument('--checkpoint',type=Path);p.add_argument('--split',choices=['val','test'],default='val');a=p.parse_args()
    torch.set_num_threads(4);a.out.mkdir(parents=True,exist_ok=False)
    if a.phase=='train':
        assert a.config and a.arch;cfg=r.read(a.config);r.setseed(a.seed)
        if cfg.get('expert_input_coverage')=='full':import kather_coverage
        f.drop_probability=cfg.get('input_dropout',0.);f.drop_generator=torch.Generator().manual_seed(a.seed+901234);f.mask_hashes=[]
        src=f.sources();r.save(a.out/'metadata.json',dict(config=cfg,arch=a.arch,depth=a.depth,seed=a.seed,sources=src,data_sha256=r.sha(a.data),environment=m.environment(),test_read=False))
        result=b.train(a.arch,a.depth,a.seed,k.load_data(a.data,'train'),k.load_data(a.data,'val'),cfg,a.out,src);r.save(a.out/'result.json',result)
    else:
        assert a.checkpoint;ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);cfg=ck['config'];arch=ck['arch']
        if cfg.get('expert_input_coverage')=='full':import kather_coverage
        model=m.build(arch,ck['depth'],cfg);model.load_state_dict(ck['model'])
        metrics,rows=b.evaluate(model,k.load_data(a.data,a.split),arch,cfg['batch_size']);r.csvwrite(a.out/(a.split+'_predictions.csv'),rows);r.save(a.out/'metrics.json',dict(metrics=metrics,checkpoint_sha256=r.sha(a.checkpoint),data_sha256=r.sha(a.data),environment=m.environment(),arch=arch,depth=ck['depth'],seed=ck['seed'],selected_epoch=ck['epoch']))
        print(json.dumps(metrics),flush=True)
    r.save(a.out/'status.json',dict(state='complete',time=r.now()))

if __name__=='__main__':main()
