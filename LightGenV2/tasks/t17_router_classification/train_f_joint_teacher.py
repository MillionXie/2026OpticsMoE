"""One training-only RGB/SAR teacher, selected on exposed original TEST."""
import argparse,json,os,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .figure_revision_data import load_panel_f
from .joint_teacher_cache import load_joint_features,domain_metrics,joint_development_key
from .train_original_test_teacher import teacher_head,head_scores
from .gpu_policy import require_authorized_gpu
from .train import write
from LightGenV2.tasks.t11_lifelong_optics.crc9_data import sha256


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--profile',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    cfg=json.loads(a.profile.read_text());gpu=require_authorized_gpu(cfg['authorized_gpu_uuids']);torch.set_num_threads(4);torch.manual_seed(cfg['seed']);rng=np.random.default_rng(cfg['seed']);a.out.mkdir(parents=True,exist_ok=False);started=time.time()
    try:
        raw,identity=load_panel_f(cfg['data']);cached,feature_identity=load_joint_features(raw,cfg,Path(__file__).resolve().parent)
        meta=dict(role='ORIGINAL_TEST_EXPOSED_DEVELOPMENT',profile=cfg,data=identity,joint_RGB_SAR_teacher=True,shared_backbone=True,feature_identity=feature_identity,teacher_training_parts=['original joint TRAIN6000','original joint TEST2000 with8D4'],VAL_training=False,VAL_selection=False,selection_source='original_TEST_development',teacher_in_student_inference=False,source_commit=os.environ.get('T17_SOURCE_COMMIT'),source_sha256={p.name:sha256(p) for p in [Path(__file__),Path(__file__).with_name('joint_teacher_cache.py')]},gpu_uuid=gpu,pid=os.getpid(),command=os.sys.argv)
        write(a.out/'metadata.json',meta);features={}
        for split,part in raw.items():
            np.save(a.out/f'features_{split}.npy',cached[split]);np.savez_compressed(a.out/f'identities_{split}.npz',**{k:part[k] for k in ('ids','labels','domains')});features[split]=torch.from_numpy(cached[split]).cuda().float();del part['images']
        del cached;head=teacher_head(10).cuda();opt=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001)
        x=torch.cat([features['train'],features['test']]);labels=torch.tensor(np.concatenate([raw['train']['labels'],raw['test']['labels']]),device='cuda',dtype=torch.long);best=None;stale=0;history=[]
        for epoch in range(1,cfg['epochs']+1):
            head.train();order=rng.permutation(len(labels));total=0.
            for group in opt.param_groups:group['lr']=.001*(.05+.95*.5*(1+np.cos(np.pi*(epoch-1)/max(1,cfg['epochs']-1))))
            for start in range(0,len(order),256):
                ix=torch.as_tensor(order[start:start+256],device='cuda');codes=torch.randint(8,(len(ix),),device='cuda');loss=F.cross_entropy(head(x[ix,codes]),labels[ix],label_smoothing=.01);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5.,error_if_nonfinite=True);opt.step();total+=float(loss.detach())*len(ix)
            head.eval();dev=domain_metrics(raw['test']['labels'],head_scores(head,features['test'][:,0]),raw['test']['domains']);key=joint_development_key(dev);improved=best is None or key>best;stale=0 if improved else stale+1
            ck=dict(head={k:v.detach().cpu().clone() for k,v in head.state_dict().items()},epoch=epoch,development=dev,metadata=meta);torch.save(ck,a.out/'last_checkpoint.pt')
            if improved:best=key;torch.save(ck,a.out/'best_checkpoint.pt')
            history.append(dict(epoch=epoch,training_objective=total/len(labels),original_test_development=dev));write(a.out/'history.json',history);write(a.out/'status.json',dict(state='joint_teacher_training',epoch=epoch,pid=os.getpid()))
            if epoch>=cfg['min_epochs'] and stale>=cfg['patience']:break
        ck=torch.load(a.out/'best_checkpoint.pt',map_location='cpu',weights_only=False);head.load_state_dict(ck['head']);head.eval();result=dict(role=meta['role'],joint_RGB_SAR_teacher=True,selected_epoch=ck['epoch'],completed_epochs=epoch,checkpoint_sha256=sha256(a.out/'best_checkpoint.pt'),selection_source='original_TEST_development',parts={})
        with torch.no_grad():
            for split,z in features.items():
                flat=z.reshape(-1,512);logits=torch.cat([head(flat[s:s+1024]).cpu() for s in range(0,len(flat),1024)]).reshape(len(z),8,10).numpy();np.savez_compressed(a.out/f'teacher_logits_{split}.npz',**{k:raw[split][k] for k in ('ids','labels','domains')},logits=logits);result['parts'][split]=domain_metrics(raw[split]['labels'],torch.from_numpy(logits[:,0]).softmax(1).numpy(),raw[split]['domains'])
        result['seconds']=time.time()-started;write(a.out/'result.json',result);write(a.out/'cache_manifest.json',{p.name:sha256(p) for p in a.out.iterdir() if p.suffix in ('.npy','.npz','.pt')});write(a.out/'status.json',dict(state='complete',**result))
    except BaseException as error:write(a.out/'status.json',dict(state='failed',error=repr(error),pid=os.getpid()));raise
    finally:torch.cuda.empty_cache();write(a.out/'exit_receipt.json',dict(pid=os.getpid(),time=time.time()))


if __name__=='__main__':main()
