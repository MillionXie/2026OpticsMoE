"""Training-only electronic teacher on the SAME fixed optical input; no test access."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time
import run as t
from test_sweep import ALLOWED_GPUS, DATA_SHA


class Teacher(t.torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features=t.torch.nn.Sequential(
            t.torch.nn.Conv2d(1,16,5,stride=2,padding=2),t.torch.nn.ReLU(),
            t.torch.nn.Conv2d(16,32,3,stride=2,padding=1),t.torch.nn.ReLU(),
            t.torch.nn.Conv2d(32,64,3,stride=2,padding=1),t.torch.nn.ReLU(),
            t.torch.nn.AdaptiveAvgPool2d((4,4)))
        self.head=t.torch.nn.Linear(1024,8)
    def forward(self,x):return self.head(self.features(x).flatten(1))


@t.torch.no_grad()
def evaluate(model,data):
    model.eval();probs=[]
    for idx in t.torch.arange(len(data[1])).split(64):
        probs.append(model(t.b.encode(data[0][idx])).softmax(1).cpu().numpy())
    return t.b.metrics(data[1].cpu().numpy(),t.np.concatenate(probs))


def source():return dict(parent=t.source_identity(),teacher=t.r.sha(Path(__file__)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES');assert gpu in ALLOWED_GPUS
    occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
    assert gpu not in occupied and t.r.sha(a.data)==DATA_SHA
    a.out.mkdir(parents=True,exist_ok=False);t.torch.set_num_threads(4);t.r.setseed(17)
    cfg=dict(seed=17,epochs=40,minimum_epochs=10,patience=8,lr=.001,
        weight_decay=.01,label_smoothing=.02,batch_size=64,encoding='same b.encode amplitude100',
        augmentation=dict(degrees=10.,translation_pixels=3.,scale_delta=.05),
        augmentation_epoch_offset=300,selection='validation balanced NLL, never test',
        purpose='training-only distillation teacher; absent from optical inference')
    src=source();model=Teacher().cuda();opt=t.torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    scheduler=t.torch.optim.lr_scheduler.CosineAnnealingLR(opt,40,eta_min=.0001)
    data=t.k.load_data(a.data,'train');val=t.k.load_data(a.data,'val')
    weights=len(data[1])/(8*t.torch.bincount(data[1],minlength=8).float())
    t.r.save(a.out/'metadata.json',dict(config=cfg,sources=src,command=sys.argv,pid=os.getpid(),gpu_uuid=gpu,
        data_sha256=DATA_SHA,parameters=sum(q.numel() for q in model.parameters()),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        time=t.r.now(),test_read=False,scope='Teacher validation only; never used during deployed optical inference'))
    best=float('inf');wait=0;history=[];started=time.time()
    for epoch in range(1,41):
        model.train();order=t.r.epoch_order(17,epoch+300,len(data[1]))
        theta=t.b.affine_parameters(len(data[1]),17,epoch+300,cfg['augmentation']);total=0
        for idx in order.split(64):
            logits=model(t.b.encode(data[0][idx],theta[idx]));y=data[1][idx]
            target=t.b.F.one_hot(y,8)*(1-.02)+.02/8
            loss=(-(target*logits.log_softmax(1)).sum(1)*weights[y]).mean()
            opt.zero_grad(set_to_none=True);loss.backward();t.torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
            total+=float(loss.detach())*len(idx)
        vm=evaluate(model,val);h=dict(epoch=epoch,val=vm,train_loss=total/len(data[1]))
        if epoch==1 or epoch%5==0:h['train']=evaluate(model,data)
        history.append(h)
        if vm['balanced_nll']<best-.0005:
            best=vm['balanced_nll'];wait=0
            t.r.save_torch(a.out/'best_checkpoint.pt',dict(model=model.state_dict(),epoch=epoch,config=cfg,sources=src,data_sha256=DATA_SHA))
        else:wait+=1
        scheduler.step()
        t.r.save_torch(a.out/'last_checkpoint.pt',dict(model=model.state_dict(),epoch=epoch,config=cfg,sources=src,data_sha256=DATA_SHA))
        t.r.save(a.out/'history.json',history)
        status=dict(state='training_teacher',epoch=epoch,val_accuracy=vm['accuracy'],val_balanced_nll=vm['balanced_nll'],seconds=time.time()-started,test_read=False)
        t.r.save(a.out/'status.json',status);print(t.json.dumps(status),flush=True)
        if epoch>=10 and wait>=8:break
    ck=t.torch.load(a.out/'best_checkpoint.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['model'])
    vm=evaluate(model,val);tm=evaluate(model,data)
    result=dict(selected_epoch=ck['epoch'],epochs_completed=epoch,val=vm,train=tm,
        parameters=sum(q.numel() for q in model.parameters()),checkpoint_sha256=t.r.sha(a.out/'best_checkpoint.pt'),
        eligible_for_distillation=vm['balanced_accuracy']>=.90,selection='validation only',test_read=False)
    t.r.save(a.out/'result.json',result);t.r.save(a.out/'status.json',dict(state='teacher_complete',test_read=False,time=t.r.now()))
    print(t.json.dumps(result),flush=True)


if __name__=='__main__':main()
