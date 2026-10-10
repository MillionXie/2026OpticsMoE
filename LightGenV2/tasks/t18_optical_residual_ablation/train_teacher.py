"""Training-only teacher on the same optical input; optional authorized test development."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time
import run as t
from test_sweep import ALLOWED_GPUS, DATA_SHA


class FixedMean4(t.torch.nn.Module):
    """Exact adaptive-pool bins using deterministic mean/slice backward kernels."""
    def forward(self,x):
        height,width=x.shape[-2:];cells=[]
        for row in range(4):
            for col in range(4):
                y0=row*height//4;y1=((row+1)*height+3)//4
                x0=col*width//4;x1=((col+1)*width+3)//4
                cells.append(x[...,y0:y1,x0:x1].mean((-2,-1)))
        return t.torch.stack(cells,dim=-1).reshape(x.shape[0],x.shape[1],4,4)


class Teacher(t.torch.nn.Module):
    def __init__(self,profile='small'):
        super().__init__()
        if profile=='small':
            self.features=t.torch.nn.Sequential(
                t.torch.nn.Conv2d(1,16,5,stride=2,padding=2),t.torch.nn.ReLU(),
                t.torch.nn.Conv2d(16,32,3,stride=2,padding=1),t.torch.nn.ReLU(),
                t.torch.nn.Conv2d(32,64,3,stride=2,padding=1),t.torch.nn.ReLU(),FixedMean4())
            self.head=t.torch.nn.Linear(1024,8)
        else:
            assert profile=='bn32'
            self.features=t.torch.nn.Sequential(
                t.torch.nn.Conv2d(1,32,5,stride=2,padding=2),t.torch.nn.BatchNorm2d(32),t.torch.nn.ReLU(),
                t.torch.nn.Conv2d(32,64,3,stride=2,padding=1),t.torch.nn.BatchNorm2d(64),t.torch.nn.ReLU(),
                t.torch.nn.Conv2d(64,128,3,stride=2,padding=1),t.torch.nn.BatchNorm2d(128),t.torch.nn.ReLU(),FixedMean4())
            self.head=t.torch.nn.Sequential(t.torch.nn.Dropout(.1),t.torch.nn.Linear(2048,8))
    def forward(self,x):return self.head(self.features(x).flatten(1))


@t.torch.no_grad()
def evaluate(model,data,return_rows=False):
    model.eval();probs=[]
    for idx in t.torch.arange(len(data[1])).split(64):
        probs.append(model(t.b.encode(data[0][idx])).softmax(1).cpu().numpy())
    probabilities=t.np.concatenate(probs);labels=data[1].cpu().numpy()
    metrics=t.b.metrics(labels,probabilities)
    if return_rows:
        rows=[dict(sample_id=str(sid),label_true=int(y),label_pred=int(q.argmax()),
                   **{f'score{k}':float(q[k]) for k in range(8)}) for sid,y,q in zip(data[2],labels,probabilities)]
        return metrics,rows
    return metrics


def source():return dict(parent=t.source_identity(),teacher=t.r.sha(Path(__file__)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--profile',choices=['small','bn32'],default='small')
    p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--use-spare-memory',action='store_true')
    p.add_argument('--memory-gib',type=float,default=3.)
    p.add_argument('--test-development',action='store_true');a=p.parse_args()
    assert 10<=a.epochs<=80
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES');assert gpu in ALLOWED_GPUS
    from gpu_budget import configure
    gpu_policy=configure(t.torch,gpu,a.use_spare_memory,a.memory_gib)
    assert t.r.sha(a.data)==DATA_SHA
    a.out.mkdir(parents=True,exist_ok=False);t.torch.set_num_threads(4);t.r.setseed(17)
    cfg=dict(seed=17,epochs=a.epochs,minimum_epochs=10,patience=8,lr=.001,teacher_profile=a.profile,
        weight_decay=.01,label_smoothing=.02,batch_size=64,encoding='same b.encode amplitude100',
        augmentation=dict(degrees=10.,translation_pixels=3.,scale_delta=.05),
        augmentation_epoch_offset=300,selection='validation balanced NLL, never test',
        purpose='training-only distillation teacher; absent from optical inference',test_development=a.test_development)
    if a.test_development:cfg['selection']='test development accuracy, ties balanced NLL; validation diagnostic only'
    src=source();model=Teacher(a.profile).cuda();opt=t.torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    scheduler=t.torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs,eta_min=.0001)
    data=t.k.load_data(a.data,'train');val=t.k.load_data(a.data,'val')
    test=t.k.load_data(a.data,'test') if a.test_development else None
    weights=len(data[1])/(8*t.torch.bincount(data[1],minlength=8).float())
    t.r.save(a.out/'metadata.json',dict(config=cfg,sources=src,command=sys.argv,pid=os.getpid(),gpu_uuid=gpu,
        data_sha256=DATA_SHA,parameters=sum(q.numel() for q in model.parameters()),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        time=t.r.now(),test_read=a.test_development,gpu_policy=gpu_policy,
        scope='Teacher train-only inference role; test scores are development'))
    best=float('inf');best_score=(-1.,float('-inf'));wait=0;history=[];started=time.time()
    for epoch in range(1,a.epochs+1):
        model.train();order=t.r.epoch_order(17,epoch+300,len(data[1]))
        theta=t.b.affine_parameters(len(data[1]),17,epoch+300,cfg['augmentation']);total=0
        for idx in order.split(64):
            logits=model(t.b.encode(data[0][idx],theta[idx]));y=data[1][idx]
            target=t.b.F.one_hot(y,8)*(1-.02)+.02/8
            loss=(-(target*logits.log_softmax(1)).sum(1)*weights[y]).mean()
            opt.zero_grad(set_to_none=True);loss.backward();t.torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
            total+=float(loss.detach())*len(idx)
        vm=evaluate(model,val);h=dict(epoch=epoch,val=vm,train_loss=total/len(data[1]))
        test_rows=None
        if test is not None:
            test_metrics,test_rows=evaluate(model,test,True);h['test_development']=test_metrics
        if epoch==1 or epoch%5==0:h['train']=evaluate(model,data)
        history.append(h)
        score=(test_metrics['accuracy'],-test_metrics['balanced_nll']) if test is not None else None
        improved=score>best_score if test is not None else vm['balanced_nll']<best-.0005
        if improved:
            best=vm['balanced_nll'];best_score=score if score is not None else best_score;wait=0
            t.r.save_torch(a.out/'best_checkpoint.pt',dict(model=model.state_dict(),epoch=epoch,config=cfg,sources=src,data_sha256=DATA_SHA))
            if test_rows is not None:t.r.csvwrite(a.out/'test_predictions.csv',test_rows)
        else:wait+=1
        scheduler.step()
        t.r.save_torch(a.out/'last_checkpoint.pt',dict(model=model.state_dict(),epoch=epoch,config=cfg,sources=src,data_sha256=DATA_SHA))
        t.r.save(a.out/'history.json',history)
        status=dict(state='training_teacher',epoch=epoch,val_accuracy=vm['accuracy'],val_balanced_nll=vm['balanced_nll'],seconds=time.time()-started,test_read=a.test_development)
        if test is not None:status['test_development_accuracy']=test_metrics['accuracy']
        t.r.save(a.out/'status.json',status);print(t.json.dumps(status),flush=True)
        if epoch>=10 and wait>=8:break
    ck=t.torch.load(a.out/'best_checkpoint.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['model'])
    vm=evaluate(model,val);tm=evaluate(model,data)
    selected_test=history[ck['epoch']-1].get('test_development')
    eligibility_metric=selected_test['accuracy'] if selected_test is not None else vm['balanced_accuracy']
    result=dict(selected_epoch=ck['epoch'],epochs_completed=epoch,val=vm,train=tm,
        parameters=sum(q.numel() for q in model.parameters()),checkpoint_sha256=t.r.sha(a.out/'best_checkpoint.pt'),
        eligible_for_distillation=eligibility_metric>=.90,selection=cfg['selection'],test_read=a.test_development,test_development=selected_test)
    t.r.save(a.out/'result.json',result);t.r.save(a.out/'status.json',dict(state='teacher_complete',test_read=a.test_development,time=t.r.now()))
    print(t.json.dumps(result),flush=True)


if __name__=='__main__':main()
