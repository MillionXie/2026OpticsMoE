"""Controlled native8 MNIST training; A is fixed old phase, B starts at pi.

Execute on the training server using a committed git-show source. Original
17um data/model modules are reused read-only and their hashes are recorded.
No hardware access, no test-subset selection, no electronics or CCD enhancement.
"""
import argparse,hashlib,json,math,os,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
from experiments.d2nn_mnist4_single_layer_17um_10cm_v2.settings import load_settings
from experiments.d2nn_mnist4_single_layer_17um_10cm_v2.data import build_datasets
from experiments.d2nn_mnist4_single_layer_17um_10cm_v2.modeling import RobustRawCCDMNIST4D2NN,AngularSpectrumKSpacePropagator,translate_zero_fill

PIN='e297b9baa4c028b49695daed24bb291bb71cd93a23f43e2b01ee1d87b1607887'
BASE=Path('experiments/d2nn_mnist4_single_layer_17um_10cm_v2')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2),encoding='utf-8');os.replace(tmp,p)
class PitchModel(nn.Module):
    def __init__(self,s,pitch=8.):
        super().__init__();self.s=s;self.pitch=float(pitch);self.n=round(478*17/pitch);self.grid=round(1024*17/pitch)
        assert self.n%2==0 and self.grid%2==0
        self.raw_phase=nn.Parameter(torch.zeros(self.n,self.n));self.jitter=False
        ix=torch.floor((torch.arange(self.n,dtype=torch.float64)+.5-self.n/2)*pitch/17+239).long().clamp(0,477)
        self.register_buffer('ix',ix,persistent=False)
        # Pixel-area overlap with the ORIGINAL physical detector rectangles;
        # fractional boundary pixels preserve exact region position and area.
        lo=(torch.arange(self.n,dtype=torch.float64)-self.n/2)*pitch;hi=lo+pitch;masks=[]
        for x0,y0,x1,y1 in s.detector_bounds():
            def overlap(v0,v1):return ((torch.minimum(hi,torch.tensor((v1-239)*17.))-torch.maximum(lo,torch.tensor((v0-239)*17.))).clamp(0,pitch)/pitch).float()
            masks.append(overlap(y0,y1)[:,None]*overlap(x0,x1)[None,:])
        self.register_buffer('masks',torch.stack(masks),persistent=False)
        self.propagator=AngularSpectrumKSpacePropagator(grid_size=self.grid,wavelength_nm=s.wavelength_nm,pixel_pitch_um=pitch,
            distance_m=s.detector_distance_m,k_space_enabled=s.k_space_enabled,theta_max_deg=s.k_space_theta_max_deg)
    def phase(self):return 2*torch.pi*torch.sigmoid(self.raw_phase)
    def old_to_native(self,a):return a.index_select(-2,self.ix).index_select(-1,self.ix)
    def shift(self,a):
        if not self.training or not self.jitter or random.random()>=.5:return a
        d=round(17/self.pitch);dy,dx=random.choice([(-d,0),(d,0),(0,-d),(0,d)])
        return translate_zero_fill(a,dy=dy,dx=dx)
    def forward(self,x):
        amp=self.old_to_native(F.pad(x[:,0],(39,39,39,39)))
        field=self.shift(amp)*torch.exp(1j*self.shift(self.phase()))
        guard=(self.grid-self.n)//2
        out=self.propagator(F.pad(field,(guard,guard,guard,guard)))[:,guard:guard+self.n,guard:guard+self.n]
        intensity=self.shift(out).abs().square().float()
        return intensity,torch.einsum('bhw,khw->bk',intensity,self.masks)
    def loss(self,I,y):
        target=self.masks[y]
        # Exact area-average MSE to a 0/1 physical target at partial pixels.
        return 100*(I.square()-2*I*target+target).mean()

@torch.inference_mode()
def evaluate(model,loader,device):
    model.eval();correct=0;total=0;loss=0.;conf=np.zeros((4,4),int)
    for x,y in loader:
        x=x.to(device);y=y.to(device);I,e=model(x);pred=e.argmax(1);correct+=int((pred==y).sum());total+=len(y);loss+=float(model.loss(I,y))*len(y)
        for t,p in zip(y.cpu().tolist(),pred.cpu().tolist()):conf[t,p]+=1
    return dict(accuracy=correct/total,loss=loss/total,n=total,confusion=conf.tolist())

@torch.no_grad()
def export(model,out,name):
    # Native grid already 8um: NO spatial resize, no quantization in training.
    phase=model.phase().cpu().numpy();np.save(out/(name+'_phase_rad.npy'),phase)
    a=np.floor(np.mod(phase,2*np.pi)/(2*np.pi)*256).clip(0,255).astype(np.uint8)[::-1,::-1]
    v=np.zeros((1200,1920),np.uint8);top=(1200-len(a))//2;left=(1920-len(a))//2;v[top:top+len(a),left:left+len(a)]=a
    path=out/(name+'_xy_inverse.bmp');Image.fromarray(255-v).save(path)
    return dict(bmp=path.name,sha256=sha(path),center_xy=[960,600],size_wh=[1920,1200],flip_horizontal=True,flip_vertical=True,gray_encoding='255-minus-g',native_size=model.n)

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--source-commit',required=True);p.add_argument('--mode',choices=['smoke','train'],default='train');p.add_argument('--epochs',type=int,default=60)
    p.add_argument('--microbatch',type=int,default=10);p.add_argument('--effective-batch',type=int,default=100);p.add_argument('--device',default='cuda:0')
    a=p.parse_args();assert a.effective_batch%a.microbatch==0
    torch.manual_seed(42);np.random.seed(42);random.seed(42)
    s=load_settings(BASE/'configs/release/mnist4_single_layer_17um_10cm_v2_notebook_mse_angle_roi.yaml');s.dataset_root=a.dataset_root.resolve();s.download=False
    ckpt=BASE/'runs/mnist4_single_layer_17um_10cm_v2_angle_roi/mask_candidates/checkpoints/post_robust_best.pt';assert sha(ckpt)==PIN
    payload=torch.load(ckpt,map_location='cpu',weights_only=False)
    dataset=build_datasets(s);device=torch.device(a.device)
    a.out.mkdir(parents=True,exist_ok=False)
    protocol=dict(source_commit=a.source_commit,args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},checkpoint_sha256=PIN,
       source_hashes={f.name:sha(f) for f in BASE.glob('*.py')},data_counts=dataset.metadata,
       dimensions=dict(old_n=478,old_pitch_um=17,native_n=1016,native_pitch_um=8,native_grid=2176,numerical_width_um=17408),
       old_support_um=8126,native_raster_support_um=8128,rounding_note='1um raster overhang per edge, both arms identical; detector areas kept exact by pixel-overlap weights',
       same_inputs='original bicubic336 + pad32 + pad39, then physical nearest native raster',k_space_theta_max_deg=s.k_space_theta_max_deg,
       phase_initialization='B raw_phase=0 => pi, A fixed pretrained raw_phase physical nearest',
       parameter_counts=dict(A_original=478**2,A_trainable=0,B_trainable=1016**2),
       no_new_noise=True,no_training_quantization=True,jitter='after epoch8, 0.5 probability independently input/phase/preCCD cardinal 2 native pixels=16um (original1px=17um)',
       selection='existing stratified validation (same split as original), clean accuracy then MSE; official test only A once and B final',
       caution='Different capacity/training history and integer jitter rounding; not a causal isolation of pitch alone',gpu=torch.cuda.get_device_name(device))
    write(a.out/'protocol.json',protocol)
    model=PitchModel(s).to(device);raw=payload['model_state_dict']['raw_phase'].to(device)
    if a.mode=='smoke':
        x,y=dataset.test[0];x=x.unsqueeze(0).to(device)
        old=RobustRawCCDMNIST4D2NN(s).to(device).eval();old.load_state_dict(payload['model_state_dict']);same=PitchModel(s,17).to(device).eval();same.raw_phase.data.copy_(raw)
        with torch.no_grad():
            v=old(x);I,e=same(x);err=float((I-v['ccd_intensity']).abs().max());assert err<1e-5
            assert torch.allclose(model.masks.sum((1,2))*64,torch.full((4,),59**2*17**2,device=device,dtype=torch.float32),atol=1.)
        I,e=model(x);loss=model.loss(I,torch.tensor([int(y)],device=device));loss.backward();grad=float(model.raw_phase.grad.square().mean().sqrt());assert grad>0 and torch.isfinite(model.raw_phase.grad).all()
        oldp=model.phase().detach().clone();opt=torch.optim.Adam([model.raw_phase],lr=.01);opt.step();delta=float((model.phase()-oldp).abs().max());assert delta>0
        write(a.out/'smoke.json',dict(passed=True,old_model_max_abs_error=err,gradient_rms=grad,phase_delta_max_rad=delta,memory_peak_mib=torch.cuda.max_memory_allocated()/2**20));print('SMOKE PASSED',flush=True);return
    eval_loaders={k:torch.utils.data.DataLoader(getattr(dataset,k),batch_size=a.microbatch,num_workers=4,pin_memory=True) for k in ['validation','test']}
    # Evaluate fixed A under the SAME native8 propagator and physical detectors.
    # Match the already-tested A BMP exactly: old export flips the LOGICAL
    # phase before physical rasterization. Undo the native display flip here
    # to simulate those same physical pixels. Exact half-open bin ties mean
    # "raster then flip" and "flip then raster" do not always commute.
    mapped=torch.flip(model.old_to_native(torch.flip(raw,(-2,-1))),(-2,-1))
    model.raw_phase.data.copy_(mapped);model.raw_phase.requires_grad_(False)
    baseline={k:evaluate(model,l,device) for k,l in eval_loaders.items()};baseline['export']=export(model,a.out,'A_old_fixed_native8')
    write(a.out/'baseline_A.json',baseline);print('BASELINE A',json.dumps(baseline),flush=True)
    model.raw_phase.data.zero_();model.raw_phase.requires_grad_(True)
    loader=torch.utils.data.DataLoader(dataset.train,batch_size=a.effective_batch,shuffle=True,num_workers=4,pin_memory=True,generator=torch.Generator().manual_seed(42))
    opt=torch.optim.Adam([model.raw_phase],lr=.01);history=[];best=(-1.,-float('inf'));start=time.time()
    for epoch in range(1,a.epochs+1):
        model.train();model.jitter=epoch>8;count=0;loss_sum=0.;correct=0;grad_sum=0.;updates=0
        previous=model.phase().detach().clone()
        for step,(x,y) in enumerate(loader):
            opt.zero_grad(set_to_none=True);total=len(y)
            for offset in range(0,total,a.microbatch):
                xx=x[offset:offset+a.microbatch].to(device);yy=y[offset:offset+a.microbatch].to(device);I,e=model(xx);loss=model.loss(I,yy)
                if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
                (loss*len(yy)/total).backward();loss_sum+=float(loss.detach())*len(yy);correct+=int((e.argmax(1)==yy).sum());count+=len(yy)
            grad=model.raw_phase.grad
            if grad is None or not torch.isfinite(grad).all():raise RuntimeError('Invalid phase gradient')
            grad_sum+=float(grad.square().mean().sqrt());updates+=1;torch.nn.utils.clip_grad_norm_([model.raw_phase],5.);opt.step()
            if step%40==0:print(f'epoch={epoch} step={step}/{len(loader)} loss={loss_sum/count:.5f} accuracy={correct/count:.4f}',flush=True)
        val=evaluate(model,eval_loaders['validation'],device);delta=float((model.phase()-previous).square().mean().sqrt())
        row=dict(epoch=epoch,train_loss=loss_sum/count,train_accuracy=correct/count,validation=val,phase_gradient_rms=grad_sum/updates,phase_delta_rms_rad=delta,
                 phase_std_rad=float(model.phase().std()),elapsed_s=time.time()-start,memory_peak_mib=torch.cuda.max_memory_allocated()/2**20)
        if delta<=0:raise RuntimeError('Phase did not move')
        history.append(row);write(a.out/'history.json',history)
        state=dict(model_state_dict=model.state_dict(),optimizer_state_dict=opt.state_dict(),epoch=epoch,metrics=row,protocol=protocol)
        torch.save(state,a.out/'last_checkpoint.pt')
        score=(val['accuracy'],-val['loss'])
        if score>best:best=score;torch.save(state,a.out/'best_checkpoint.pt');write(a.out/'best.json',row)
        print('EPOCH',json.dumps(row),flush=True)
    saved=torch.load(a.out/'best_checkpoint.pt',map_location=device,weights_only=False);model.load_state_dict(saved['model_state_dict'])
    result=dict(status='complete',best_epoch=saved['epoch'],baseline_A=baseline,test_B=evaluate(model,eval_loaders['test'],device),export_B=export(model,a.out,'B_native8_best'),
                note='Simulation only; separate paired hardware test required. No measured 60% results overwritten.')
    write(a.out/'result.json',result);print('FINAL',json.dumps(result),flush=True)
if __name__=='__main__':main()
