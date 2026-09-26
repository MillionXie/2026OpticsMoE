"""Fixed-architecture sealed-small recovery with deployment-power constraints."""
import argparse,copy,hashlib,json,math
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader,Subset
from ..sealed_editor import build_sealed
from ..product_unified_edit_data_v2 import ExpandedUnifiedProductEditDataset
from ..qwen_mini_small import PromptEmbeddingLookup
from .ccd_bridge import attach


def write(path,value):
    path.write_text(json.dumps(value,indent=2),encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=600);p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--learning-rate',type=float,default=1e-5)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--noise-probability',type=float,default=.5)
    p.add_argument('--phase-lr-multiplier',type=float,default=5.)
    p.add_argument('--clean-mse-limit',type=float)
    p.add_argument('--evaluate-only',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);torch.manual_seed(1042)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    model=build_sealed(saved).cuda()
    phases=[value for name,value in model.named_parameters() if 'raw_phase' in name or 'raw_router_phase' in name]
    phase_ids={id(value) for value in phases}
    optimizer=torch.optim.AdamW([dict(params=phases,lr=a.phase_lr_multiplier*a.learning_rate),
        dict(params=[value for value in model.parameters() if id(value) not in phase_ids],lr=a.learning_rate)],weight_decay=.01)
    data=a.assets/'datasets';lookup=PromptEmbeddingLookup(data/'abo_unified_expanded_qwen_embeddings_v2.pt')
    datasets={split:ExpandedUnifiedProductEditDataset(data/'abo_cleanrender_lamp_table_pillow_256_v1',split,256,
        data/'abo_unified_expanded_instructions_qwen2_v2.pt') for split in ('train','val','test')}
    train=DataLoader(datasets['train'],batch_size=a.batch_size,shuffle=True,num_workers=0)
    indices=torch.linspace(0,len(datasets['val'])-1,min(8 if a.smoke else 96,len(datasets['val']))).long().tolist()
    val=DataLoader(Subset(datasets['val'],indices),batch_size=a.batch_size)
    state=dict(noisy=False,penalties=[],power=[],roi=[]);hooks=[]
    paths=[model.text.optical,model.editor.bottleneck.optical]
    for path in paths:
        path.zero_order_enabled=True;path.amplitude_zero_order_intensity_min=0.;path.amplitude_zero_order_intensity_max=0.
        path.phase_zero_order_intensity_min=.3;path.phase_zero_order_intensity_max=.3
        path.zero_order_random_relative_phase=True
        path.input_shift_pixels=1;path.phase_shift_pixels=1;path.ccd_shift_pixels=1
        router=path.core.router;router.input_shift_pixels=1;router.phase_shift_pixels=1;router.ccd_shift_pixels=1
        phase_method=router._phase_modulation
        def dc(batch,original=phase_method):
            value=original(batch)
            if state['noisy']:
                angle=value.real.new_empty((batch,1,1)).uniform_(-torch.pi,torch.pi)
                value=math.sqrt(.7)*value+math.sqrt(.3)*torch.exp(1j*angle)
            return value
        router._phase_modulation=dc
        active=path.core.geometry.active_aperture
        for prop in (path.core.propagator,router.propagator):
            def energy_hook(module,inputs,output,active=active):
                incident=inputs[0].abs().square();detector=output.abs().square()
                total=detector.sum((-2,-1)).clamp_min(1e-8)
                eta=detector[:,active.y0:active.y1,active.x0:active.x1].sum((-2,-1))/total
                state['roi'].append(eta.mean());state['penalties'].append(F.relu(.95-eta).mean())
            hooks.append(prop.register_forward_hook(energy_hook))
    def callback(stage,amplitude,phase,ideal):
        # Peak-bounded amplitude is the SLM contract; no gamma or clipping of features.
        scale=amplitude.amax((-2,-1),keepdim=True).clamp_min(1e-6)
        squared=(amplitude/scale).square()
        occupied=(amplitude.detach()>1e-6).sum((-2,-1)).clamp_min(1)
        utilization=squared.sum((-2,-1))/occupied
        state['power'].append(utilization.mean())
        state['penalties'].append(F.relu(.1-utilization).mean())
        if not state['noisy']:return ideal
        # Absolute read floor in peak-normalized optical-intensity units, plus
        # signal-dependent noise. Approximation, not calibrated camera electrons.
        peak_intensity=ideal/scale.square()
        mean=peak_intensity.mean((-2,-1),keepdim=True).detach()
        gain=peak_intensity.new_empty((len(ideal),1,1)).uniform_(.9,1.1)
        noisy=(gain*peak_intensity+.002+torch.randn_like(ideal)*(.002+.03*mean)).clamp_min(0)
        return noisy*scale.square()
    restore=attach(model,callback)
    def forward(batch,noisy,seed=None):
        state.update(noisy=noisy,penalties=[],power=[],roi=[])
        for path in paths:path.train(noisy)
        if seed is not None:torch.manual_seed(seed)
        reference=batch['reference'].cuda();embeddings,mask,_=lookup.batch(list(batch['prompt']),torch.device('cuda'))
        noise=torch.randn_like(reference)
        return model(reference,embeddings.float(),mask,noise),batch['target'].cuda()
    def validate(loader,noisy):
        model.eval();total=0.;count=0;power=[];roi=[]
        with torch.no_grad():
            for i,batch in enumerate(loader):
                pred,target=forward(batch,noisy,9000+i)
                total+=float(F.mse_loss(pred,target))*len(pred);count+=len(pred)
                power.append(float(torch.stack(state['power']).mean()));roi.append(float(torch.stack(state['roi']).mean()))
        return dict(mse_minus1_1=total/count,sample_count=count,power_utilization=sum(power)/len(power),roi_fraction=sum(roi)/len(roi))
    baseline=validate(val,False);baseline_noisy=validate(val,True);best=float('inf');selected=None;history=[]
    if a.evaluate_only:
        test=DataLoader(datasets['test'],batch_size=a.batch_size)
        write(a.output/'report.json',dict(status='complete',scope='original checkpoint matched full TEST evaluation',
            source_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
            test_clean=validate(test,False),test_noisy=validate(test,True)))
        restore()
        for hook in hooks:hook.remove()
        return
    write(a.output/'protocol.json',dict(source_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        geometry='17um/10cm/532nm; no architecture or parameter addition',dc_intensity_fraction=.3,
        pixel_shift=1,noise=f'{a.noise_probability} of batches; absolute .002 + signal .03 mean; proxy, not calibrated electrons',
        losses='MSE+.1L1+.001 average(ROI<.95 and occupied peak-normalized power<.1 deficits)',
        selection='96 fixed VAL members; clean MSE within10% baseline, minimize noisy VAL MSE; TEST no selection',
        clean_mse_limit=a.clean_mse_limit,baseline=baseline,baseline_noisy=baseline_noisy))
    def save(name,step):
        payload=copy.copy(saved);payload['model']={k:v.detach().cpu() for k,v in model.state_dict().items()}
        payload['physical_robust_training']=dict(step=step,dc=.3,pixel_shift=1,source=str(a.checkpoint))
        torch.save(payload,a.output/name)
    step=0
    while step<a.steps:
        for batch in train:
            model.train();optimizer.zero_grad(set_to_none=True)
            prediction,target=forward(batch,bool(torch.rand(())<a.noise_probability))
            quality=F.mse_loss(prediction,target)+.1*F.l1_loss(prediction,target)
            operating=torch.stack(state['penalties']).mean();loss=quality+.001*operating
            if not bool(loss.isfinite()):raise RuntimeError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();step+=1
            if step==1 or step%25==0:
                record=dict(step=step,loss=float(loss),quality=float(quality),operating=float(operating),
                    power=float(torch.stack(state['power']).mean()),roi=float(torch.stack(state['roi']).mean()))
                if step==1:record['phase_gradient_l1']={n:float(p.grad.abs().sum()) if p.grad is not None else None for n,p in model.named_parameters() if 'raw_phase' in n or 'raw_router_phase' in n}
                print(json.dumps(record),flush=True)
            if step%100==0 or step==a.steps:
                clean=validate(val,False);noisy=validate(val,True)
                eligible=clean['mse_minus1_1']<=(a.clean_mse_limit if a.clean_mse_limit is not None else baseline['mse_minus1_1']*1.1)
                record=dict(step=step,clean=clean,noisy=noisy,eligible=eligible);history.append(record)
                if eligible and noisy['mse_minus1_1']<best:best=noisy['mse_minus1_1'];selected=step;save('best.pt',step)
                write(a.output/'history.json',history);print(json.dumps(record),flush=True)
            if step>=a.steps:break
    save('last.pt',step)
    if selected is not None and not a.smoke:
        chosen=torch.load(a.output/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(chosen['model'])
        test=DataLoader(datasets['test'],batch_size=a.batch_size)
        report=dict(status='complete',selected_step=selected,test_clean=validate(test,False),test_noisy=validate(test,True))
    else:report=dict(status='smoke complete' if a.smoke else 'no candidate passed clean quality guard',selected_step=selected)
    write(a.output/'report.json',report);restore()
    for hook in hooks:hook.remove()

if __name__=='__main__':main()
