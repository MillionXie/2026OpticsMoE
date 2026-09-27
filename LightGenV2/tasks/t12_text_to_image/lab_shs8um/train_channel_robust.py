"""TRAIN-only clean-anchored channel adaptation, VAL-only selection."""
import argparse
import copy
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from ..sealed_editor import build_sealed
from ..audited_unified import architecture_report,configure_fusion_bounds,optical_diagnostics,add_decoder_refinement
from ..product_unified_edit_data_v2 import ExpandedUnifiedProductEditDataset
from ..qwen_mini_small import PromptEmbeddingLookup
from .robust_channel import RobustChannel, PROFILES


def write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def ssim_per_image(a, b):
    x = torch.arange(11, device=a.device, dtype=a.dtype)-5
    g = torch.exp(-x.square()/(2*1.5**2)); g = g/g.sum()
    kernel = (g[:,None]*g[None,:]).expand(3,1,11,11)
    blur = lambda v: F.conv2d(v,kernel,groups=3)
    ma, mb = blur(a), blur(b)
    va, vb = blur(a*a)-ma*ma, blur(b*b)-mb*mb
    cov = blur(a*b)-ma*mb
    return (((2*ma*mb+.01**2)*(2*cov+.03**2))/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2))).mean((1,2,3))


def circular_phase_tv(model):
    planes=[]
    for module in model.modules():
        if any(n in ('raw_phase','raw_router_phase') for n,_ in module.named_parameters(recurse=False)):
            if not callable(getattr(module,'phase',None)):raise ValueError('Missing physical phase accessor')
            planes.append(module.phase())
    return sum((1-torch.cos(v[...,1:,:]-v[...,:-1,:])).mean()+(1-torch.cos(v[...,:,1:]-v[...,:,:-1])).mean() for v in planes)/len(planes)


class IndexedDataset:
    def __init__(self, dataset): self.dataset=dataset
    def __len__(self): return len(self.dataset)
    def __getitem__(self, index): return dict(self.dataset[index], index=index)


def main():
    p = argparse.ArgumentParser()
    for name in ('checkpoint','assets','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--profile',choices=tuple(PROFILES),default='camera')
    p.add_argument('--train-profiles',nargs='+',choices=tuple(PROFILES))
    p.add_argument('--eval-profiles',nargs='+',default=['clean','camera','combined','stress'])
    p.add_argument('--selection-profiles',nargs='+',default=['camera','combined'])
    p.add_argument('--teacher-checkpoint',type=Path)
    p.add_argument('--alpha-min',type=float)
    p.add_argument('--lr-phase',type=float,default=2e-5)
    p.add_argument('--lr-electronic',type=float,default=2e-6)
    p.add_argument('--lr-alpha',type=float,default=2e-6)
    p.add_argument('--feature-consistency',type=float,default=0.)
    p.add_argument('--routing-consistency',type=float,default=0.)
    p.add_argument('--phase-tv',type=float,default=0.)
    p.add_argument('--clean-weight',type=float,default=.5)
    p.add_argument('--anchor-weight',type=float,default=.1)
    p.add_argument('--noisy-anchor-weight',type=float,default=.05)
    p.add_argument('--region-weight',type=float,default=0.)
    p.add_argument('--source-gate-weight',type=float,default=0.)
    p.add_argument('--decoder-refinement',action='store_true')
    p.add_argument('--lr-refinement',type=float,default=1e-4)
    p.add_argument('--ccd-floor-quantile',type=float)
    p.add_argument('--steps',type=int,default=600)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--val-samples',type=int,default=96)
    p.add_argument('--evaluate-only',action='store_true')
    p.add_argument('--split',choices=('val','test'),default='val')
    p.add_argument('--seed',type=int,default=1042)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    if not 0<a.clean_weight<1:raise ValueError('clean-weight must lie in (0,1)')
    torch.set_num_threads(4);torch.manual_seed(a.seed)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    model=build_sealed(saved).cuda().eval()
    if a.decoder_refinement:
        add_decoder_refinement(model)
        model.cuda()
    if a.alpha_min is not None:configure_fusion_bounds(model,minimum=a.alpha_min)
    if a.ccd_floor_quantile is not None:
        from .detector_correction import install
        install(model,dict(floor_quantile=a.ccd_floor_quantile))
    report=architecture_report(model)
    if report['counted_parameters']>20_000_000: raise ValueError('20M budget exceeded')
    channel=RobustChannel(model)
    data=a.assets/'datasets'
    cache=data/'abo_unified_expanded_instructions_qwen2_v2.pt'
    lookup=PromptEmbeddingLookup(data/'abo_unified_expanded_qwen_embeddings_v2.pt')
    def dataset(split):
        return IndexedDataset(ExpandedUnifiedProductEditDataset(data/'abo_cleanrender_lamp_table_pillow_256_v1',split,256,cache))
    valset=dataset('val')
    indices=torch.linspace(0,len(valset)-1,min(a.val_samples,len(valset))).long().tolist()
    val=DataLoader(Subset(valset,indices),batch_size=a.batch_size)
    execution=dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                   command=sys.argv,torch=torch.__version__,gpu=torch.cuda.get_device_name(),
                   source_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
                   data_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in (cache,data/'abo_cleanrender_lamp_table_pillow_256_v1/val.jsonl',data/'abo_cleanrender_lamp_table_pillow_256_v1/train.jsonl')})
    if a.teacher_checkpoint:execution['teacher_sha256']=hashlib.sha256(a.teacher_checkpoint.read_bytes()).hexdigest()
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    write(a.output/'protocol.json',dict(execution=execution,config=config,
          counted_parameters=report['counted_parameters'],profiles=PROFILES,
          amplitude_contract=saved['bounded_amplitude'],validation_indices=indices,
          selection='minimize mean '+ '/'.join(a.selection_profiles)+' VAL MSE; clean VAL mean per-image PSNR drop <=0.2dB, SSIM drop <=0.002; TEST only after VAL selection',
          leakage='nominal coherent branch POWER fraction; sqrt(1-eta)*E_mod + sqrt(eta)*exp(i phi)*E_unmod; same bounded incident field; no image constants',
          detector='bounded-field intensity proxy, not calibrated camera electrons; no TEST CCD fitting'))

    def inputs(batch):
        ref=batch['reference'].cuda();target=batch['target'].cuda()
        emb,mask,_=lookup.batch(list(batch['prompt']),torch.device('cuda'))
        # Identical per-index seed to deployment full TEST, independent of batch size.
        noise=torch.stack([torch.randn(ref[0].shape,generator=torch.Generator().manual_seed(1042+int(i))) for i in batch['index']]).cuda()
        return ref,emb.float(),mask,noise,target

    @torch.no_grad()
    def evaluate(loader, profiles):
        model.eval(); results={};clean_masks={}
        for profile in profiles:
            sums=dict(mse_0_1=0.,psnr_db=0.,ssim=0.); count=0; rows=[]
            routing={branch:dict(selection_sum=torch.zeros(4),probability_sum=torch.zeros(4),pairs=Counter(),changed=0) for branch in ('language','vision')}
            for batch_index,batch in enumerate(loader):
                ref,emb,mask,noise,target=inputs(batch)
                # Fixed perturbation seed per batch for reproducible VAL ranking.
                torch.manual_seed(9000+int(batch['index'][0]))
                channel.configure(None if profile=='clean' else profile)
                output=model(ref,emb,mask,noise)
                pred=output.float().add(1).div(2).clamp(0,1)
                gt=target.float().add(1).div(2).clamp(0,1)
                mse=(pred-gt).square().mean((1,2,3));psnr=-10*torch.log10(mse.clamp_min(1e-15))
                ss=ssim_per_image(pred,gt)
                region=batch['object_union'].cuda()
                roi_mse=((pred-gt).square()*region).sum((1,2,3))/(3*region.sum((1,2,3))).clamp_min(1.)
                for branch,obj in (('language',model.text),('vision',model.editor.bottleneck)):
                    live=obj.last_routing
                    selected_mask=live['selected_mask'].detach().cpu().bool()
                    probabilities=live['probabilities'].detach().cpu()
                    r=routing[branch]
                    r['selection_sum']+=selected_mask.float().sum(0)
                    r['probability_sum']+=probabilities.sum(0)
                    for j,index in enumerate(batch['index'].tolist()):
                        r['pairs'][str(selected_mask[j].nonzero().flatten().tolist())]+=1
                        if profile=='clean':clean_masks[branch,index]=selected_mask[j]
                        elif (branch,index) in clean_masks:r['changed']+=int(not torch.equal(clean_masks[branch,index],selected_mask[j]))
                for j in range(len(ref)):
                    row=dict(index=int(batch['index'][j]),sample_id=batch['sample_id'][j],mode=batch['mode'][j],category=batch['category'][j],mse_0_1=float(mse[j]),psnr_db=float(psnr[j]),ssim=float(ss[j]),roi_mse=float(roi_mse[j]),roi_psnr_db=float(-10*torch.log10(roi_mse[j].clamp_min(1e-15))))
                    rows.append(row)
                    for k in sums:sums[k]+=row[k]
                count+=len(ref)
                if a.evaluate_only and batch_index%200==0:
                    print(json.dumps(dict(profile=profile,completed=count)),flush=True)
            audit={branch:dict(selection_rate=r['selection_sum'].div(count).tolist(),probability_mean=r['probability_sum'].div(count).tolist(),pair_counts=dict(r['pairs']),top2_changed_fraction=r['changed']/count) for branch,r in routing.items()}
            by_mode={}
            for mode in ('background','object','joint'):
                subset=[r for r in rows if r['mode']==mode]
                if subset:by_mode[mode]=dict(samples=len(subset),**{k:sum(r[k] for r in subset)/len(subset) for k in ('mse_0_1','psnr_db','ssim','roi_mse','roi_psnr_db')})
            results[profile]=dict(samples=count,**{k:v/count for k,v in sums.items()},routing=audit,by_mode=by_mode)
            if a.evaluate_only: write(a.output/(profile+'_per_image.json'),rows)
        return results

    if a.evaluate_only:
        loader=DataLoader(dataset(a.split),batch_size=a.batch_size)
        write(a.output/'evaluation.json',dict(split=a.split,execution=execution,metrics=evaluate(loader,a.eval_profiles),alpha=optical_diagnostics(model),
              caveat='perturbed simulations are NOT new physical CCD measurements'))
        channel.restore();return

    teacher_saved=torch.load(a.teacher_checkpoint,map_location='cpu',weights_only=False) if a.teacher_checkpoint else saved
    teacher=build_sealed(teacher_saved).cuda().eval().requires_grad_(False)
    validation_profiles=list(dict.fromkeys(['clean']+a.selection_profiles))
    baseline=evaluate(val,validation_profiles)
    best_score=sum(baseline[p]['mse_0_1'] for p in a.selection_profiles)/len(a.selection_profiles)
    baseline_clean=baseline['clean']
    if a.teacher_checkpoint:
        student=model;model=teacher
        baseline_clean=evaluate(val,['clean'])['clean'];model=student
    selected=0; history=[]
    def save(name,step):
        payload=copy.copy(saved)
        payload['model']={k:v.detach().cpu() for k,v in model.state_dict().items()}
        payload['channel_robust_training']=dict(step=step,profiles=a.train_profiles or [a.profile],training_config=config,execution=execution)
        payload['decoder_refinement']=getattr(model,'decoder_refinement',False)
        payload['detector_correction']=getattr(model,'detector_correction',None)
        if hasattr(model,'fusion_bounds'):payload['fusion_bounds']=model.fusion_bounds
        torch.save(payload,a.output/name)
    save('best_checkpoint.pt',0)
    write(a.output/'baseline_val.json',baseline)
    phases=[v for n,v in model.named_parameters() if 'raw_phase' in n or 'raw_router_phase' in n]
    groups=[dict(params=phases,lr=a.lr_phase),
            dict(params=[v for n,v in model.named_parameters() if 'raw_phase' not in n and 'raw_router_phase' not in n and 'raw_alpha' not in n and '.details.' not in n],lr=a.lr_electronic),
            dict(params=[v for n,v in model.named_parameters() if 'raw_alpha' in n],lr=a.lr_alpha),
            dict(params=[v for n,v in model.named_parameters() if '.details.' in n],lr=a.lr_refinement)]
    optimizer=torch.optim.AdamW(groups,weight_decay=0.)
    loader=DataLoader(dataset('train'),batch_size=a.batch_size,shuffle=True,num_workers=0,
                      generator=torch.Generator().manual_seed(a.seed))
    step=0
    features={}
    gates={}
    handles=[]
    if a.source_gate_weight:
        if model.editor.source_gate is None:raise ValueError('No learned source gate')
        handles.append(model.editor.source_gate.register_forward_hook(lambda module,inputs,output:gates.__setitem__('logits',output)))
    if a.feature_consistency:
        for name,module in (('language1',model.text.fusion1),('language2',model.text.fusion2),('vision1',model.editor.bottleneck.fusion1),('vision2',model.editor.bottleneck.fusion2)):
            handles.append(module.register_forward_hook(lambda module,inputs,output,name=name:features.__setitem__(name,output)))
    training_profiles=a.train_profiles or [a.profile]
    while step<a.steps:
        for batch in loader:
            model.train();optimizer.zero_grad(set_to_none=True)
            ref,emb,mask,noise,target=inputs(batch)
            with torch.no_grad(): anchor=teacher(ref,emb,mask,noise)
            channel.configure(None);clean=model(ref,emb,mask,noise)
            clean_gate=gates.get('logits')
            clean_features={k:v.detach() for k,v in features.items()}
            clean_routing=[obj.last_routing['probabilities'].detach() for obj in (model.text,model.editor.bottleneck)]
            strength=.2+.8*min(1.,step/max(1,a.steps*.5))
            profile=training_profiles[step%len(training_profiles)]
            channel.configure(profile,strength);noisy=model(ref,emb,mask,noise)
            quality=lambda pred:F.mse_loss(pred,target)+.1*F.l1_loss(pred,target)
            loss=a.clean_weight*quality(clean)+(1-a.clean_weight)*quality(noisy)+a.anchor_weight*F.mse_loss(clean,anchor)+a.noisy_anchor_weight*F.mse_loss(noisy,anchor)
            if a.region_weight:
                region=batch['object_union'].cuda()
                edited=torch.tensor([mode!='background' for mode in batch['mode']],device=ref.device)[:,None,None,None]
                region=region*edited
                denominator=(3*region.sum()).clamp_min(1.)
                regional=lambda pred:(((pred-target).square()+.1*(pred-target).abs())*region).sum()/denominator
                loss=loss+a.region_weight*(a.clean_weight*regional(clean)+(1-a.clean_weight)*regional(noisy))
            if a.source_gate_weight:
                retention=batch['source_retention'].cuda()
                loss=loss+a.source_gate_weight*(a.clean_weight*F.binary_cross_entropy_with_logits(clean_gate,retention)+(1-a.clean_weight)*F.binary_cross_entropy_with_logits(gates['logits'],retention))
            if a.feature_consistency:
                consistency=sum(F.mse_loss(F.normalize(features[k].float(),dim=-1),F.normalize(v.float(),dim=-1))*v.shape[-1] for k,v in clean_features.items())/len(clean_features)
                loss=loss+a.feature_consistency*consistency
            if a.phase_tv:
                # Circular phase TV: equivalent phases separated by 2pi are equal.
                loss=loss+a.phase_tv*circular_phase_tv(model)
            if a.routing_consistency:
                loss=loss+a.routing_consistency*sum(F.kl_div(obj.last_routing['probabilities'].clamp_min(1e-8).log(),target,reduction='batchmean') for obj,target in zip((model.text,model.editor.bottleneck),clean_routing))/2
            if not bool(loss.isfinite()): raise RuntimeError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();step+=1
            if step%25==0:print(json.dumps(dict(step=step,loss=float(loss),strength=strength)),flush=True)
            if step%100==0 or step==a.steps:
                metrics=evaluate(val,validation_profiles)
                clean_val=metrics['clean']
                eligible=(clean_val['psnr_db']>=baseline_clean['psnr_db']-.2 and clean_val['ssim']>=baseline_clean['ssim']-.002)
                score=sum(metrics[p]['mse_0_1'] for p in a.selection_profiles)/len(a.selection_profiles)
                record=dict(step=step,metrics=metrics,eligible=eligible,score=score,alpha=optical_diagnostics(model))
                if eligible and score<best_score:best_score=score;selected=step;save('best_checkpoint.pt',step)
                history.append(record);write(a.output/'history.json',history);print(json.dumps(record),flush=True)
            if step>=a.steps:break
    save('last_checkpoint.pt',step)
    write(a.output/'report.json',dict(status='complete',selected_step=selected,baseline_val=baseline,history=history,
                                    counted_parameters=report['counted_parameters'],execution=execution))
    channel.restore()
    for handle in handles:handle.remove()


if __name__=='__main__':main()
