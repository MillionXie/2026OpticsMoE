"""Raw RGB/text joint optical conditioning recovery, without a generator update.

Unlike independent feature KD, language prefix states here receive the student's
actual visual merger/DeepStack outputs. Native retained TF and the readout head
are frozen. Only existing optical macros update, with TRAIN-only supervision.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import traceback
import torch
from PIL import Image
from .qwen3_scene_conditioning import QwenSDXLConditions
from .qwen3_scene_paired_train import new_qwen,electronic_condition_loss,TRAINReferenceContrast
from .qwen_vl_hybrid import directional_feature_loss,optimizer_groups
from .scene_paired_render import install_optical_frontend
from .scene_tail_distill import load_cache,complete_state
from .white_reference_scene_probe import GPU,digest,write


def forward_with_visual(model,inputs):
    """Keep both native outputs attached to the same differentiable forward."""
    captured=[]
    hook=model.visual.register_forward_hook(lambda module,args,out:captured.append(out[0]))
    try:out=model(**inputs,use_cache=False,return_dict=True).last_hidden_state
    finally:hook.remove()
    if len(captured)!=1:raise ValueError('Exactly one native visual forward required')
    return out,captured[0]


def recover(a):
    parent,ready=complete_state(a.generator);payload,cache_ready=load_cache(a.student_cache)
    teacher,teacher_ready=load_cache(a.teacher_cache)
    assert not teacher.get('optical_frontend') and payload.get('TRAIN_electronic_conditions')
    assert parent['pair_sha256']==payload['pair_sha256']==teacher['pair_sha256']
    qwen,processor=new_qwen(a);bound=install_optical_frontend(qwen,a.frontend,a.features,a.teacher_cache)
    assert bound['checkpoint_sha256']==parent['optical_frontend']['checkpoint_sha256']==payload['optical_frontend']['checkpoint_sha256']
    macros=qwen.model.hybrid
    cfg=json.loads((a.frontend/'execution.json').read_text())['config']
    cfg=dict(cfg,joint_lr=1e-5,phase_lr=1e-4,router_lr=1e-4)
    cfg['adapt_native_readouts']=bool(getattr(a,'adapt_native_readouts',False))
    head=QwenSDXLConditions(joint_image_context=parent.get('joint_image_context',False)).cuda().eval().requires_grad_(False)
    head.load_state_dict(parent['head'],strict=True)
    rows={r['input_key']:r for r in payload['records'] if r['split']=='train'}
    assert set(payload['TRAIN_electronic_conditions'])==set(rows)
    assert not set(rows)&{r['input_key'] for r in payload['records'] if r['split']=='dev'}
    groups={category:[r for r in rows.values() if r['category']==category] for category in ['lamp','pillow']}
    contrast=TRAINReferenceContrast(payload)
    def prepare(row):
        if row['split']!='train':raise ValueError('Only TRAIN is opened by this recovery stage')
        src=a.data/row['reference'];assert digest(src)==row['reference_sha256']
        with Image.open(src) as im:reference=im.convert('RGB').copy()
        text=processor.apply_chat_template([{'role':'user','content':[{'type':'image','image':reference},{'type':'text','text':row['prompt']}]}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        inputs=processor(text=[text],images=[reference],return_tensors='pt').to('cuda')
        assert inputs['image_grid_thw'].tolist()==[[1,16,16]] and inputs['attention_mask'].all()
        assert torch.equal(inputs['input_ids'].cpu(),payload['features'][row['input_key']]['input_ids'])
        return inputs
    # Measure initial raw/cache identity before any parameter update. No GT image
    # or mask is ever opened, including in these controls.
    controls=[]
    for category in groups:
        for row in groups[category][:3]:
            with torch.inference_mode():h,v=forward_with_visual(qwen.model,prepare(row))
            f=payload['features'][row['input_key']]
            assert torch.equal(h[0].cpu(),f['hidden']) and torch.equal(v.cpu(),f['visual'])
            controls.append(row['sample_id'])
    write(a.output/'initial_raw_identity.json',dict(complete=True,TRAIN_controls=controls,hidden_and_visual_exact=True))
    macros.requires_grad_(True)
    readouts=None;readout_initial=None
    if getattr(a,'adapt_native_readouts',False):
        from .scene_native_readouts import ReadoutMasters,snapshot
        readout_initial=snapshot(qwen);readouts=ReadoutMasters(qwen)
    allowed=set() if readouts is None else {id(p) for p in readouts.actual}
    assert all(not p.requires_grad for n,p in qwen.named_parameters() if '.hybrid.' not in n and id(p) not in allowed)
    optimization=torch.nn.ModuleDict({'optics':macros})
    if readouts is not None:optimization['readout_masters']=readouts
    direct_gradient=[]
    def loss_for(row,training,probe=False):
        h,v=forward_with_visual(qwen.model,prepare(row));assert h.shape[0]==1 and v.shape==(64,2048)
        predicted=head(h.float(),v.unsqueeze(0).float());f=teacher['features'][row['input_key']]
        condition=electronic_condition_loss(predicted,payload,row)
        if probe:
            target=payload['TRAIN_electronic_conditions'][row['input_key']]['text'].unsqueeze(0).cuda().float()
            text_loss=(predicted['text']-target).square().mean()
            parameters=[p for n,p in macros['vision'].named_parameters() if 'raw_phase' in n]
            gradients=torch.autograd.grad(text_loss,parameters,retain_graph=True,allow_unused=True)
            value=sum(g.abs().sum().item() for g in gradients if g is not None)
            assert math.isfinite(value) and value>0;direct_gradient.append(value)
        direction=directional_feature_loss(h[0],f['hidden'].cuda())+directional_feature_loss(v,f['visual'].cuda())
        loss=condition+.1*direction
        if training:loss=loss+.02*contrast.loss(predicted,row)+.02*sum(m.balance_loss() for m in macros.values())
        return loss
    def optimizer():return torch.optim.AdamW(optimizer_groups(optimization,cfg),weight_decay=.001)
    original={n:v.detach().clone() for n,v in macros.state_dict().items()};opt=optimizer();smoke=[]
    for i,row in enumerate([groups['lamp'][0],groups['pillow'][0],groups['lamp'][1],groups['pillow'][1]]):
        macros.train();opt.zero_grad()
        if readouts is not None:readouts.zero_native_grad()
        loss=loss_for(row,True,probe=True);loss.backward()
        if readouts is not None:
            readouts.collect();assert sum(p.grad.abs().sum().item() for p in readouts.master if p.grad is not None)>0
        grads={kind:sum(p.grad.abs().sum().item() for n,p in macros.named_parameters() if kind in n and p.grad is not None) for kind in ['raw_phase','raw_router_phase']}
        assert all(math.isfinite(g) and g>0 for g in grads.values());norm=torch.nn.utils.clip_grad_norm_(optimization.parameters(),1.,error_if_nonfinite=True);opt.step()
        if readouts is not None:readouts.copy_to_native()
        smoke.append(dict(step=i+1,loss=loss.item(),phase_gradients=grads,text_condition_to_visual_phase_gradient=direct_gradient[-1],gradient_norm=float(norm),
            native_readout_gradient_sum=0 if readouts is None else sum(p.grad.abs().sum().item() for p in readouts.master if p.grad is not None)))
    macros.load_state_dict(original,strict=True)
    if readouts is not None:
        from .scene_native_readouts import restore
        restore(qwen,readout_initial);readouts.reset_from_native()
    del original,opt;opt=optimizer();contrast=TRAINReferenceContrast(payload)
    write(a.output/'smoke_report.json',dict(complete=True,real_backward_steps=smoke,all_macro_state_reset_before_training=True,
        native_readouts_reset_before_training=readouts is not None,TRAIN_only=True,head_and_native_Qwen_TF_frozen=True,
        existing_native_readouts_trainable=readouts is not None))
    rng=random.Random(1042);history=[];best=math.inf;fixed=[r for category in groups for r in groups[category][:6]]
    for step in range(1,a.steps+1):
        macros.train();opt.zero_grad()
        if readouts is not None:readouts.zero_native_grad()
        # Accumulate across both categories at one parameter version; EMA soft
        # balance is diagnostic only, never claimed to prove hard-route diversity.
        for category in ['lamp','pillow']:
            loss=loss_for(rng.choice(groups[category]),True)/2;assert torch.isfinite(loss);loss.backward()
        if readouts is not None:readouts.collect()
        norm=torch.nn.utils.clip_grad_norm_(optimization.parameters(),1.,error_if_nonfinite=True);opt.step()
        if readouts is not None:readouts.copy_to_native()
        for g in opt.param_groups:g['lr']=g['base_lr']*(.1+.9*.5*(1+math.cos(math.pi*step/a.steps)))
        if step%10==0:write(a.output/'progress.json',dict(stage='actual_coupled_RGB_text_optical_condition_training',step=step,total=a.steps,last_loss=loss.item()*2))
        if step%100==0 or step==a.steps:
            macros.eval()
            with torch.no_grad():score=statistics_mean([loss_for(row,False).item() for row in fixed])
            history.append(dict(step=step,fixed_TRAIN_condition_feature_loss=score));write(a.output/'history.json',history)
            source=torch.load(a.frontend/'best_checkpoint.pt',map_location='cpu',weights_only=False)
            source.update(step=step,config=cfg,macros={n:v.detach().cpu() for n,v in macros.state_dict().items()},source_commit=os.environ['T12_SOURCE_COMMIT'],raw_coupled_condition_recovery=True)
            if readouts is not None:
                from .scene_native_readouts import snapshot
                source['native_readouts']=snapshot(qwen)
            torch.save(source,a.output/'last_checkpoint.pt')
            if score<best:best=score;torch.save(source,a.output/'best_checkpoint.pt')
    write(a.output/'report.json',dict(complete=True,stage='distill',steps=a.steps,best_sha256=digest(a.output/'best_checkpoint.pt'),best_fixed_TRAIN_loss=best,
        source_commit=os.environ['T12_SOURCE_COMMIT'],parent_frontend_sha256=bound['checkpoint_sha256'],frozen_generator_head_sha256=ready['best_sha256'],
        teacher_cache_sha256=teacher_ready['cache_sha256'],student_cache_sha256=cache_ready['cache_sha256'],TRAIN_contexts=len(rows),
        only_existing_optical_macros_trainable=readouts is None,raw_visual_to_language_gradient_path=True,
        head_UNet_VAE_retained_Qwen_frozen=readouts is None,head_UNet_VAE_retained_Qwen_TF_frozen=True,
        existing_native_readouts_also_trainable=readouts is not None,
        native_readout_parameters_trained=0 if readouts is None else sum(p.numel() for p in readouts.master),
        DEV_or_TEST_forward_performed=False,GT_images_or_masks_opened=False,extra_deployment_parameters=0,quality_or_speed_accepted=False,TEST_opened=False))
    write(a.output/'effective_config.json',cfg)


def statistics_mean(values):return sum(values)/len(values)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['generator','student-cache','teacher-cache','data','qwen','frontend','features','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--steps',type=int,default=400);p.add_argument('--adapt-native-readouts',action='store_true');a=p.parse_args();a.qwen_side=256
    assert a.steps>0 and not a.output.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')==GPU and len(os.environ.get('T12_SOURCE_COMMIT',''))==40
    a.output.mkdir(parents=True);torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.8,0);torch.manual_seed(1042)
    write(a.output/'execution.json',dict(argv=sys.argv,source_commit=os.environ['T12_SOURCE_COMMIT'],source_sha256=digest(Path(__file__)),pid=os.getpid()))
    try:recover(a)
    except Exception:write(a.output/'failure.json',dict(traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
