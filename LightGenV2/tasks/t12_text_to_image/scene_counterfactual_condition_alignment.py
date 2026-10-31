"""TRAIN-only correction of existing Qwen text conditions, no new inference graph.

Same-reference triplets separate instruction-dependent CLIP token variation
from common token components. Frozen native encoders supplied TRAIN features;
no generated image/DEV/TEST score is used for fitting or checkpoint selection.
Image head, shared projection, Qwen, UNet LoRA and VAE remain unchanged.
"""
import argparse,copy,hashlib,json,os,random,time
from pathlib import Path
from collections import defaultdict
import torch
from .qwen3_scene_conditioning import QwenSDXLConditions
from .scene_instruction_data import sha,write


def correction_loss(pred,target):
    terms={}
    for field in ['text','pooled']:
        p=pred[field].float();t=target[field].float().detach()
        if len(p)!=3 or p.shape!=t.shape:raise ValueError('Three same-reference distinct conditions required')
        delta=t-t.mean(0,keepdim=True);native_variance=delta.square().mean()
        if native_variance.item()<=1e-10:raise ValueError('Distinct native text conditions required')
        terms[field+'_absolute']=(p-t).square().mean()/t.square().mean().clamp_min(1e-6)
        terms[field+'_counterfactual']=(p-p.mean(0,keepdim=True)-delta).square().mean()/native_variance
    return terms['text_absolute']+.5*terms['text_counterfactual']+.25*terms['pooled_absolute']+.1*terms['pooled_counterfactual'],terms


def text_forward(head,features):
    length=max(len(r['hidden'])for r in features);hidden=torch.zeros(3,length,2048);padding=torch.ones(3,length,dtype=torch.bool)
    for i,r in enumerate(features):hidden[i,:len(r['hidden'])]=r['hidden'].float();padding[i,:len(r['hidden'])]=False
    context=head.project(hidden);text,query=head.text(context,padding)
    return dict(text=text,pooled=head.pooled(query.mean(1)))


def fit(a):
    from .scene_instruction_balanced_data import assert_admission_current
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=''or a.output.exists():raise ValueError('New CPU-only run required')
    if len(os.environ.get('T12_SOURCE_COMMIT',''))!=40 or a.steps!=800:raise ValueError('Published fixed800-step diagnostic required')
    torch.set_num_threads(4);assert_admission_current(a.data)
    dr=json.loads((a.data/'report.json').read_text());cr=json.loads((a.cache/'report.json').read_text());pr=json.loads((a.parent/'report.json').read_text())
    if not(dr['eligible_for_training']and cr['complete']and pr['complete']):raise ValueError('Completed actual data/cache/parent required')
    if sha(a.data/'pairs.json')!=dr['pairs_sha256']or sha(a.cache/'paired_cache.pt')!=cr['cache_sha256']or sha(a.parent/'best_checkpoint.pt')!=pr['best_sha256']:raise ValueError('Input identity mismatch')
    payload=torch.load(a.cache/'paired_cache.pt',map_location='cpu',weights_only=False);parent=torch.load(a.parent/'best_checkpoint.pt',map_location='cpu',weights_only=False)
    if parent['variant']!='qwen'or parent.get('optical_frontend')or parent.get('depth_compression')or parent['step']<=0:raise ValueError('Full electronic Qwen parent required')
    if not(parent['pair_sha256']==payload['pair_sha256']==dr['pairs_sha256'])or parent['cache_sha256']!=cr['cache_sha256']:raise ValueError('Unrelated parent/cache/data')
    groups=defaultdict(dict)
    for r in payload['records']:
        if r['split']=='train':groups[r['reference_id']][r['condition_id']]=r
    if len(groups)!=44 or any(len(r)!=3 for r in groups.values()):raise ValueError('All44 TRAIN source triplets required')
    triples=[sorted(g.values(),key=lambda r:r['condition_id'])for _,g in sorted(groups.items())]
    train_keys=sorted({r['input_key']for g in triples for r in g});a.output.mkdir()
    def new():
        head=QwenSDXLConditions(joint_image_context=parent.get('joint_image_context',False)).float();head.load_state_dict(parent['head'],strict=True);head.requires_grad_(False)
        for part in [head.text,head.pooled]:part.requires_grad_(True)
        head.train();optimizer=torch.optim.AdamW([p for p in head.parameters()if p.requires_grad],lr=5e-5,weight_decay=.01)
        return head,optimizer
    def loss_for(head,group):
        features=[payload['features'][r['input_key']]for r in group]
        targets={k:torch.stack([payload['captions'][r['native_caption']][k].float()for r in group])for k in ['text','pooled']}
        return correction_loss(text_forward(head,features),targets)
    head,opt=new();smoke=[]
    for step in range(4):
        loss,_=loss_for(head,triples[step]);opt.zero_grad();loss.backward();gradient=sum(p.grad.abs().sum().item()for p in head.parameters()if p.grad is not None)
        if not torch.isfinite(loss)or gradient<=0 or any(not torch.isfinite(p.grad).all()for p in head.parameters()if p.grad is not None):raise ValueError('Invalid real CPU backward')
        opt.step();smoke.append(dict(step=step+1,loss=loss.item(),absolute_gradient_sum=gradient))
    if any(not torch.equal(v,parent['head'][k])for k,v in head.state_dict().items()if not k.startswith(('text.','pooled.'))):raise ValueError('Frozen inference components changed')
    write(a.output/'smoke_report.json',dict(complete=True,real_TRAIN_backward=smoke,unselected_components_exact=True,smoke_weights_discarded=True))
    head,opt=new();rng=random.Random(20261031);best=float('inf');history=[];began=time.monotonic()
    for step in range(1,a.steps+1):
        group=triples[rng.randrange(len(triples))];loss,terms=loss_for(head,group);opt.zero_grad();loss.backward()
        if not torch.isfinite(loss)or any(not torch.isfinite(p.grad).all()for p in head.parameters()if p.grad is not None):raise ValueError('Nonfinite TRAIN loss/gradient')
        torch.nn.utils.clip_grad_norm_([p for p in head.parameters()if p.requires_grad],1.);opt.step()
        if step%40==0:write(a.output/'progress.json',dict(stage='CPU_TRAIN_only_same_reference_condition_alignment',step=step,total=a.steps,loss=loss.item(),elapsed_seconds=time.monotonic()-began))
        if step%200==0:
            head.eval();values=[];parts=defaultdict(list)
            with torch.inference_mode():
                for group in triples:
                    value,terms=loss_for(head,group);values.append(value.item())
                    for k,v in terms.items():parts[k].append(v.item())
            mean=sum(values)/len(values);history.append(dict(step=step,TRAIN_loss=mean,components={k:sum(v)/len(v)for k,v in parts.items()}));write(a.output/'history.json',history)
            state=copy.deepcopy(parent);state['head']={k:v.detach().cpu().clone()for k,v in head.state_dict().items()}
            assert all(torch.equal(v,parent['head'][k])for k,v in state['head'].items()if not k.startswith(('text.','pooled.')))
            state['condition_correction']=dict(schema='same_reference_token_delta_TRAIN_alignment_v1',steps=step,parent_sha256=pr['best_sha256'],TRAIN_loss=mean,TRAIN_input_keys_sha256=hashlib.sha256('\n'.join(train_keys).encode()).hexdigest(),shared_projection_and_image_head_unchanged=True,UNet_LoRA_unchanged=True,new_inference_parameters=0,source_commit=os.environ['T12_SOURCE_COMMIT'])
            torch.save(state,a.output/'last_checkpoint.pt')
            if mean<best:best=mean;torch.save(state,a.output/'best_checkpoint.pt')
            head.train()
    write(a.output/'report.json',dict(complete=True,source_commit=os.environ['T12_SOURCE_COMMIT'],variant='qwen',schema='scene_counterfactual_condition_correction_v1',best_sha256=sha(a.output/'best_checkpoint.pt'),parent_sha256=pr['best_sha256'],correction_steps=a.steps,actual_paired_updates=0,inherited_paired_updates=parent['step'],new_inference_parameters=0,TRAIN_references=44,TRAIN_distinct_conditions=132,TRAIN_only=True,DEV_or_TEST_used_for_selection=False,scene_target_pixels_or_latents_used_for_loss=False,native_conditions_are_TRAIN_teacher_only=True,selected_by='all44 TRAIN condition loss only, not image quality',quality_accepted=False,inference_graph_unchanged=True,CPU_only=True))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['data','cache','parent','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--steps',type=int,default=800);fit(p.parse_args())


if __name__=='__main__':main()
