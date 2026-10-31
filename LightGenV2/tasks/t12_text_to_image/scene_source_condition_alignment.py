"""TRAIN-only source-feature correction of the existing Qwen image readout.

No scene target is used. Four distinct white references of the same category
teach the frozen native image encoder's common and source-dependent features.
All text/pooled/projection weights and the complete decoder remain unchanged.
"""
import argparse,copy,hashlib,json,os,random,time
from pathlib import Path
from collections import defaultdict
import torch
from .qwen3_scene_conditioning import QwenSDXLConditions
from .scene_instruction_data import sha,write


def source_loss(pred,target):
    p=pred.float();t=target.float().detach()
    if len(p)!=4 or p.shape!=t.shape:raise ValueError('Four distinct same-category sources required')
    delta=t-t.mean(0,keepdim=True);variance=delta.square().mean()
    if variance.item()<=1e-10:raise ValueError('Source-dependent teacher variation required')
    absolute=(p-t).square().mean()/t.square().mean().clamp_min(1e-6)
    relative=(p-p.mean(0,keepdim=True)-delta).square().mean()/variance
    return absolute+.5*relative,dict(absolute=absolute,source_delta=relative)


def fit(a):
    from .scene_instruction_balanced_data import assert_admission_current
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=''or a.output.exists()or len(os.environ.get('T12_SOURCE_COMMIT',''))!=40:raise ValueError('Published new CPU-only run required')
    torch.set_num_threads(4);assert_admission_current(a.data)
    dr=json.loads((a.data/'report.json').read_text());cr=json.loads((a.cache/'report.json').read_text());pr=json.loads((a.parent/'report.json').read_text())
    if not(dr['eligible_for_training']and cr['complete']and pr['complete'])or sha(a.data/'pairs.json')!=dr['pairs_sha256']or sha(a.cache/'paired_cache.pt')!=cr['cache_sha256']or sha(a.parent/'best_checkpoint.pt')!=pr['best_sha256']:raise ValueError('Actual completed assets required')
    payload=torch.load(a.cache/'paired_cache.pt',map_location='cpu',weights_only=False);parent=torch.load(a.parent/'best_checkpoint.pt',map_location='cpu',weights_only=False)
    if parent['variant']!='qwen'or parent.get('joint_image_context',False)or not parent.get('condition_correction')or parent['cache_sha256']!=cr['cache_sha256']or not(parent['pair_sha256']==payload['pair_sha256']==dr['pairs_sha256']):raise ValueError('Source-only image branch and actual corrected electronic parent required')
    unique={r['reference_sha256']:r for r in payload['records']if r['split']=='train'};categories={k:[r for r in unique.values()if r['category']==k]for k in ['lamp','pillow']}
    if len(unique)!=44 or[len(categories[k])for k in ['lamp','pillow']]!=[20,24]:raise ValueError('All44 TRAIN sources required')
    a.output.mkdir();rng=random.Random(20261101)
    def new():
        head=QwenSDXLConditions().float();head.load_state_dict(parent['head'],strict=True);head.requires_grad_(False);head.image.requires_grad_(True)
        return head,torch.optim.AdamW(head.image.parameters(),lr=5e-5,weight_decay=.01)
    def loss_for(head,rows):
        if any(r['split']!='train'for r in rows)or len({r['reference_sha256']for r in rows})!=4 or len({r['category']for r in rows})!=1:raise ValueError('Four unique same-category TRAIN references required')
        visual=torch.stack([payload['features'][r['input_key']]['visual'].float()for r in rows]);target=torch.stack([payload['images'][r['reference_sha256']].float()for r in rows]);pred,_=head.image(head.project(visual));return source_loss(pred,target)
    head,opt=new();smoke=[]
    for step in range(4):
        loss,_=loss_for(head,rng.sample(categories[['lamp','pillow'][step%2]],4));opt.zero_grad();loss.backward();grad=sum(p.grad.abs().sum().item()for p in head.image.parameters()if p.grad is not None)
        if not torch.isfinite(loss)or grad<=0 or any(not torch.isfinite(p.grad).all()for p in head.image.parameters()if p.grad is not None):raise ValueError('Invalid actual backward')
        opt.step();smoke.append(dict(step=step+1,loss=loss.item(),absolute_gradient_sum=grad))
    assert all(torch.equal(v,parent['head'][k])for k,v in head.state_dict().items()if not k.startswith('image.'))
    write(a.output/'smoke_report.json',dict(complete=True,real_TRAIN_backward=smoke,smoke_weights_discarded=True,text_pooled_projection_exact=True));head,opt=new();rng=random.Random(20261101);best=float('inf');history=[];began=time.monotonic()
    for step in range(1,601):
        loss,parts=loss_for(head,rng.sample(categories[['lamp','pillow'][step%2]],4));opt.zero_grad();loss.backward()
        if not torch.isfinite(loss)or any(not torch.isfinite(p.grad).all()for p in head.image.parameters()if p.grad is not None):raise ValueError('Nonfinite loss/gradient')
        torch.nn.utils.clip_grad_norm_(head.image.parameters(),1.);opt.step()
        if step%40==0:write(a.output/'progress.json',dict(stage='CPU_TRAIN_only_source_feature_alignment',step=step,total=600,loss=loss.item(),elapsed_seconds=time.monotonic()-began))
        if step%200==0:
            values=[];components=defaultdict(list)
            with torch.inference_mode():
                for rows in categories.values():
                    rows=sorted(rows,key=lambda r:r['reference_sha256'])
                    for start in range(0,len(rows),4):
                        value,parts=loss_for(head,rows[start:start+4]);values.append(value.item())
                        for k,v in parts.items():components[k].append(v.item())
            mean=sum(values)/len(values);history.append(dict(step=step,TRAIN_loss=mean,components={k:sum(v)/len(v)for k,v in components.items()}));write(a.output/'history.json',history);state=copy.deepcopy(parent);state['head']={k:v.detach().cpu().clone()for k,v in head.state_dict().items()}
            assert all(torch.equal(v,parent['head'][k])for k,v in state['head'].items()if not k.startswith('image.'))
            state['image_condition_correction']=dict(schema='same_category_source_delta_TRAIN_alignment_v1',steps=step,parent_sha256=pr['best_sha256'],TRAIN_loss=mean,TRAIN_RGB_sha256s=sorted(unique),text_pooled_projection_unchanged=True,UNet_LoRA_unchanged=True,new_inference_parameters=0,source_commit=os.environ['T12_SOURCE_COMMIT']);torch.save(state,a.output/'last_checkpoint.pt')
            if mean<best:best=mean;torch.save(state,a.output/'best_checkpoint.pt')
    write(a.output/'report.json',dict(complete=True,source_commit=os.environ['T12_SOURCE_COMMIT'],schema='scene_source_condition_correction_v1',variant='qwen',best_sha256=sha(a.output/'best_checkpoint.pt'),parent_sha256=pr['best_sha256'],correction_steps=600,actual_paired_updates=0,inherited_paired_updates=parent['step'],new_inference_parameters=0,TRAIN_references=44,TRAIN_only=True,DEV_or_TEST_used_for_selection=False,scene_target_pixels_or_latents_used_for_loss=False,native_image_encoder_is_TRAIN_teacher_only=True,selected_by='all44 TRAIN source condition loss only, not generated image quality',quality_accepted=False,inference_graph_unchanged=True,CPU_only=True))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['data','cache','parent','output']:p.add_argument('--'+k,type=Path,required=True)
    fit(p.parse_args())


if __name__=='__main__':main()
