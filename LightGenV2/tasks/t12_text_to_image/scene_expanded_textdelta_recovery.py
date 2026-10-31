"""TRAIN-only text/pooled correction in the same middle model, CPU only.

Same-reference instruction differences are matched to the selected electronic
head. Image head, shared projection, UNet LoRA, Qwen, phases and VAE stay exact.
No additional inference branch, parser, parameter or denoising step is added.
"""
import argparse,copy,hashlib,json,math,os,random,sys,time,traceback
from collections import defaultdict
from pathlib import Path
from .scene_instruction_data import sha,write
from .scene_R2_middle_protocol import F6


def groups_for(records):
    groups=defaultdict(dict)
    for row in records:
        if row['split']!='train':raise ValueError('TRAIN records only')
        old=groups[row['reference_id']].setdefault(row['input_key'],row)
        if old['prompt']!=row['prompt']:raise ValueError('Conflicting actual input key')
    result=[]
    for _,items in sorted(groups.items()):
        ordered=sorted(items.values(),key=lambda r:(int(r['condition_id']),r['input_key']))
        for start in range(0,len(ordered),3):
            if len(ordered)<=3:part=ordered
            else:part=[ordered[(start+j)%len(ordered)]for j in range(3)]
            result.append(part)
    return result


def text_forward(head,features):
    import torch
    length=max(len(r['hidden'])for r in features)
    hidden=torch.zeros(len(features),length,2048);padding=torch.ones(len(features),length,dtype=torch.bool)
    for i,row in enumerate(features):hidden[i,:len(row['hidden'])]=row['hidden'].float();padding[i,:len(row['hidden'])]=False
    text,query=head.text(head.project(hidden),padding)
    return dict(text=text,pooled=head.pooled(query.mean(1)))


def delta_loss(pred,target):
    import torch
    terms={}
    for field in ['text','pooled']:
        p=pred[field].float();t=target[field].float().detach()
        if p.shape!=t.shape or not 1<=len(p)<=3:raise ValueError('One to three actual same-reference requests required')
        terms[field+'_absolute']=(p-t).square().mean()/t.square().mean().clamp_min(1e-6)
        delta=t-t.mean(0,keepdim=True);variance=delta.square().mean()
        if len(p)>1 and variance.item()<=1e-10:raise ValueError('Teacher has no distinguishable instruction variation')
        terms[field+'_counterfactual']=((p-p.mean(0,keepdim=True)-delta).square().mean()/variance)if len(p)>1 else p.sum()*0
    loss=terms['text_absolute']+.5*terms['text_counterfactual']+.25*terms['pooled_absolute']+.1*terms['pooled_counterfactual']
    if not torch.isfinite(loss):raise ValueError('Nonfinite condition objective')
    return loss,terms


def fit(a):
    import torch
    from .qwen3_scene_conditioning import QwenSDXLConditions
    from .scene_tail_distill import complete_state,load_cache
    if a.output.exists()or os.environ.get('CUDA_VISIBLE_DEVICES')!=''or len(os.environ.get('T12_SOURCE_COMMIT',''))!=40:
        raise ValueError('New published CPU-only correction required')
    torch.set_num_threads(4);torch.manual_seed(20261101)
    parent,pr=complete_state(a.parent);payload,cr=load_cache(a.cache)
    if (pr['best_sha256']!=a.parent_sha256 or payload['condition_teacher_sha256']!=a.teacher_sha256
        or parent['pair_sha256']!=payload['pair_sha256'] or not parent.get('no_validation_test_PT_selection')
        or parent.get('joint_image_context',False) or parent['optical_frontend']['checkpoint_sha256']!=F6
        or payload['optical_frontend']['checkpoint_sha256']!=F6
        or not pr['compression_training_performed']or parent['sampling_contract']['steps']!=8
        or parent['depth_compression']['profile']['UNet_deep_keep']!=9):
        raise ValueError('Exact expanded single-EG64-block8-step parent and selected E condition bank required')
    records=payload['records'];groups=groups_for(records);keys={r['input_key']for r in records}
    if len(keys)!=624 or set(payload['TRAIN_electronic_conditions'])!=keys:raise ValueError('All actual624 TRAIN condition targets required')
    targets=payload['TRAIN_electronic_conditions'];a.output.mkdir()
    write(a.output/'execution.json',dict(argv=sys.argv,pid=os.getpid(),source_commit=os.environ['T12_SOURCE_COMMIT'],source_sha256=sha(__file__),CPU_only=True))
    def new():
        head=QwenSDXLConditions(joint_image_context=False).float();head.load_state_dict(parent['head'],strict=True);head.requires_grad_(False)
        for module in [head.text,head.pooled]:module.requires_grad_(True)
        head.train();params=[p for p in head.parameters()if p.requires_grad]
        return head,torch.optim.AdamW(params,lr=a.lr,weight_decay=.01)
    def sample(head,group):
        pred=text_forward(head,[payload['features'][r['input_key']]for r in group])
        target={name:torch.stack([targets[r['input_key']][name]for r in group])for name in ['text','pooled']}
        return delta_loss(pred,target)
    def measure(head):
        head.eval();total=[];parts=defaultdict(list)
        with torch.no_grad():
            for group in groups:
                loss,terms=sample(head,group);total.append(float(loss))
                for name,value in terms.items():parts[name].append(float(value))
        head.train();return sum(total)/len(total),{name:sum(v)/len(v)for name,v in parts.items()}
    head,opt=new();before,initial_terms=measure(head);smoke=[]
    for i in range(4):
        loss,_=sample(head,groups[i]);opt.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_([p for p in head.parameters()if p.requires_grad],1.,error_if_nonfinite=True)
        if not torch.isfinite(loss)or norm<=0:raise ValueError('Invalid actual CPU backward')
        opt.step();smoke.append(dict(step=i+1,loss=float(loss),gradient_norm=float(norm)))
    def frozen_exact(model):
        return all(torch.equal(value,parent['head'][name])for name,value in model.state_dict().items()if not name.startswith(('text.','pooled.')))
    assert frozen_exact(head)
    write(a.output/'smoke_report.json',dict(complete=True,real_CPU_backward=smoke,shared_projection_image_head_exact=True,smoke_weights_discarded=True,initial_TRAIN_loss=before,initial_components=initial_terms))
    head,opt=new();rng=random.Random(20261101);order=[];best=math.inf;history=[];seen=set();began=time.monotonic()
    for step in range(1,a.steps+1):
        if not order:order=list(range(len(groups)));rng.shuffle(order)
        group=groups[order.pop()];seen.update(r['input_key']for r in group)
        opt.zero_grad();loss,_=sample(head,group);loss.backward();torch.nn.utils.clip_grad_norm_([p for p in head.parameters()if p.requires_grad],1.,error_if_nonfinite=True);opt.step()
        if step%20==0:write(a.output/'progress.json',dict(stage='CPU_same_reference_selected_E_text_delta_KD',step=step,total=a.steps,loss=float(loss),seen_TRAIN_contexts=len(seen)))
        if step%200==0 or step==a.steps:
            score,parts=measure(head);assert frozen_exact(head);history.append(dict(step=step,TRAIN_loss=score,components=parts,seconds=time.monotonic()-began));write(a.output/'history.json',history)
            state=copy.deepcopy(parent);state['head']={name:value.detach().cpu().clone()for name,value in head.state_dict().items()}
            state['depth_compression']=dict(state['depth_compression'],parent_head_unchanged=False)
            state['step']=step;state['source_commit']=os.environ['T12_SOURCE_COMMIT']
            state['condition_correction']=dict(schema='expanded_same_reference_selected_E_text_delta_v1',parent_sha256=pr['best_sha256'],teacher_sha256=a.teacher_sha256,steps=step,TRAIN_loss=score,image_shared_projection_UNet_unchanged=True,new_inference_parameters=0)
            assert all(torch.equal(v,parent['unet_lora'][n])for n,v in state['unet_lora'].items())
            torch.save(state,a.output/'last_checkpoint.pt')
            if score<best:best=score;torch.save(state,a.output/'best_checkpoint.pt')
    write(a.output/'report.json',dict(complete=True,source_commit=os.environ['T12_SOURCE_COMMIT'],best_sha256=sha(a.output/'best_checkpoint.pt'),last_sha256=sha(a.output/'last_checkpoint.pt'),
        compression_training_performed=True,depth_compression=state['depth_compression'],sampling_steps=8,epochs=None,actual_paired_updates=0,condition_alignment_updates=a.steps,inherited_paired_updates=pr['actual_paired_updates'],
        training_update_kind='existing text and pooled condition heads only; no denoising updates',TRAIN_contexts=624,actual_gradient_TRAIN_contexts=len(seen),TRAIN_references=len({r['reference_id']for r in records}),
        TRAIN_groups=len(groups),single_request_groups=sum(len(g)==1 for g in groups),initial_TRAIN_loss=before,best_TRAIN_loss=best,
        best_selection='All reused TRAIN condition groups, not validation or final model selection',parent_checkpoint_sha256=pr['best_sha256'],teacher_sha256=a.teacher_sha256,student_cache_sha256=cr['cache_sha256'],
        existing_text_pooled_trainable_parameters=sum(p.numel()for p in head.parameters()if p.requires_grad),frozen_shared_projection_image_head_UNet_exact=True,Qwen_phases_VAE_frozen=True,
        extra_deployment_parameters=0,TEST_used_for_gradient=False,validation_set=False,final_TEST_PT_selection_pending=True,quality_or_speed_accepted=False,CPU_only=True))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['parent','cache','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--parent-sha256',required=True);p.add_argument('--teacher-sha256',required=True)
    p.add_argument('--steps',type=int,default=800);p.add_argument('--lr',type=float,default=5e-5);a=p.parse_args()
    if a.steps!=800 or a.lr!=5e-5:raise ValueError('Fixed800-step text delta alignment recipe required')
    try:fit(a)
    except BaseException:
        if a.output.exists():write(a.output/'failure.json',dict(traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
