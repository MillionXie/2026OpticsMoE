"""Native optical Qwen front-end KD for the two-sentence SDXL scene model.

Feature targets are frozen complete Qwen activations, never synthetic scene GT.
This stage does not establish comparable generation or physical 2x speed.
"""
import argparse,hashlib,json,math,os,random,sys,traceback
from pathlib import Path
import torch
from torch import nn
from PIL import Image
from .qwen3_scene_paired_train import new_qwen,raw_scene_features
from .qwen3_scene_conditioning import qwen_asset_manifest
from .qwen_vl_hybrid import OpticalMacro,directional_feature_loss,optimizer_groups,hybrid_config
from .qwen_vl_postfree import QuietFusion
from .white_reference_scene_probe import GPU,digest,write


def input_rows(data):
    report=json.loads((data/'report.json').read_text())
    assert report['complete'] and report['role_disjoint'] and not report['TEST_opened']
    assert digest(data/'pairs.json')==report['pairs_sha256']
    train=[r for r in json.loads((data/'pairs.json').read_text()) if r['split']=='train']
    dev=[dict(r,split='dev') for r in json.loads((data/'inference_dev_inputs.json').read_text())]
    seen={};owners={}
    for row in train+dev:
        key=(row['reference_sha256'],row['prompt'])
        if owners.setdefault(row['reference_sha256'],row['split'])!=row['split']:raise ValueError('Cross-role RGB')
        if key not in seen:seen[key]={k:row[k] for k in ['sample_id','reference','reference_sha256','prompt','split']}
    return list(seen.values())


def cache(a):
    write(a.output/'model_assets.json',qwen_asset_manifest(a.qwen))
    qwen,processor=new_qwen(a);rows=input_rows(a.data);captured={};handles=[];records=[];vision_pool={}
    def pre(key):
        def hook(module,args):captured[key]=args[0].detach().cpu().clone()
        return hook
    def post(key):
        def hook(module,args,out):captured[key]=(out[0] if isinstance(out,tuple) else out).detach().cpu().clone()
        return hook
    handles.append(qwen.model.visual.blocks[6].register_forward_pre_hook(pre('vision_input')))
    handles.append(qwen.model.language_model.layers[4].register_forward_pre_hook(pre('language_input')))
    for i in [11,23]:handles.append(qwen.model.visual.blocks[i].register_forward_hook(post('vision_'+str(i))))
    for i in [13,27]:handles.append(qwen.model.language_model.layers[i].register_forward_hook(post('language_'+str(i))))
    try:
        for i,row in enumerate(rows):
            src=a.data/row['reference'];assert digest(src)==row['reference_sha256'];captured.clear()
            with Image.open(src) as im:h,v,ids=raw_scene_features(qwen,processor,im.convert('RGB'),row['prompt'],256)
            record=dict(captured)
            for key in list(record):
                if key.startswith('vision_'):record[key]=record[key].unsqueeze(0)
            assert record['vision_input'].shape==(1,256,1024)
            assert record['language_input'].shape==(1,len(h),2048)
            pool=vision_pool.setdefault(row['reference_sha256'],{k:x for k,x in record.items() if k.startswith('vision_')})
            for k,value in pool.items():
                assert torch.equal(record[k],value);record[k]=value
            record.update({**row,'input_fingerprint':hashlib.sha256(src.read_bytes()+b'\0'+row['prompt'].encode()+b'\0Qwen256').hexdigest()})
            records.append(record)
            write(a.output/'progress.json',{'stage':'complete_Qwen_native_intermediate_cache','done':i+1,'total':len(rows)})
    finally:
        for handle in handles:handle.remove()
    torch.save({'records':records,'pairs_sha256':digest(a.data/'pairs.json')},a.output/'teacher_features.pt')
    write(a.output/'report.json',{'complete':True,'source_commit':os.environ['T12_SOURCE_COMMIT'],'n':len(records),
        'cache_sha256':digest(a.output/'teacher_features.pt'),'Qwen_config_sha256':digest(a.qwen/'config.json'),
        'vision_layers':24,'language_layers':28,'Qwen_input_side':256,'premerger_vision_tokens':256,
        'merger_tokens':64,'only_RGB_text_model_inputs':True,'teacher_is_frozen_public_Qwen_not_generated_images':True,
        'TEST_opened':False,'quality_or_speed_accepted':False})


def distill(a,cfg):
    report=json.loads((a.features/'report.json').read_text());assert report['complete'] and not report['TEST_opened']
    assert digest(a.features/'teacher_features.pt')==report['cache_sha256']
    records=torch.load(a.features/'teacher_features.pt',map_location='cpu',weights_only=False)['records']
    train=[r for r in records if r['split']=='train'];dev=[r for r in records if r['split']=='dev'];assert train and dev
    torch.manual_seed(cfg['seed'])
    macros=nn.ModuleDict({m:OpticalMacro(m,w,1,cfg['residual_width']) for m,w in [('vision',1024),('language',2048)]}).cuda()
    for macro in macros.values():
        for i,fusion in enumerate(macro.fusions):
            quiet=QuietFusion(.65,.6,.9,1e-6).cuda();quiet.load_state_dict(fusion.state_dict());macro.fusions[i]=quiet
    opt=torch.optim.AdamW(optimizer_groups(macros,cfg),weight_decay=.001);rng=random.Random(cfg['seed'])
    def loss_for(row):
        loss=0
        for modality,ends in [('vision',[11,23]),('language',[13,27])]:
            macro=macros[modality];macro(row[modality+'_input'].cuda())
            loss=loss+sum(directional_feature_loss(x,row[modality+'_'+str(j)].cuda()) for x,j in zip(macro.last_stages,ends))/2
            loss=loss+cfg['balance_weight']*macro.balance_loss()
        return loss
    best=math.inf;history=[];steps=4 if a.stage=='smoke' else cfg['feature_steps']
    for step in range(1,steps+1):
        macros.train();opt.zero_grad();loss=loss_for(rng.choice(train));loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(macros.parameters(),1.,error_if_nonfinite=True)
        assert torch.isfinite(loss) and torch.isfinite(norm)
        if step==4:
            grads={kind:sum(p.grad.float().abs().sum().item() for n,p in macros.named_parameters() if kind in n and p.grad is not None) for kind in ['raw_phase','raw_router_phase']}
            assert all(math.isfinite(v) and v>0 for v in grads.values()),grads
            write(a.output/'smoke_report.json',{'complete':True,'real_backward_steps':4,'finite_nonzero_phase_and_router_gradients':grads,'TRAIN_only_gradients':True})
        opt.step()
        if step%10==0:write(a.output/'progress.json',{'stage':'LG_frontend_directional_distillation','step':step,'total':steps,'TRAIN_loss':loss.item()})
        if step%cfg['eval_every']==0 or step==steps:
            macros.eval()
            with torch.no_grad():score=sum(loss_for(row).item() for row in dev)/len(dev)
            entry={'step':step,'TRAIN_sample_loss':loss.item(),'DEV_directional_loss':score};history.append(entry);write(a.output/'history.json',history)
            state={'schema':'qwen3_scene_optical_frontend_v1','step':step,'config':cfg,'macros':macros.state_dict(),
                'feature_cache_sha256':report['cache_sha256'],'source_commit':os.environ['T12_SOURCE_COMMIT'],'postfree':True}
            torch.save(state,a.output/'last_checkpoint.pt')
            if score<best:best=score;torch.save(state,a.output/'best_checkpoint.pt')
    routes=[]
    macros.eval()
    with torch.no_grad():
        for row in dev:
            loss_for(row);routes.append({'sample_id':row['sample_id'],**{m:x.diagnostics()['selected_mask'] for m,x in macros.items()}})
    write(a.output/'DEV_hard_routing.json',routes)
    write(a.output/'report.json',{'complete':True,'stage':a.stage,'steps':steps,'best_sha256':digest(a.output/'best_checkpoint.pt'),
        'best_DEV_directional_loss':best,'diagnostics':{m:x.diagnostics() for m,x in macros.items()},
        'source_commit':os.environ['T12_SOURCE_COMMIT'],'planned_Qwen_TF_removal_count':42,'feature_KD_only_not_recovered_scene_generation':True,
        'requires_actual_postfree_RGB_text_recovery_and_tail_speed_work':True,'quality_or_2x_speed_accepted':False,'TEST_opened':False})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['cache','smoke','distill'],required=True)
    for key in ['data','qwen','features','config','output']:p.add_argument('--'+key,type=Path,required=key in ['config','output'])
    a=p.parse_args();a.qwen_side=256
    if a.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU or len(os.environ.get('T12_SOURCE_COMMIT',''))!=40:raise ValueError('New run/published source/GPU6 required')
    a.output.mkdir(parents=True);torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.8,0)
    cfg=json.loads(a.config.read_text());assert cfg['alpha_minimum']==.6 and cfg['postfree']
    write(a.output/'execution.json',{'argv':sys.argv,'pid':os.getpid(),'source_commit':os.environ['T12_SOURCE_COMMIT'],'source_sha256':digest(Path(__file__)),'config':cfg})
    try:cache(a) if a.stage=='cache' else distill(a,cfg)
    except Exception:write(a.output/'failure.json',{'traceback':traceback.format_exc()});raise


if __name__=='__main__':main()
