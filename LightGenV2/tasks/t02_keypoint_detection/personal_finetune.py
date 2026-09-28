"""Matched small-domain adaptation; unreviewed labels are diagnostic only."""
from __future__ import annotations
import argparse,copy,json,math,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from .personal_data import load_personal
from .personal_prepare import sha256
from .run import _seed
from .refine import apply_stage,checked_fusion
from .settings import load_settings,save_resolved_config
from .modeling import build_student,optimizer,architecture_label
from .train_qwen_head import audit_frozen_teacher
from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.modeling import build_teacher,load_vision_backbone,preprocess_vision
from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.datasets import LSPPoseDataset
from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.training import _train_epoch,evaluate_model
from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_router.training import _loader

TASK=Path(__file__).resolve().parent
SOURCES={'ours':'dbc059e2a7eddefac73d3b9bb158bf0140d440dfb396e0aa2956ad41670fc96a',
         'baseline':'0a4569f288f6de424b1b412452fa804a96e58d8f7e8655efaa23f461ab8cc735'}


@torch.no_grad()
def visuals(model,loaded,records,settings,out,label):
    out.mkdir(parents=True,exist_ok=True);ds=LSPPoseDataset(records,settings,training=False);items=[]
    edges=[(0,1),(1,2),(2,3),(3,4),(4,5),(6,7),(7,8),(8,9),(9,10),(10,11),(8,12),(9,12),(12,13)]
    model.eval()
    for k in range(len(ds)):
        sample=ds[k];result=model(**preprocess_vision(loaded.processor,[sample['image']],loaded.device))
        hm=(result[0] if isinstance(result,tuple) else result).float().cpu()[0]
        flat=hm.flatten(1).argmax(1);xy=torch.stack([flat%hm.shape[-1],flat//hm.shape[-1]],dim=1).numpy()*settings.image_size/settings.heatmap_size
        im=sample['image'].copy();d=ImageDraw.Draw(im)
        for a,b in edges:d.line([tuple(xy[a]),tuple(xy[b])],fill='#fa4d48',width=2)
        for x,y in xy:d.ellipse((x-2,y-2,x+2,y+2),fill='#ffeb76')
        im.save(out/f"{sample['sample_id']}.png");items.append({'id':sample['sample_id'],'prediction_xy_224':xy.tolist(),'crop_box':sample['crop_box']})
    (out/'predictions.json').write_text(json.dumps({'label':label,'samples':items},indent=2),encoding='utf-8')


def run(a):
    if a.evaluate_all and not a.evaluate_only:raise ValueError('--evaluate-all requires --evaluate-only; never train on the all-photo evaluation')
    if sha256(a.source)!=SOURCES[a.method]:raise ValueError('Wrong source checkpoint; keep reviewed source identity')
    bundle=load_personal(a.annotations,a.allow_provisional,a.fewshot_photos,a.seed)
    evaluation_records=bundle.train+bundle.test if a.evaluate_all else bundle.test
    out=a.run_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    def write(n,d):(out/n).write_text(json.dumps(d,indent=2,default=str)+'\n',encoding='utf-8')
    cfg=TASK/'configs/moe_alpha40.yaml' if a.method=='ours' else TASK.parents[2]/'experiments/qwen3_vl_embedding_2b_lsp_pose_optical_moe16/configs/lsp_pose_opt2.yaml'
    if a.method=='baseline':
        from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.settings import load_settings as baseline_settings
        s=baseline_settings(cfg)
    else:s=load_settings(cfg)
    s.output_dir=out;s.cache_dir=a.cache_dir;s.local_files_only=True;s.download=False;s.num_workers=0
    s.student_batch_size=s.teacher_batch_size=s.inference_batch_size=a.batch_size
    s.visualization_sample_count=0;s.coordinate_loss_weight=0.;s.teacher_distill_weight=0.
    s.crop_scale_jitter=.10;s.crop_center_jitter=.03;s.horizontal_flip_probability=.5
    s.brightness_jitter=.10;s.contrast_jitter=.10;s.augmentation_enabled=True;s.log_interval_batches=20
    _seed(a.seed);device=torch.device('cuda:0');loaded=load_vision_backbone(s,device);model=None
    try:
        payload=torch.load(a.source,map_location='cpu',weights_only=False)
        if a.method=='ours':
            model=build_student(loaded,s)
            if payload['checkpoint_architecture']!=architecture_label(s):raise ValueError('Architecture mismatch')
            model.core.load_state_dict(payload['core'],strict=True);model.head.load_state_dict(payload['head'],strict=True)
            opt=optimizer(model,s);groups={g['name']:sum(p.numel() for p in g['params']) for g in opt.param_groups}
            if sum(groups[k] for k in ['electronic','ccd_readout','pose_head'])!=836248:raise ValueError('Electronic budget changed')
            checked_fusion(model,s);model.core.set_phase_dropout_active(True)
        else:
            model=build_teacher(loaded,s);model.head.load_state_dict(payload['head'],strict=True)
            audit=audit_frozen_teacher(model)
            if audit['head']['parameters']!=1102990:raise ValueError('Use original full baseline head')
            opt=torch.optim.AdamW(model.head.parameters(),lr=1e-4,weight_decay=1e-4);groups={'pose_head':1102990}
        frozen=[p for p in loaded.visual.parameters() if not p.requires_grad]
        manifest={'command':sys.argv,'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  'method':a.method,'source':str(a.source),'source_sha256':sha256(a.source),'data':bundle.metadata,
                  'seed':a.seed,'epochs':a.epochs,'parameter_groups':groups,'test_used_for_selection':False,
                  'selection':'minimum unaugmented TRAIN heatmap MSE (not a claim of best generalization)',
                  'formal_result':not a.allow_provisional,'metric_warning':'PILOT: agreement with independent pseudo labels, NOT ground-truth accuracy' if a.allow_provisional else None,
                  'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'architecture_changed':False}
        manifest.update(evaluate_only=a.evaluate_only,evaluation_scope='all photos' if a.evaluate_all else 'held-out split',evaluation_people=len(evaluation_records))
        write('run_manifest.json',manifest)
        if a.method=='ours':save_resolved_config(s)
        else:
            from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.settings import save_resolved_config as save_baseline_config
            save_baseline_config(s)
        write('status.json',{'status':'training'})
        kind='student' if a.method=='ours' else 'teacher'
        tr=_loader(bundle.train,s,training=True);clean=_loader(bundle.train,s,training=False);te=_loader(evaluation_records,s,training=False)
        def ev(loader,phase,epoch):return evaluate_model(model,kind,loader,loaded.processor,device,s,phase=phase,epoch=epoch,save_outputs=False,tta=False)[0]
        initial=ev(te,'personal_before',0);write('before.json',initial)
        visuals(model,loaded,evaluation_records,s,out/'before_images','original LSP checkpoint; no personal adaptation')
        if a.evaluate_only:
            write('final_report.json',{'metrics':initial,'evaluation_people':len(evaluation_records),'training_performed':False,
                'formal_ground_truth_evaluation':not a.allow_provisional,'metric_warning':manifest['metric_warning'],
                'source_sha256':sha256(a.source),'fusion':checked_fusion(model,s) if a.method=='ours' else None})
            write('status.json',{'status':'complete'});return
        history=[];best=float('inf');best_epoch=0
        for epoch in range(1,a.epochs+1):
            started=time.perf_counter();factor=.1+.9*.5*(1+math.cos(math.pi*(epoch-1)/max(a.epochs-1,1)))
            if a.method=='ours':
                rates={'electronic':3e-6,'router':3e-5,'feature_phase':3e-4,'ccd_readout':1e-4,'pose_head':1e-4}
                if epoch<=3:rates.update(electronic=0.,router=0.,feature_phase=0.)
                apply_stage(opt,{k:v*factor for k,v in rates.items()});checked_fusion(model,s)
            else:opt.param_groups[0]['lr']=1e-4*factor
            train=_train_epoch(model,kind,tr,loaded.processor,device,opt,s,epoch)
            if any(p.grad is not None for p in frozen):raise ValueError('Frozen backbone acquired gradients')
            train_eval=ev(clean,'train_selection',epoch)
            state={'head':model.head.state_dict(),'epoch':epoch,'manifest':manifest}
            if a.method=='ours':state.update(core=model.core.state_dict(),checkpoint_architecture=architecture_label(s),fusion=checked_fusion(model,s))
            torch.save(state,out/'last_checkpoint.pt')
            if train_eval['heatmap_loss']<best:
                best=train_eval['heatmap_loss'];best_epoch=epoch;torch.save(state,out/'best_checkpoint.pt')
            row={'epoch':epoch,'train':train,'unaugmented_train':train_eval,'best_epoch':best_epoch,'seconds':time.perf_counter()-started}
            history.append(row);write('training_history.json',history);print('EPOCH',epoch,'TRAIN_MSE',best,flush=True)
        state=torch.load(out/'best_checkpoint.pt',map_location=device,weights_only=False)
        model.head.load_state_dict(state['head'],strict=True)
        if a.method=='ours':model.core.load_state_dict(state['core'],strict=True)
        final=ev(te,'personal_after',best_epoch);visuals(model,loaded,bundle.test,s,out/'after_images','after adaptation')
        write('final_report.json',{'before':initial,'after':final,'best_epoch':best_epoch,
              'formal_ground_truth_evaluation':not a.allow_provisional,'metric_warning':manifest['metric_warning'],
              'fusion':checked_fusion(model,s) if a.method=='ours' else None,'checkpoint_sha256':sha256(out/'best_checkpoint.pt'),
              'test_used_for_selection':False})
        write('status.json',{'status':'complete'})
    except Exception as e:write('status.json',{'status':'failed','error':repr(e)});raise
    finally:
        if model is not None:
            if a.method=='ours':model.restore_native()
            else:model.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['ours','baseline'],required=True)
    p.add_argument('--annotations',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--cache-dir',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--allow-provisional',action='store_true');p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--evaluate-only',action='store_true');p.add_argument('--evaluate-all',action='store_true')
    p.add_argument('--fewshot-photos',type=int,default=0)
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--seed',type=int,default=42)
    run(p.parse_args())
