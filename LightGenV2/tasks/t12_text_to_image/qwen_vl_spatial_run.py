"""TRAIN/VAL-only spatial Qwen experiment; resume public-backbone T12 baseline.

All image conditioning is read from a verified full Qwen-VL cache. Token layout
is checked against the unchanged public processor, never guessed from indices.
"""
import argparse,collections,json,math,os,random,subprocess,sys,time,traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image,ImageDraw
from torch.nn import functional as F

from .qwen_vl_turbo_cache import image_float
from .qwen_vl_turbo_data import normalized_product,pilot_subset,sha256,validate_roles,write_json
from .qwen_vl_turbo_model import pack_hidden,region_weighted_loss
from .qwen_vl_turbo_run import git_identity,noise_for,tensor_image
from .qwen_vl_spatial import balanced_region_l1,build_spatial_head


def identity():
    result=git_identity()
    result['additional_source_sha256']={name:sha256(Path(__file__).with_name(name)) for name in
        ('qwen_vl_spatial.py','qwen_vl_spatial_run.py')}
    return result


def verify_layout(payload,qwen,output):
    from transformers import AutoProcessor
    from .qwen_vl_postfree import instruction_bounds
    processor=AutoProcessor.from_pretrained(qwen,local_files_only=True,min_pixels=65536,max_pixels=65536)
    config=json.loads((qwen/'config.json').read_text(encoding='utf-8'))
    token_id=config['image_token_id'];count=64
    positions=None;span=None
    probes={0,len(payload['rows'])-1,next(i for i,r in enumerate(payload['rows']) if r['split']=='val')}
    dummy=tensor_image(payload['reference'][0])
    for i,row in enumerate(payload['rows']):
        messages=[{'role':'user','content':[{'type':'image','image':dummy},{'type':'text','text':row['prompt']}]}]
        text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        expanded=text.replace(processor.image_token,processor.image_token*count)
        ids=processor.tokenizer(expanded,return_tensors='pt')['input_ids'][0]
        bounds=instruction_bounds(ids,processor.tokenizer.convert_tokens_to_ids('<|vision_end|>'),processor.tokenizer.convert_tokens_to_ids('<|im_end|>'))
        if span is None:span=bounds
        if bounds!=span:raise ValueError('Native instruction token boundaries changed')
        found=torch.where(ids==token_id)[0].tolist()
        if len(found)!=count or len(ids)!=len(payload['hidden'][i]):
            raise ValueError('Processor and frozen hidden layout differ for '+row['sample_id'])
        if positions is None:positions=found
        if found!=positions:raise ValueError('This fixed-grid experiment requires consistent visual token positions')
        if i in probes:
            actual=processor(text=[text],images=[tensor_image(payload['reference'][i])],return_tensors='pt')
            if actual['image_grid_thw'].tolist()!=[[1,16,16]] or not torch.equal(actual['input_ids'][0],ids):
                raise ValueError('Actual processor image expansion disagrees with layout verification')
    result={'positions':positions,'instruction_bounds':span,'merged_grid':[8,8],'unmerged_grid':[1,16,16],
      'checked_rows':len(payload['rows']),'actual_processor_probes':sorted(probes),
      'image_token_id':token_id,'qwen_config_sha256':sha256(qwen/'config.json'),
      'ordering':'Qwen3-VL rot_pos_emb traverses merged block rows/columns; merger reduces each contiguous 2x2 block; no window permutation in this installed forward',
      'inputs_used':'reference RGB + original instruction only; no target image passed to processor'}
    write_json(output/'visual_layout.json',result)
    return result


def training_ids(payload,parent,exclude=True):
    descriptions=json.loads((parent/'product_descriptions.json').read_text(encoding='utf-8'))
    groups=collections.defaultdict(set)
    for r in payload['rows']:
        if r['split']=='train' and r['mode']!='background':
            groups[descriptions[r['target']['sample_id']].strip().lower()].add(r['target']['product_id'])
    ambiguous={k:sorted(v) for k,v in groups.items() if len(v)>1}
    excluded=[];ids=[]
    for i,r in enumerate(payload['rows']):
        if r['split']!='train':continue
        desc=descriptions[r['target']['sample_id']].strip().lower()
        if exclude and r['mode']!='background' and desc in ambiguous:excluded.append(r['sample_id'])
        else:ids.append(i)
    return ids,{'ambiguous_descriptions':ambiguous,'excluded_training_pairs':excluded,
       'reason':'Coarse identical target description names multiple TRAIN product identities under deterministic paired latent regression; no brand/SKU labels introduced',
       'remaining_training_pairs':len(ids),'validation_filtered':False}


def category_ids(rows, categories):
    allowed=set(categories) if categories else {r['category'] for r in rows}
    if not allowed or allowed-{r['category'] for r in rows}:raise ValueError('Unknown/empty training categories')
    ids=[i for i,r in enumerate(rows) if r['category'] in allowed]
    for split in ('train','val'):
        for cat in allowed:
            for mode in ('background','object','joint'):
                if not any(rows[i]['split']==split and rows[i]['category']==cat and rows[i]['mode']==mode for i in ids):
                    raise ValueError('Missing category/mode/split cell')
    return ids


def lr_factor(step, total, warmup=0, minimum=0.1):
    if not 0<=warmup<total or not 0<=minimum<=1:raise ValueError('Invalid cosine schedule')
    if warmup and step<=warmup:return step/warmup
    fraction=min(1.,max(0.,(step-warmup)/max(1,total-warmup)))
    return minimum+(1-minimum)*.5*(1+math.cos(math.pi*fraction))


def selection_score(val, metric):
    if metric=='mean_lpips':return -val[metric]
    if metric=='mean_lpips_plus_object_crop':return -(val['mean_lpips']+.5*val['mean_object_crop_lpips'])
    if metric=='mean_per_image_psnr_db':return val[metric]
    raise ValueError('Unknown VAL selection metric')


def masks(row,device):
    _,source=normalized_product(row['source'],256)
    _,target=normalized_product(row['target'],256)
    src=torch.from_numpy(np.asarray(source).copy()).float()[None,None].to(device)/255
    dst=torch.from_numpy(np.asarray(target).copy()).float()[None,None].to(device)/255
    edit=(1-src if row['mode']=='background' else torch.maximum(src,dst) if row['mode']=='object' else torch.ones_like(src))
    return edit,dst


def counterfactual_negatives(payload,ids):
    """Different TRAIN instructions on the SAME RGB, never DEV negatives."""
    groups=collections.defaultdict(list)
    for i in ids:
        row=payload['rows'][i]
        if row['split']!='train' or row.get('render_protocol')!='rich_product_scene_v1':raise ValueError('Counterfactual ranking requires rich TRAIN only')
    for i,row in enumerate(payload['rows']):
        if row['split']!='train':continue
        if row.get('render_protocol')!='rich_product_scene_v1':raise ValueError('Mixed TRAIN render protocol')
        groups[row['source']['sample_id'],row['mode']].append(i)
    result={}
    for i in ids:
        row=payload['rows'][i];group=groups[row['source']['sample_id'],row['mode']]
        candidates=[j for j in group if payload['rows'][j]['prompt']!=row['prompt']]
        if not candidates:raise ValueError('No distinct same-reference TRAIN instruction')
        if any(not torch.equal(payload['reference'][i],payload['reference'][j]) for j in candidates):raise ValueError('Counterfactual source pixels changed')
        result[i]=candidates
    return result


@torch.no_grad()
def evaluate(model,payload,ids,device,output,label,metric,export=False):
    model.eval();records=[]
    selected=[]
    categories=list(dict.fromkeys(payload['rows'][i]['category'] for i in ids))
    for cat in categories:
        for mode in ('background','object','joint'):
            choices=[j for j,i in enumerate(ids) if (payload['rows'][i]['category'],payload['rows'][i]['mode'])==(cat,mode)]
            if choices:selected.append(choices[0])
    panel=Image.new('RGB',(768,310*len(selected)),'white');draw=ImageDraw.Draw(panel)
    for pos,i in enumerate(ids):
        row=payload['rows'][i];h,valid=pack_hidden([payload['hidden'][i]],device)
        _,prediction=model(h,valid,noise_for([row],device));prediction=prediction.clamp(-1,1)
        target=image_float(payload['target'][i:i+1],device)
        err=(prediction-target).square()/4
        edit,foreground=masks(row,device);keep=1-edit
        # Same source/target union box for both images. Offline metric only;
        # crops/masks never condition the model or alter exported predictions.
        union=foreground if row['mode']=='background' else edit if row['mode']=='object' else torch.maximum(foreground,masks({**row,'mode':'object'},device)[0])
        coords=torch.where(union[0,0]>.05)
        if not len(coords[0]):raise ValueError('Empty metric object support')
        y0=max(0,int(coords[0].min())-8);y1=min(256,int(coords[0].max())+9)
        x0=max(0,int(coords[1].min())-8);x1=min(256,int(coords[1].max())+9)
        crop_a=F.interpolate(prediction[:,:,y0:y1,x0:x1].float(),(128,128),mode='bilinear',align_corners=False)
        crop_b=F.interpolate(target[:,:,y0:y1,x0:x1].float(),(128,128),mode='bilinear',align_corners=False)
        mse=float(err.mean());psnr=-10*math.log10(max(mse,1e-12))
        record={'sample_id':row['sample_id'],'category':row['category'],'mode':row['mode'],'prompt':row['prompt'],
          'psnr':psnr,'mse_01':mse,'lpips':float(metric(prediction.float(),target.float()).mean()),
          'object_crop_lpips':float(metric(crop_a,crop_b).mean()),'object_crop_box':[x0,y0,x1,y1],
          'edited_psnr':-10*math.log10(max(float((err*edit).sum()/(edit.sum().clamp_min(1)*3)),1e-12)),
          'retained_psnr':None if float(keep.sum())<1 else -10*math.log10(max(float((err*keep).sum()/(keep.sum()*3)),1e-12))}
        records.append(record)
        values=(payload['reference'][i],payload['target'][i],prediction[0])
        if export:
            for role,value in zip(('reference','target','generated'),values):
                path=output/'images'/label/role;path.mkdir(parents=True,exist_ok=True)
                tensor_image(value).save(path/f'sample_{pos:04d}.png')
        if pos in selected:
            y=selected.index(pos)*310
            draw.text((4,y+3),f"{label} {row['category']}/{row['mode']} PSNR {psnr:.2f} LPIPS {record['lpips']:.3f}",fill='black')
            draw.text((4,y+23),'REFERENCE                          TARGET                              GENERATED',fill='black')
            for col,value in enumerate(values):panel.paste(tensor_image(value),(col*256,y+54))
    panel.save(output/(label+'.png'))
    result={'count':len(records),'mean_per_image_psnr_db':float(np.mean([r['psnr'] for r in records])),
       'mean_lpips':float(np.mean([r['lpips'] for r in records])),
       'mean_object_crop_lpips':float(np.mean([r['object_crop_lpips'] for r in records])),'images':records,
       'by_mode':{m:float(np.mean([r['psnr'] for r in records if r['mode']==m])) for m in ('background','object','joint')},
       'lpips_by_mode':{m:float(np.mean([r['lpips'] for r in records if r['mode']==m])) for m in ('background','object','joint')},
       'object_crop_lpips_by_mode':{m:float(np.mean([r['object_crop_lpips'] for r in records if r['mode']==m])) for m in ('background','object','joint')},
       'edited_psnr_by_mode':{m:float(np.mean([r['edited_psnr'] for r in records if r['mode']==m])) for m in ('background','object','joint')},
       'retained_psnr_by_mode':{m:(float(np.mean(a)) if (a:=[r['retained_psnr'] for r in records if r['mode']==m and r['retained_psnr'] is not None]) else None) for m in ('background','object','joint')},
       'by_category_mode':{cat:{mode:{'n':len(cell),
           'psnr':float(np.mean([r['psnr'] for r in cell])),
           'lpips':float(np.mean([r['lpips'] for r in cell])),
           'object_crop_lpips':float(np.mean([r['object_crop_lpips'] for r in cell])),
           'edited_psnr':float(np.mean([r['edited_psnr'] for r in cell]))}
           for mode in ('background','object','joint') if (cell:=[r for r in records if r['category']==cat and r['mode']==mode])}
           for cat in categories},
       'scope':'Development TRAIN diagnostics or VAL; no TEST evaluation; masks used for metrics only'}
    write_json(output/(label+'.json'),result)
    return result


def run(args,cfg):
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'report.json').exists():raise ValueError('Existing completed run; inspect instead of restarting')
    parent_report=json.loads((args.parent/'report.json').read_text(encoding='utf-8'))
    parent_execution=json.loads((args.parent/'execution.json').read_text(encoding='utf-8'))
    if not parent_report['complete'] or parent_report['best_step']<=0:raise ValueError('Parent VAL-selected model is not ready')
    checkpoint=args.parent/'best_checkpoint.pt'
    if sha256(checkpoint)!=parent_report['checkpoint_sha256']:raise ValueError('Parent checkpoint SHA mismatch')
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    data_run=args.data_run or args.parent
    cache=data_run/'frozen_features.pt'
    cache_digest=sha256(cache)
    if args.data_run:
        prepared=json.loads((data_run/'cache_complete.json').read_text())
        if not prepared['complete'] or cache_digest!=prepared['sha256']:raise ValueError('Reviewed feature cache incomplete/changed')
        data_audit=json.loads((data_run/'data_audit.json').read_text())
        if not data_audit['role_disjoint'] or data_audit['test_opened']:raise ValueError('Reviewed split audit failed')
    elif cache_digest!=state['cache_sha256']:raise ValueError('Frozen feature cache SHA mismatch')
    payload=torch.load(cache,map_location='cpu',weights_only=False,mmap=True)
    validate_roles(payload['rows'])
    # Modify only in-memory sampling state; the parent cache is never overwritten.
    if cfg.get('shared_noise_seed') is not None:
        payload['rows']=[{**r,'noise_seed':cfg['shared_noise_seed']} for r in payload['rows']]
    layout=verify_layout(payload,args.qwen,out)
    if cfg.get('focus_instruction'):
        cfg['instruction_bounds']=layout['instruction_bounds']
    native_digest=None
    if args.visual_cache:
        ready=json.loads((args.visual_cache/'report.json').read_text())
        native_path=args.visual_cache/'native_vision.pt';native_digest=sha256(native_path)
        if not ready['complete'] or native_digest!=ready['sha256'] or ready['source_cache_sha256']!=cache_digest:
            raise ValueError('Native visual cache identity differs')
        native=torch.load(native_path,map_location='cpu',weights_only=False,mmap=True)
        if native['sample_ids']!=[r['sample_id'] for r in payload['rows']]:raise ValueError('Native vision sample order differs')
        if cfg.get('semantic_prefix')!=64:raise ValueError('Native vision requires an explicit 64-token prefix')
        if any(v.shape!=(64,2048) for v in native['vision']):raise ValueError('Native vision shape differs')
        payload['hidden']=[torch.cat([v,h],0) for v,h in zip(native['vision'],payload['hidden'])]
        layout.update(qwen_image_positions=layout['positions'],positions=list(range(64)),
                      semantic_prefix=64,visual_source='native Qwen visual merger before language')
        write_json(out/'visual_layout.json',layout)
    elif cfg.get('semantic_prefix',0):raise ValueError('Native visual prefix configured without verified cache')
    ids,audit=training_ids(payload,data_run,cfg['exclude_ambiguous_descriptions'])
    permitted=set(category_ids(payload['rows'],cfg.get('categories')))
    ids=[i for i in ids if i in permitted]
    val_ids=[i for i,r in enumerate(payload['rows']) if r['split']=='val' and i in permitted]
    audit.update(active_categories=sorted({payload['rows'][i]['category'] for i in permitted}),
        remaining_training_pairs=len(ids),selected_val_pairs=len(val_ids),category_filter_is_not_quality_selection=True)
    diagnostics={r['sample_id'] for r in pilot_subset(payload['rows'],2) if r['split']=='train'}
    train_eval=[i for i in ids if payload['rows'][i]['sample_id'] in diagnostics]
    if cfg.get('smoke',False):
        ids=train_eval
        names={r['sample_id'] for r in pilot_subset(payload['rows'],2) if r['split']=='val'}
        val_ids=[i for i in val_ids if payload['rows'][i]['sample_id'] in names]
    if cfg.get('expected_train_pairs',len(ids))!=len(ids) or cfg.get('expected_evaluation_pairs',len(val_ids))!=len(val_ids):
        raise ValueError('Approved pair budget differs; inspect exclusions before training')
    evaluation_label=cfg.get('evaluation_label','validation')
    if evaluation_label=='development_test' and not (cfg.get('test_selection_authorized') and data_audit.get('test_selection_authorized')):
        raise ValueError('Development-test selection requires the authorized data protocol')
    write_json(out/'annotation_audit.json',audit)
    write_json(out/'active_pairs.json',{'cache_sha256':cache_digest,
        'train':[payload['rows'][i] for i in ids],'val':[payload['rows'][i] for i in val_ids],
        'evaluation_label':evaluation_label,'independent_test_claim':False,
        'note':'Identity-preserving subset of verified full-Qwen cache; excluded categories receive no optimization/evaluation'})
    write_json(out/'config.json',cfg)
    write_json(out/'execution.json',{'argv':sys.argv,'git':identity(),'started_unix':time.time(),
       'torch':torch.__version__,'parent_run':str(args.parent),'parent_checkpoint_sha256':sha256(checkpoint),
       'cache_sha256':cache_digest,'data_run':str(data_run),'parent_cache_sha256':state['cache_sha256'],
       'native_visual_cache_sha256':native_digest,
       'data_profile_sha256':(data_audit.get('profile_sha256') or json.loads((data_run/'report.json').read_text())['profile_sha256']) if args.data_run else None,
       'train_pairs':len(ids),'val_pairs':len(val_ids),
       'test_opened':False,'visual_layout':layout,'gpu_uuid':os.environ.get('CUDA_VISIBLE_DEVICES')})
    if args.stage=='layout':return
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=cfg['gpu_uuid']:raise ValueError('Unauthorized GPU UUID')
    device=torch.device('cuda:0');torch.cuda.set_per_process_memory_fraction(cfg['gpu_memory_fraction'],0)
    h,valid=pack_hidden([payload['hidden'][val_ids[0]]],device)
    probe=(h,valid,noise_for([payload['rows'][val_ids[0]]],device))
    if 'spatial' in state:
        for key in ('adapter_hidden','adapter_layers','spatial_width','spatial_channels','visual_grid','unet_lora_rank','unet_training'):
            if cfg[key]!=state['config'][key]:raise ValueError('Resume architecture mismatch: '+key)
        if layout['positions']!=state['visual_layout']['positions']:raise ValueError('Resume token layout changed')
        if cfg.get('semantic_prefix',0)!=state['config'].get('semantic_prefix',0):raise ValueError('Resume visual source changed; start from unaugmented parent')
    model=build_spatial_head(args.sd,state,cfg,layout['positions'],device,
        identity_probe=None if 'spatial' in state else probe)
    write_json(out/'zero_injection_identity.json',model.initial_identity if 'spatial' not in state else
        {'applicable':False,'reason':'Continue a trained spatial checkpoint; its injection is already nonzero',
         'parent_checkpoint_sha256':parent_report['checkpoint_sha256'],'architecture_and_layout_matched':True})
    model.unet.enable_gradient_checkpointing()
    import lpips
    metric=lpips.LPIPS(net='alex').to(device).eval().requires_grad_(False)
    semantic=None;semantic_scene={};semantic_product={};semantic_meta={'enabled':False}
    if cfg.get('semantic_scene_weight',0) or cfg.get('semantic_product_weight',0):
        from .qwen_vl_semantic_loss import prepare_caption_loss
        descriptions=json.loads((data_run/'product_descriptions.json').read_text(encoding='utf-8'))
        semantic,semantic_scene,semantic_product,semantic_meta=prepare_caption_loss(
            cfg['clip_repo'],cfg['clip_weights'],payload['rows'],ids,descriptions,device)
    write_json(out/'semantic_loss_protocol.json',semantic_meta)
    before=evaluate(model,payload,val_ids,device,out,'val_before_training',metric)
    # TRAIN-only recalibration; no VAL conditions or images supervise this step.
    if cfg.get('condition_warm_steps',0):
        warm_condition=torch.optim.AdamW(model.adapter.parameters(),lr=cfg['condition_warm_lr'])
        warm_rng=random.Random(cfg['seed'])
        for step in range(1,cfg['condition_warm_steps']+1):
            batch=warm_rng.sample(ids,min(cfg['condition_warm_batch_size'],len(ids)))
            h,valid=pack_hidden([payload['hidden'][i] for i in batch],device)
            goal=torch.stack([payload['train_clip_teacher'][i] for i in batch]).float().to(device)
            loss=F.mse_loss(model.semantic_conditions(h,valid).float(),goal)
            warm_condition.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.adapter.parameters(),1.);warm_condition.step()
            if step%100==0:
                write_json(out/'progress.json',{'stage':'TRAIN_condition_warm','step':step,'total_steps':cfg['condition_warm_steps'],'mse':float(loss.detach())})
        write_json(out/'condition_warm_report.json',{'steps':step,'last_train_batch_mse':float(loss.detach()),'val_used':False})
        del warm_condition
    expected={r['sample_id']:r['psnr'] for r in parent_report['best_val']['images']}
    deviations=[abs(r['psnr']-expected[r['sample_id']]) for r in before['images'] if r['sample_id'] in expected]
    write_json(out/'cross_device_parent_comparison.json',{'max_per_sample_psnr_difference':max(deviations) if deviations else None,
        'samples':len(deviations),'original_parent_gpu':parent_execution.get('gpu_uuid',parent_execution.get('CUDA_VISIBLE_DEVICES')),
        'current_gpu':torch.cuda.get_device_name(0),
        'note':'New data profiles require the freshly rescored before-training cohort, not old VAL means. Any overlapping historical scores are diagnostic only.'})
    ref_latent={};target_latent={};edit_regions={};target_regions={}
    model.eval()
    with torch.no_grad():
        for start in range(0,len(ids),16):
            batch=ids[start:start+16]
            for role,destination in (('reference',ref_latent),('target',target_latent)):
                images=image_float(payload[role][batch],device,model.vae.dtype)
                z=(model.vae.encode(images).latent_dist.mode()*model.vae.config.scaling_factor).float().cpu()
                for i,value in zip(batch,z):destination[i]=value
            for i in batch:edit_regions[i],target_regions[i]=[v.cpu() for v in masks(payload['rows'][i],device)]
            if start%256==0:
                write_json(out/'progress.json',{'stage':'TRAIN_vae_supervision','done':start+len(batch),'total':len(ids)})
    rng=random.Random(cfg['seed']);warm=torch.optim.AdamW(model.spatial.parameters(),lr=cfg['warm_lr'])
    negatives=counterfactual_negatives(payload,ids) if cfg.get('counterfactual_weight',0) else None
    for step in range(1,cfg['warm_steps']+1):
        batch=rng.sample(ids,min(cfg['warm_batch_size'],len(ids)))
        h,valid=pack_hidden([payload['hidden'][i] for i in batch],device)
        visual,context=model.spatial_inputs(h,valid)
        _,source=model.spatial(visual,with_aux=True,context=context)
        loss=F.mse_loss(source.float(),torch.stack([ref_latent[i] for i in batch]).to(device))
        warm.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.spatial.parameters(),1.);warm.step()
        if step%50==0 or step==cfg['warm_steps']:
            print('spatial warm',step,float(loss.detach()),flush=True)
            write_json(out/'progress.json',{'stage':'spatial_warm','step':step,'total_steps':cfg['warm_steps'],'loss':float(loss.detach())})
    del warm
    params=[p for p in model.parameters() if p.requires_grad]
    injection=list(model.unet.conv_in.spatial.parameters());injection_ids={id(p) for p in injection}
    optimizer=torch.optim.AdamW([{'params':model.spatial.parameters(),'lr':cfg['spatial_lr']},
       {'params':model.adapter.parameters(),'lr':cfg['adapter_lr']},
       {'params':[p for p in model.unet.parameters() if p.requires_grad and id(p) not in injection_ids],'lr':cfg['train_lr']},
       {'params':injection,'lr':cfg.get('spatial_injection_lr',cfg['train_lr'])}],weight_decay=.01)
    aux_count=sum(p.numel() for p in model.spatial.aux_reference.parameters())
    counts={'qwen_full_forward':payload['qwen_checks']['qwen_forward_parameters'],
      'qwen_embedding':payload['qwen_checks']['qwen_embedding_parameters'],
      'adapter':sum(p.numel() for p in model.adapter.parameters()),
      'unet_including_lora':sum(p.numel() for p in model.unet.parameters()),
      'spatial_readout':sum(p.numel() for p in model.spatial.parameters())-aux_count,
      'training_only_spatial_auxiliary':aux_count,
      'vae_decoder':sum(p.numel() for p in model.vae.decoder.parameters())+sum(p.numel() for p in model.vae.post_quant_conv.parameters()),
      'trainable':sum(p.numel() for p in params)}
    counts['training_only_vae_encoder']=sum(p.numel() for p in model.vae.encoder.parameters())+sum(p.numel() for p in model.vae.quant_conv.parameters())
    if semantic is not None:counts['training_only_semantic_encoder']=semantic_meta['frozen_parameters']
    has_detail=any(cfg.get(k,0) for k in ('foreground_rgb_weight','object_crop_weight','boundary_weight'))
    discriminator=None;optimizer_disc=None
    if has_detail or cfg.get('gan_weight',0):
        from .qwen_vl_turbo_losses import foreground_detail_losses,region_boxes,crop_regions,PatchDiscriminator,discriminator_views,hinge_discriminator_loss
    if cfg.get('gan_weight',0):
        discriminator=PatchDiscriminator(cfg.get('discriminator_width',32)).to(device)
        if 'training_only_discriminator' in state:discriminator.load_state_dict(state['training_only_discriminator'],strict=True)
        # Spectral-norm power iteration must run before the first frozen-critic
        # G pass. Zero tensors initialize weight-only buffers, not DEV statistics.
        with torch.no_grad():
            for _ in range(3):discriminator(torch.zeros(1,3,128,128,device=device))
        optimizer_disc=torch.optim.Adam(discriminator.parameters(),lr=cfg.get('discriminator_lr',1e-4),betas=(.5,.999))
        counts['training_only_discriminator']=sum(p.numel() for p in discriminator.parameters())
    counts['inference_total']=sum(counts[k] for k in ('qwen_full_forward','adapter','unet_including_lora','spatial_readout','vae_decoder'))
    write_json(out/'parameter_counts.json',counts)
    effective=cfg['batch_size']*cfg['gradient_accumulation']
    steps=cfg.get('train_steps') or math.ceil(len(ids)*cfg['epochs']/effective)
    rates=[g['lr'] for g in optimizer.param_groups]
    selector=cfg.get('selection_metric','mean_per_image_psnr_db')
    selection_score(before,selector)
    order=[]
    while len(order)<steps*effective:
        block=ids.copy();rng.shuffle(block);order.extend(block)
    best=-float('inf');best_step=0;history=[];best_psnr=None
    for step in range(1,steps+1):
        if cfg.get('lr_schedule')=='cosine':
            factor=lr_factor(step,steps,cfg.get('lr_warmup_steps',0),cfg.get('lr_min_factor',.1))
            for group,rate in zip(optimizer.param_groups,rates):group['lr']=rate*factor
        model.train();model.vae.eval();optimizer.zero_grad(set_to_none=True);total=0.;terms={};real_views=[];fake_views=[]
        for micro in range(cfg['gradient_accumulation']):
            start=((step-1)*cfg['gradient_accumulation']+micro)*cfg['batch_size'];batch=order[start:start+cfg['batch_size']]
            h,valid=pack_hidden([payload['hidden'][i] for i in batch],device)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                z,rgb,source=model.forward_with_aux(h,valid,noise_for([payload['rows'][i] for i in batch],device))
                target=image_float(payload['target'][batch],device)
                latent=region_weighted_loss(z,torch.stack([target_latent[i] for i in batch]).to(device),
                    torch.cat([target_regions[i] for i in batch]).to(device),cfg['latent_foreground_weight'])
                pixel=balanced_region_l1(rgb,target,torch.cat([edit_regions[i] for i in batch]).to(device))
                correct_perceptual=metric(rgb.clamp(-1,1),target).flatten()
                perceptual=correct_perceptual.mean()
                spatial_loss=F.mse_loss(source.float(),torch.stack([ref_latent[i] for i in batch]).to(device))
                conditions=model.semantic_conditions(h,valid)
                align=F.mse_loss(conditions.float(),torch.stack([payload['train_clip_teacher'][i] for i in batch]).float().to(device))
                loss=(cfg['latent_weight']*latent+cfg['rgb_weight']*pixel+cfg['lpips_weight']*perceptual+
                      cfg['reference_weight']*spatial_loss+cfg['condition_weight']*align)
                if semantic is not None:
                    from .qwen_vl_turbo_losses import crop_regions,region_boxes
                    scene_loss=semantic(rgb,torch.stack([semantic_scene[i] for i in batch]).to(device))
                    boxes=region_boxes(torch.cat([masks({**payload['rows'][i],'mode':'object'},device)[0] for i in batch]))
                    product_loss=semantic(crop_regions(rgb,boxes),torch.stack([semantic_product[i] for i in batch]).to(device))
                    loss=loss+cfg.get('semantic_scene_weight',0)*scene_loss+cfg.get('semantic_product_weight',0)*product_loss
                if negatives is not None:
                    wrong_ids=[rng.choice(negatives[i]) for i in batch]
                    wrong_target=image_float(payload['target'][wrong_ids],device)
                    wrong_perceptual=metric(rgb.clamp(-1,1),wrong_target).flatten()
                    ranking=F.relu(cfg.get('counterfactual_margin',.05)+correct_perceptual-wrong_perceptual).mean()
                    loss=loss+cfg['counterfactual_weight']*ranking
                if has_detail or discriminator is not None:
                    foreground=torch.cat([target_regions[i] for i in batch]).to(device)
                    detail_boxes=region_boxes(foreground)
                if has_detail:
                    foreground_rgb,boundary=foreground_detail_losses(rgb,target,foreground)
                    crop_perceptual=metric(crop_regions(rgb.clamp(-1,1),detail_boxes),crop_regions(target,detail_boxes)).mean()
                    loss=loss+cfg.get('foreground_rgb_weight',0)*foreground_rgb+cfg.get('boundary_weight',0)*boundary+cfg.get('object_crop_weight',0)*crop_perceptual
                if discriminator is not None:
                    discriminator.eval().requires_grad_(False)
                    fake=discriminator_views(rgb.clamp(-1,1),detail_boxes)
                    adversarial=-discriminator(fake).mean()
                    ramp=min(1.,step/max(1,cfg.get('gan_warmup_steps',128)))
                    loss=loss+cfg['gan_weight']*ramp*adversarial
                    fake_views.append(fake.detach());real_views.append(discriminator_views(target,detail_boxes).detach())
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            (loss/cfg['gradient_accumulation']).backward();total+=float(loss.detach())/cfg['gradient_accumulation']
            terms={k:float(v.detach()) for k,v in {'latent':latent,'rgb':pixel,'lpips':perceptual,'reference':spatial_loss,'condition':align}.items()}
            if negatives is not None:terms['counterfactual_rank']=float(ranking.detach())
            if has_detail:terms.update(foreground_rgb=float(foreground_rgb.detach()),boundary=float(boundary.detach()),object_crop_lpips=float(crop_perceptual.detach()))
            if discriminator is not None:terms['generator_adversarial']=float(adversarial.detach())
        norm=torch.nn.utils.clip_grad_norm_(params,1.)
        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
        optimizer.step()
        if discriminator is not None:
            discriminator.train().requires_grad_(True);optimizer_disc.zero_grad(set_to_none=True)
            loss_disc=hinge_discriminator_loss(discriminator(torch.cat(real_views)),discriminator(torch.cat(fake_views)))
            if not torch.isfinite(loss_disc):raise FloatingPointError('Nonfinite TRAIN discriminator loss')
            loss_disc.backward();disc_norm=torch.nn.utils.clip_grad_norm_(discriminator.parameters(),1.)
            if not torch.isfinite(disc_norm):raise FloatingPointError('Nonfinite TRAIN discriminator gradient')
            optimizer_disc.step();terms['discriminator']=float(loss_disc.detach())
        if step==4:
            write_json(out/'smoke_report.json',{'real_backward_steps':4,'finite_loss_gradient':True,
                       'TRAIN_only':True,'Qwen_frozen':True,'VAE_frozen':not any(p.requires_grad for p in model.vae.parameters()),'adversarial_training':discriminator is not None})
        if step%10==0:
            print('paired',step,'/',steps,'loss',total,terms,flush=True)
            write_json(out/'progress.json',{'stage':'paired','step':step,'total_steps':steps,'loss':total,'terms':terms,'best_step':best_step,'best_val_psnr':best_psnr,'selection_metric':selector,'learning_rates':[g['lr'] for g in optimizer.param_groups]})
        if step%cfg['eval_every']==0 or step==steps:
            val=evaluate(model,payload,val_ids,device,out,'val_latest',metric)
            tr=evaluate(model,payload,train_eval,device,out,'train_latest',metric)
            entry={'step':step,'epoch':step*effective/len(ids),'train_loss':total,'val_psnr':val['mean_per_image_psnr_db'],'val_lpips':val['mean_lpips'],'val_object_crop_lpips':val['mean_object_crop_lpips'],'train_psnr':tr['mean_per_image_psnr_db'],'learning_rates':[g['lr'] for g in optimizer.param_groups]}
            history.append(entry);print('validation',entry,flush=True)
            saved={'schema':'qwen_spatial_v1','step':step,'adapter':model.adapter.state_dict(),
              'spatial':model.spatial.state_dict(),'unet_trainable':{n:p.detach().cpu().clone() for n,p in model.unet.named_parameters() if p.requires_grad},
              'config':cfg,'visual_layout':layout,'git':identity(),'cache_sha256':cache_digest,
              'native_visual_cache_sha256':native_digest,
              'parent_checkpoint_sha256':parent_report['checkpoint_sha256'],'val':val}
            if discriminator is not None:saved['training_only_discriminator']={k:v.detach().cpu().clone() for k,v in discriminator.state_dict().items()}
            torch.save(saved,out/'last_checkpoint.pt')
            if selection_score(val,selector)>best:
                best=selection_score(val,selector);best_psnr=val['mean_per_image_psnr_db'];best_step=step;torch.save(saved,out/'best_checkpoint.pt')
            write_json(out/'history.json',history)
            if cfg.get('early_stop_evaluations') and step-best_step>=cfg['early_stop_evaluations']*cfg['eval_every']:
                break
    selected=torch.load(out/'best_checkpoint.pt',map_location='cpu',weights_only=False)
    model.adapter.load_state_dict(selected['adapter']);model.spatial.load_state_dict(selected['spatial'])
    with torch.no_grad():
        for n,p in model.unet.named_parameters():
            if n in selected['unet_trainable']:p.copy_(selected['unet_trainable'][n].to(device))
    val=evaluate(model,payload,val_ids,device,out,'val_best',metric,export=True)
    train=evaluate(model,payload,train_eval,device,out,'train_best',metric,export=True)
    model.zero_spatial=True
    without=evaluate(model,payload,val_ids,device,out,'val_without_spatial',metric,export=True)
    model.zero_spatial=False
    write_json(out/'report.json',{'complete':True,'best_step':best_step,'best_val':val,'train':train,
       'evaluation_label':evaluation_label,'evaluation_used_for_selection':True,'independent_test_claim':False,
       'selection_metric':selector,'best_selection_score':best,'epochs_completed':step*effective/len(ids),
       'val_before_paired_training':before,'val_without_spatial':without,'parameters':counts,
       'train_optimization_pairs':len(ids),'train_evaluation_pairs':len(train_eval),'test_evaluated':False,
       'checkpoint_sha256':sha256(out/'best_checkpoint.pt'),'parent_checkpoint_sha256':parent_report['checkpoint_sha256'],
       'scope':'Development TRAIN/VAL only. Source VAE latents supervise auxiliary readout only; all inference image information enters through full Qwen-VL.',
       'new_spatial_contribution_psnr':val['mean_per_image_psnr_db']-without['mean_per_image_psnr_db'],
       'quality_accepted':False,'quality_note':'Requires visual review and separate raw Qwen image/text ablations; completion alone is not an acceptance claim'})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('config','parent','qwen','sd','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--stage',choices=('layout','train'),default='train')
    parser.add_argument('--data-run',type=Path,help='Explicit reviewed data/cache profile; preserve parent training exposure in provenance')
    parser.add_argument('--visual-cache',type=Path,help='Verified native output from the SAME Qwen vision tower; semantic fused states still retained')
    args=parser.parse_args();cfg=json.loads(args.config.read_text(encoding='utf-8'))
    if cfg.get('test_enabled',False):raise ValueError('This development runner has no TEST path')
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.set_num_threads(4)
    try:run(args,cfg)
    except Exception:
        write_json(args.output/'failure.json',{'traceback':traceback.format_exc(),'time':time.time()});raise
    finally:
        if torch.cuda.is_initialized():torch.cuda.empty_cache()


if __name__=='__main__':main()
