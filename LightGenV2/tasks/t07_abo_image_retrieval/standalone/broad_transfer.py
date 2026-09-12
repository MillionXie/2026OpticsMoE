"""Broader ABO pretraining -> target adaptation, unchanged standalone inference graph.

The normalized category proxy head is training-only. No full Qwen/teacher forward.
One GPU; original profile freezes alpha, high_alpha enforces a strict >0.4 floor.
Original/high_alpha profiles freeze the frontend; explicit domain controls may
adapt an existing frontend submodule, recorded in model.audit(). Extra optical
heads are training-only.
"""
import argparse
import csv
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image,ImageEnhance
from .data import Sample,_load_contract
from .model import OpticalRetrieval
from .io import inputs,picture,sha256,verify_assets,write_json,source_commit
from .cli import autocast,encode,evaluate,supcon,regularization,preview
from .curriculum import parameter_kind
from .prepare_broad_abo import safe_image
from .generalization import PROFILES, overlay_config, apply_contract, backward_with_sam, parameter_decay, restore_auxiliary_head, initialize_category_proxies, supervised_loss_scale
from .generalization import learning_rate_multiplier, PINNED_TEACHER_PROFILES, phase_only_group_frozen
from .learning_curves import write_learning_curves
from .domain_data import combine_training, epoch_batches, paired_view_indices, view_consistency_loss
from .randomness import training_seed, epoch_random_streams
from .phase_optimization import router_coordinates,router_radian_step,circular_router_ema,router_learning_rate_multiplier
from .teacher_projection import projection_enabled,backward_primary_teacher


class CategoryProxies(nn.Module):
    """Training labels only; never used to restrict retrieval candidates."""
    def __init__(self, classes, dimension=64):
        super().__init__()
        self.weight=nn.Parameter(torch.randn(classes,dimension)*.02)

    def forward(self,features):
        return 16*F.normalize(features.float(),dim=-1)@F.normalize(self.weight,dim=-1).T


def load_pool(pool,abo,target):
    report=json.loads((pool/'report.json').read_text())
    if report['manifest_sha256']!=sha256(pool/'manifest.csv'):raise ValueError('Pool manifest changed')
    if report['target_manifest_sha256']!=sha256(target/'data/abo_similarity10_manifest.csv'):raise ValueError('Target exclusion identity changed')
    protected,_=_load_contract(target);blocked={s.product_id for s in protected}
    samples=[]
    for row in csv.DictReader((pool/'manifest.csv').open(encoding='utf-8')):
        path=safe_image(abo,row['image_path'])
        if row['product_id'] in blocked or sha256(path)!=row['image_sha256']:raise ValueError('Pretraining input overlap or changed image')
        samples.append(Sample(row['sample_id'],row['product_id'],int(row['category_id']),row['category'],'pretrain',path))
    if len(samples)!=report['selected_images']:raise ValueError('Pretraining sample count changed')
    if len({s.sample_id for s in samples})!=len(samples):raise ValueError('Duplicate pretraining sample')
    return samples,report


def make_groups(samples):
    groups=defaultdict(lambda:defaultdict(list))
    for i,s in enumerate(samples):groups[s.category_id][s.product_id].append(i)
    if sorted(groups)!=list(range(len(groups))):raise ValueError('Labels must be consecutive')
    return dict(groups)


def sampled_indices(groups,classes_per_batch,products_per_class,rng):
    result=[]
    for category in rng.sample(list(groups),classes_per_batch):
        products=groups[category]
        for product in rng.sample(list(products),products_per_class):result.append(rng.choice(products[product]))
    rng.shuffle(result)
    return result


def checkpoint(model,head,epoch,stage,score):
    return dict(metadata=model.metadata,state_dict=model.state_dict(),epoch=epoch,stage=stage,selection_score=score,
                auxiliary_training_head=head.state_dict(),auxiliary_head_not_used_at_inference=True)


def run_stage(args,stage,output,initial_checkpoint=None):
    seed=training_seed(getattr(args,'seed',42))
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    general=getattr(args,'profile','original') in PROFILES
    rank=getattr(args,'profile','original')=='high_alpha_retrieval' or general
    high=getattr(args,'profile','original') in ('high_alpha','high_alpha_retrieval') or general
    if high:
        from .high_alpha import convert_payload,group_kind,optical_heads,optical_classification_loss,augment,phase_change,phase_shuffle,restore_phase
    cfg_all=json.loads(Path(__file__).with_name('high_alpha.json' if high else 'broad_transfer.json').read_text())
    if rank:
        from .retrieval_training import product_bank,gallery_loss,readout_polish
        overlay=json.loads(Path(__file__).with_name('retrieval_training.json').read_text())
        for key,value in overlay.items():
            if isinstance(value,dict):cfg_all[key].update(value)
            else:cfg_all[key]=value
    if general:cfg_all=overlay_config(cfg_all,args.profile)
    lr_multiplier=learning_rate_multiplier(cfg_all)
    phase_only_group_frozen(cfg_all,1,'phase')  # Validate before loading a model.
    radian_router=router_coordinates(cfg_all)=='radians'
    router_lr_scale=router_learning_rate_multiplier(cfg_all)
    track_clean=cfg_all.get('track_clean_train',False)
    cfg=cfg_all[stage].copy()
    cfg['epochs']=getattr(args,stage+'_epochs') or cfg['epochs'];cfg['steps']=args.steps or cfg['steps']
    device=torch.device(args.device)
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(4)
    model=None
    try:
        target,_=_load_contract(args.target)
        target_train=[s for s in target if s.split=='train'];target_test=[s for s in target if s.split=='test']
        if stage=='pretrain':
            samples,pool_report=load_pool(args.pool,args.abo,args.target)
        else:samples=target_train;pool_report=None
        domain=cfg_all.get('domain_mode')
        target_count=len(target_train)
        if domain:
            external,pool_report=load_pool(args.pool,args.abo,args.target)
            if not pool_report.get('target_types_only'):raise ValueError('Domain training requires a target-mapped pool')
            expected_cap=cfg_all.get('expected_pool_products_per_category')
            if expected_cap is not None and pool_report.get('settings',{}).get('products_per_category')!=expected_cap:
                raise ValueError('Selected pool does not match this refinement profile product cap')
            samples,target_count=combine_training(target,external)
        teacher_vectors=None;teacher_audit=None;feature_targets=None;feature_alignment_audit=None
        if cfg_all.get('teacher_feature_weight',0) and stage!='adapt':
            raise ValueError('Aligned teacher features require original-training adaptation')
        if cfg_all.get('relation_teacher_weight',0) or cfg_all.get('teacher_feature_weight',0):
            from .teacher_relations import load_teacher_cache,gallery_relation_loss
            if not domain or getattr(args,'teacher_cache',None) is None:raise ValueError('Relation KD requires domain training and --teacher-cache')
            teacher_vectors,teacher_audit=load_teacher_cache(args.teacher_cache,samples,args.target,args.pool,device)
        selection_audit=None
        if cfg_all.get('teacher_agreement_external_only',False):
            if teacher_vectors is None or not domain:
                raise ValueError('External selection requires a validated training-only teacher cache')
            from .teacher_relations import select_agreeing_external
            from .io import write_csv
            samples,teacher_vectors,selection_rows=select_agreeing_external(samples,teacher_vectors,target_count)
            write_csv(output/'training_selection.csv',selection_rows)
            selection_audit=dict(policy='keep all original train; external teacher nearest-other-product agrees',
                before_images=len(selection_rows),after_images=len(samples),original_train_retained=target_count,
                selection_sha256=sha256(output/'training_selection.csv'),labels_modified=False,source_files_deleted=False)
        vision_targets=None;vision_audit=None
        if cfg_all.get('vision_patch_teacher_weight',0):
            from .vision_teacher import load_cache,merged_vision,patch_cosine_loss,patch_step_weight
            if stage!='adapt' or not domain or getattr(args,'vision_teacher_cache',None) is None:
                raise ValueError('Visual patch teacher requires domain adapt and --vision-teacher-cache')
            if cfg_all.get('input_preprocessing')!='contain_white':raise ValueError('Visual targets require clean contain_white coordinates')
            patch_step_weight(cfg_all,1,0)
            vision_targets,vision_audit=load_cache(args.vision_teacher_cache,samples,args.target,args.pool,args.assets,device)
        groups=make_groups(samples)
        if len(groups)<cfg['classes_per_batch'] or min(len(g) for g in groups.values())<cfg['products_per_class']:
            raise ValueError('Not enough distinct products/classes for sampling')
        labels=torch.tensor([s.category_id for s in samples],device=device)
        origin=args.assets/'best.pt';start=initial_checkpoint or origin
        payload=torch.load(start,map_location='cpu',weights_only=True)
        if rank and payload['metadata'].get('fusion_alpha_min',0)<=.4:
            raise ValueError('Retrieval continuation must start from a strictly high-alpha checkpoint')
        if high:payload=convert_payload(payload,cfg_all)
        if general:payload=apply_contract(payload,cfg_all)
        auxiliary_payload={k:payload.get(k) for k in ('auxiliary_training_head','auxiliary_head_not_used_at_inference','selection_variant')} if cfg_all.get('restore_auxiliary_source_sha256') else None
        model=OpticalRetrieval(payload['metadata']);model.load_state_dict(payload['state_dict'],strict=True);del payload
        model.to(device)
        from transformers import AutoProcessor
        processor=AutoProcessor.from_pretrained(str(args.assets/'processor'),local_files_only=True)
        head=CategoryProxies(len(groups),dimension=model.readout.output_dimension).to(device)
        if high:head.optical=optical_heads(len(groups)).to(device)
        if auxiliary_payload is not None:
            restore_auxiliary_head(head,auxiliary_payload,sha256(start),cfg_all['restore_auxiliary_source_sha256'])
        if cfg_all.get('preserve_restored_category_proxies',False) and auxiliary_payload is None:
            raise ValueError('Preserving category proxies requires a verified restored auxiliary head')
        del auxiliary_payload
        trainables=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        initial={n:p.detach().cpu().clone() for n,p in trainables}
        optgroups=[]
        for name,p in trainables:
            kind=group_kind(name) if high else parameter_kind(name)
            rate=(cfg[kind+'_lr'] if high or kind!='alpha' else 0.)*lr_multiplier
            if kind=='router':rate*=router_lr_scale
            if name.startswith('frontend.merger_fc2.'):
                from .generalization import merger_learning_rate_multiplier
                rate*=merger_learning_rate_multiplier(cfg_all)
            if name.startswith('frontend.patch.'):
                from .generalization import patch_learning_rate_multiplier
                rate*=patch_learning_rate_multiplier(cfg_all)
            optgroups.append(dict(params=[p],lr=rate,initial_lr=rate,kind=kind,
                                  weight_decay=parameter_decay(name,p,kind,cfg_all.get('electronic_weight_decay',0.))))
        if cfg_all.get('electronic_weight_decay',0.):
            for name,p in head.named_parameters():
                optgroups.append(dict(params=[p],lr=cfg['auxiliary_lr']*lr_multiplier,initial_lr=cfg['auxiliary_lr']*lr_multiplier,kind='auxiliary',
                                      weight_decay=parameter_decay(name,p,'auxiliary',cfg_all['electronic_weight_decay'])))
        else:optgroups.append(dict(params=list(head.parameters()),lr=cfg['auxiliary_lr']*lr_multiplier,initial_lr=cfg['auxiliary_lr']*lr_multiplier,kind='auxiliary'))
        optimizer=torch.optim.AdamW(optgroups,weight_decay=0)
        ema={n:p.detach().clone() for n,p in trainables}
        execution=dict(source_commit=source_commit(),command=sys.argv,pid=os.getpid(),python=sys.version,torch=torch.__version__,
            training_seed=seed,seed_scope='training RNG only; fixed dataset, teacher cache, eval shuffle42/noise123 unchanged; not cross-device bitwise determinism',
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),device=str(device),
            gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
            initial_checkpoint_sha256=sha256(start),accepted_checkpoint_sha256=sha256(start if general else origin),
            target_manifest_sha256=sha256(args.target/'data/abo_similarity10_manifest.csv'),
            pool_manifest_sha256=sha256(args.pool/'manifest.csv') if args.pool else None,
            stage=stage,profile=getattr(args,'profile','original'),config=cfg,common_config=cfg_all,model_audit=model.audit(),
            auxiliary_initialization='restored_pinned_live_checkpoint' if cfg_all.get('restore_auxiliary_source_sha256') else 'fresh_random')
        if domain:
            execution['data_expansion']=dict(pool_report=pool_report,original_train_images=target_count,
                external_images=len(samples)-target_count,train_products=len({s.product_id for s in samples}),
                gallery_products=len({s.product_id for s in target_train}),test_products=len({s.product_id for s in target_test}),
                eval_protocol='Original gallery and test unchanged; expanded products never enter eval gallery')
        execution['training_only_teacher']=teacher_audit
        execution['optimizer_initial_rates_by_kind']={
            kind:sorted({g['initial_lr'] for g in optimizer.param_groups if g['kind']==kind})
            for kind in sorted({g['kind'] for g in optimizer.param_groups})}
        execution['training_only_vision_teacher']=vision_audit
        execution['training_selection']=selection_audit
        execution['protected_optics_source_sha256']=sha256(Path(__file__).with_name('optics.py'))
        write_json(output/'execution.json',execution)
        best_score=(-float('inf'),-float('inf'));history=[]
        if stage=='adapt':
            # Accepted original best is the fallback, not a weaker transferred epoch0.
            transferred={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            # Changed-input/readout controls must not fall back to another contract.
            old=torch.load(start if general else origin,map_location='cpu',weights_only=True)
            if high:old=convert_payload(old,cfg_all)
            if general:old=apply_contract(old,cfg_all)
            model.load_state_dict(old['state_dict']);del old
            base=evaluate(model,processor,target_train,target_test,device,args.batch_size,include_train_metrics=track_clean)
            best_score=(base['hit_at_1'],base['map_at_10'])
            torch.save(checkpoint(model,head,-1,stage,best_score),output/'best.pt')
            history.append(dict(epoch=-1,kind='converted_high_alpha_start' if high else 'accepted_fallback',test=base))
            model.load_state_dict(transferred);del transferred
            current=evaluate(model,processor,target_train,target_test,device,args.batch_size,include_train_metrics=track_clean)
            score=(current['hit_at_1'],current['map_at_10'])
            if score>best_score:
                best_score=score;torch.save(checkpoint(model,head,0,stage,score),output/'best.pt')
            history.append(dict(epoch=0,kind='high_alpha_initial' if high else 'after_external_pretrain',test=current))
            with torch.no_grad():
                features=F.normalize(encode(model,processor,target_train,device,args.batch_size).float(),dim=-1).to(device)
                initial_labels=torch.tensor([s.category_id for s in target_train],device=device)
                proxy_initialization=initialize_category_proxies(head,features,initial_labels,
                    cfg_all.get('preserve_restored_category_proxies',False))
                if cfg_all.get('teacher_feature_weight',0):
                    from .teacher_relations import fit_feature_alignment,aligned_feature_loss,load_feature_alignment,center_teacher_prefix
                    teacher64=F.normalize(teacher_vectors[:,:features.shape[1]].float(),dim=-1)
                    alignment_file=output/'teacher_feature_alignment.pt'
                    reused=bool(cfg_all.get('teacher_alignment_sha256'))
                    center_fraction=cfg_all.get('teacher_feature_center_fraction',0.)
                    teacher_center=None
                    if center_fraction:
                        if reused:raise ValueError('Centered teacher targets require a freshly fitted basis')
                        teacher64,teacher_center=center_teacher_prefix(teacher64,target_count,center_fraction)
                    if reused:
                        if getattr(args,'teacher_alignment',None) is None:
                            raise ValueError('Continuation requires --teacher-alignment')
                        alignment=load_feature_alignment(args.teacher_alignment,cfg_all['teacher_alignment_sha256'],
                            [s.sample_id for s in target_train],teacher_audit['cache_sha256'],
                            cfg_all['teacher_alignment_origin_checkpoint_sha256'],features.shape[1])
                        rotation=alignment['rotation'].to(device)
                    else:
                        rotation=fit_feature_alignment(teacher64[:target_count],features)
                        alignment=dict(rotation=rotation.cpu(),fit_sample_ids=[s.sample_id for s in target_train],
                            source_checkpoint_sha256=execution['initial_checkpoint_sha256'],
                            teacher_cache_sha256=teacher_audit['cache_sha256'],teacher_prefix_dimensions=features.shape[1],
                            teacher_only=True)
                        if teacher_center is not None:
                            alignment.update(teacher_center=teacher_center.cpu(),teacher_center_fraction=center_fraction,
                                teacher_center_fit_scope='original train only')
                    feature_targets=F.normalize(teacher64@rotation,dim=-1).detach()
                    torch.save(alignment,alignment_file)
                    feature_alignment_audit=dict(fit_scope='original train only',fit_images=target_count,
                        prefix_dimensions=features.shape[1],
                        artifact_sha256=sha256(alignment_file),student_weights_rotated=False,at_inference=False)
                    if teacher_center is not None:
                        feature_alignment_audit.update(teacher_center_fraction=center_fraction,
                            teacher_center_norm=float(teacher_center.norm()),teacher_center_fit_images=target_count,
                            teacher_center_at_inference=False)
                    current_cosine=float((feature_targets[:target_count]*features).sum(1).mean())
                    if reused:
                        feature_alignment_audit.update(alignment_reused=True,
                            source_artifact_sha256=cfg_all['teacher_alignment_sha256'],
                            original_fit_checkpoint_sha256=alignment['source_checkpoint_sha256'],
                            current_start_mean_cosine=current_cosine)
                    else:feature_alignment_audit['fit_mean_cosine']=current_cosine
            execution['category_proxy_initialization']=proxy_initialization
            execution['teacher_feature_alignment']=feature_alignment_audit
            write_json(output/'execution.json',execution)
            del features
            write_json(output/'history.json',history);print(json.dumps(history),flush=True)
        for epoch in range(1,cfg['epochs']+1):
            rng,pair_rng=epoch_random_streams(seed,epoch)
            batches=None;domain_phase=None
            if domain:
                domain_phase,batches,active_indices=epoch_batches(samples,target_count,domain,epoch,
                    cfg_all['domain_pretrain_epochs'],cfg['steps'],rng,cfg_all.get('domain_target_products_per_class',2))
                active_samples=[samples[i] for i in active_indices]
                epoch_steps=len(batches)
            else:active_samples=target_train;epoch_steps=cfg['steps']
            if rank:
                bank,bank_labels,product_ids=product_bank(encode(model,processor,active_samples,device,args.batch_size).to(device),active_samples)
                if domain:
                    active_ids=product_ids
                    product_ids=torch.full((len(samples),),-1,device=device,dtype=torch.long)
                    product_ids[torch.tensor(active_indices,device=device)]=active_ids
                if teacher_vectors is not None:
                    teacher_bank,teacher_labels,teacher_ids=product_bank(teacher_vectors[active_indices],active_samples)
                    if not torch.equal(teacher_labels,bank_labels) or not torch.equal(teacher_ids,active_ids):
                        raise ValueError('Teacher/student product bank alignment changed')
            model.train();head.train()
            progress=(epoch-1)/max(1,cfg['epochs']-1)
            scale=min(1.,epoch/2)*(.1+.9*.5*(1+math.cos(math.pi*progress)))
            warm=high and epoch<=cfg['optical_warmup_epochs']
            polish=rank and readout_polish(epoch,cfg['epochs'],cfg)
            for g in optimizer.param_groups:
                frozen=(warm and g['kind'] in ('electronic','adapter')) or (polish and g['kind'] not in ('readout','auxiliary')) or phase_only_group_frozen(cfg_all,epoch,g['kind'])
                g['lr']=0. if frozen else g['initial_lr']*scale
            totals=dict(loss=0.,ce=0.,supcon=0.,correct=0.,optical_auxiliary=0.,gallery_nll=0.,gallery_margin=0.,train_gallery_hit1=0.,sam_loss_gap=0.,view_consistency=0.,relation_kd=0.,teacher_correct_fraction=0.,teacher_confidence=0.,aligned_feature_kd=0.,feature_teacher_correct_fraction=0.);seen=set();paired_seen=set();clean_batches=0
            vision_updates=0;vision_weight_sum=0.
            if vision_targets is not None:totals['vision_patch_kd']=0.
            project_teacher=projection_enabled(cfg_all)
            if project_teacher:
                totals.update(teacher_conflict_fraction=0.,teacher_shared_cosine_before=0.,teacher_shared_cosine_after=0.)
            feature_weight=cfg_all.get('teacher_feature_weight',0.)*min(1.,epoch/max(1,cfg_all.get('teacher_feature_warmup_epochs',3)))
            gt_scale=supervised_loss_scale(epoch,cfg_all)
            view_weight=cfg_all.get('view_consistency_weight',0.)*min(1.,epoch/max(1,cfg_all.get('view_consistency_warmup_epochs',1)))
            teacher_weight=cfg_all.get('relation_teacher_weight',0.)*min(1.,epoch/max(1,cfg_all.get('relation_teacher_warmup_epochs',1)))
            counts={m:torch.zeros(4,device=device) for m in ('vision','language')}
            for step in range(epoch_steps):
                indices=batches[step] if batches is not None else sampled_indices(groups,cfg['classes_per_batch'],cfg['products_per_class'],rng)
                seen.update(indices)
                # Pretrain half noisy; target adaptation one-quarter noisy.
                clean=step%cfg['clean_every']!=0
                clean_batches+=int(clean)
                for mode in (model.vision,model.language):
                    if cfg_all.get('independent_phase_dropout',False):
                        mode.optics.train(True)
                        mode.optics.set_training_noise(not clean)
                    else:mode.optics.train(not clean)
                images=[]
                for i in indices:
                    im=picture(samples[i].image_path,model.metadata.get('input_preprocessing','center_crop'))
                    if high:
                        images.append(augment(im,rng,cfg_all['augmentation']));continue
                    if rng.random()<.5:
                        side=round(224*rng.uniform(.90 if stage=='pretrain' else .96,1.));left,top=[rng.randint(0,224-side) for _ in range(2)]
                        im=im.crop((left,top,left+side,top+side)).resize((224,224),Image.Resampling.BICUBIC)
                    if stage=='pretrain':im=ImageEnhance.Brightness(im).enhance(rng.uniform(.9,1.1))
                    images.append(im)
                batch=inputs(processor,images,device)
                vision_batch=None;vision_weight=0.
                if vision_targets is not None:
                    vision_weight=patch_step_weight(cfg_all,epoch,step)
                    if vision_weight:
                        # Matching clean coordinate view, not the augmented main
                        # images. Existing optical noise mode is intentionally kept.
                        vision_batch=inputs(processor,[picture(samples[i].image_path,'contain_white') for i in indices],device)
                        vision_updates+=1;vision_weight_sum+=vision_weight
                paired_batch=None
                if view_weight:
                    pair_ids=paired_view_indices(samples,groups,indices,pair_rng)
                    paired_seen.update(pair_ids)
                    pair_images=[augment(picture(samples[i].image_path,model.metadata.get('input_preprocessing','center_crop')),
                                         pair_rng,cfg_all['augmentation']) for i in pair_ids]
                    paired_batch=inputs(processor,pair_images,device)
                def objective():
                    with autocast(device):
                        z=model(batch);logits=head(z)
                        ce=F.cross_entropy(logits,labels[indices],label_smoothing=.05)
                        con=supcon(z,labels[indices])
                        loss=(gt_scale*cfg.get('proxy_ce_weight',1.))*ce+(gt_scale*cfg['supcon_weight'])*con+cfg_all['regularization_weight']*regularization(model)
                        optical_aux=optical_classification_loss(model,head.optical,labels[indices]) if high else z.new_zeros(())
                        if high:loss=loss+cfg['optical_auxiliary_weight']*optical_aux
                        result=dict(ce=ce.detach(),supcon=con.detach(),correct=logits.argmax(-1).eq(labels[indices]).float().mean().detach(),optical_auxiliary=optical_aux.detach())
                        if rank:
                            nll,margin,hit=gallery_loss(z,labels[indices],product_ids[indices],bank,bank_labels,
                                class_balance=cfg_all.get('gallery_class_balance',False),
                                full_precision=cfg_all.get('gallery_loss_full_precision',False))
                            loss=loss+(gt_scale*cfg['gallery_nll_weight'])*nll+(gt_scale*cfg['gallery_margin_weight'])*margin
                            result.update(gallery_nll=nll.detach(),gallery_margin=margin.detach(),train_gallery_hit1=hit.detach())
                        if project_teacher:primary_loss=loss;teacher_losses=[]
                        if teacher_vectors is not None and teacher_weight:
                            kd,kd_audit=gallery_relation_loss(z,product_ids[indices],labels[indices],bank,bank_labels,
                                teacher_vectors[indices],teacher_bank,cfg_all['relation_teacher_temperature'],
                                cfg_all.get('relation_teacher_target_temperature'),level=cfg_all.get('teacher_relation_level','product'))
                            loss=loss+teacher_weight*kd
                            if project_teacher:teacher_losses.append(teacher_weight*kd)
                            result.update(relation_kd=kd.detach(),**kd_audit)
                        if feature_targets is not None:
                            kd,correct_fraction=aligned_feature_loss(z,feature_targets[indices],product_ids[indices],labels[indices],
                                teacher_vectors[indices],teacher_bank,bank_labels)
                            loss=loss+feature_weight*kd
                            if project_teacher:teacher_losses.append(feature_weight*kd)
                            result.update(aligned_feature_kd=kd.detach(),feature_teacher_correct_fraction=correct_fraction)
                        if project_teacher:
                            if not teacher_losses:raise RuntimeError('Teacher projection needs a teacher objective')
                            result.update(primary_loss=primary_loss,teacher_loss=sum(teacher_losses))
                    # Preserve primary-view router statistics before alternate-view forward.
                    result['selected']={m:getattr(model,m).optics.router.last['selected_mask'].detach().sum(0) for m in counts}
                    if vision_batch is not None:
                        with autocast(device):visual=merged_vision(model,vision_batch)
                        patch_loss=patch_cosine_loss(visual,vision_targets[indices])
                        loss=loss+vision_weight*patch_loss
                        result['vision_patch_kd']=patch_loss.detach()
                    if paired_batch is not None:
                        with autocast(device):paired_features=model(paired_batch)
                        alignment=view_consistency_loss(z,paired_features)
                        loss=loss+view_weight*alignment
                        result['view_consistency']=alignment.detach()
                    result['loss']=loss
                    return result
                rho=cfg_all.get('sam_rho',0.)*min(1.,epoch/cfg_all.get('sam_warmup_epochs',1))
                result,sam_diagnostics=(backward_primary_teacher(objective,optimizer) if project_teacher
                                        else backward_with_sam(objective,optimizer,rho))
                for n,p in trainables:
                    if (not high and parameter_kind(n)=='alpha') or (warm and group_kind(n) in ('electronic','adapter')) or (polish and group_kind(n)!='readout'):p.grad=None
                for g in optimizer.param_groups:
                    if phase_only_group_frozen(cfg_all,epoch,g['kind']):
                        for p in g['params']:p.grad=None
                with router_radian_step(optimizer,radian_router):
                    torch.nn.utils.clip_grad_norm_([p for _,p in trainables]+list(head.parameters()),1.)
                    optimizer.step()
                with torch.no_grad():
                    for n,p in trainables:
                        if p.grad is None:ema[n].copy_(p)
                        elif radian_router and group_kind(n)=='router':circular_router_ema(ema[n],p,cfg_all['ema'])
                        else:ema[n].mul_(cfg_all['ema']).add_(p,alpha=1-cfg_all['ema'])
                for key in totals:
                    if key=='sam_loss_gap':totals[key]+=sam_diagnostics['loss_gap']
                    elif key in sam_diagnostics:totals[key]+=sam_diagnostics[key]
                    elif key in result:totals[key]+=float(result[key].detach())
                for m in counts:counts[m]+=result['selected'][m]
            if high and not all(.4<a<=.8 for values in model.audit()['alpha'].values() for a in values):
                raise RuntimeError('Strict high-alpha contract violated')
            row=dict(epoch=epoch,stage=stage,optical_warmup=warm,phase_only_warmup=phase_only_group_frozen(cfg_all,epoch,'electronic'),readout_polish=polish,losses={k:v/epoch_steps for k,v in totals.items()},
                     unique_images=len(seen),paired_unique_images=len(paired_seen),view_consistency_weight=view_weight,relation_teacher_weight=teacher_weight,teacher_feature_weight=feature_weight,supervised_loss_scale=gt_scale,
                     clean_batches=clean_batches,alpha=model.audit()['alpha'],sam_rho=rho,sam_rho_target=cfg_all.get('sam_rho',0.),
                     router_selected_fraction={m:(c/(epoch_steps*cfg['classes_per_batch']*cfg['products_per_class'])).cpu().tolist() for m,c in counts.items()})
            if vision_targets is not None:
                row['vision_patch_supervision']=dict(updates=vision_updates,
                    mean_loss_on_updates=totals['vision_patch_kd']/max(1,vision_updates),
                    mean_weight_per_step=vision_weight_sum/epoch_steps,
                    extra_forward='existing V only, clean image with current optical noise mode; no inference change')
            if domain:
                row['data_coverage']=dict(domain_phase=domain_phase,steps=epoch_steps,
                    mixed_target_products_per_class=cfg_all.get('domain_target_products_per_class',2),
                    unique_products=len({samples[i].product_id for i in seen}),
                    active_products=len({s.product_id for s in active_samples}),
                    target_images_seen=sum(i<target_count for i in seen),external_images_seen=sum(i>=target_count for i in seen))
            torch.save(checkpoint(model,head,epoch,stage,-row['losses']['loss']),output/'last.pt')
            live={n:p.detach().clone() for n,p in trainables}
            with torch.no_grad():
                for n,p in trainables:p.copy_(ema[n])
            if stage=='pretrain':
                score=(-row['losses']['loss'],0.)
                if score>best_score:
                    best_score=score;torch.save(dict(checkpoint(model,head,epoch,stage,score),selection_variant='ema'),output/'best.pt')
            elif epoch%cfg_all['test_every']==0 or epoch==cfg['epochs']:
                metrics=evaluate(model,processor,target_train,target_test,device,args.batch_size,include_train_metrics=track_clean);row['test']=metrics
                score=(metrics['hit_at_1'],metrics['map_at_10'])
                if score>best_score:
                    best_score=score;torch.save(dict(checkpoint(model,head,epoch,stage,score),selection_variant='ema'),output/'best.pt')
                if rank:
                    with torch.no_grad():
                        for n,p in trainables:p.copy_(live[n])
                    raw_metrics=evaluate(model,processor,target_train,target_test,device,args.batch_size,include_train_metrics=track_clean);row['test_live']=raw_metrics
                    raw_score=(raw_metrics['hit_at_1'],raw_metrics['map_at_10'])
                    if raw_score>best_score:
                        best_score=raw_score;torch.save(dict(checkpoint(model,head,epoch,stage,raw_score),selection_variant='live'),output/'best.pt')
            with torch.no_grad():
                for n,p in trainables:p.copy_(live[n])
            del live
            write_json(output/'parameter_updates.json',{n:float((p.detach().cpu()-initial[n]).square().mean().sqrt()) for n,p in trainables})
            history.append(row);write_json(output/'history.json',history);print(json.dumps(row),flush=True)
            if track_clean:write_learning_curves(history,output)
        payload=torch.load(output/'best.pt',map_location=device,weights_only=True)
        model.load_state_dict(payload['state_dict']);selected_epoch=payload['epoch'];selected_variant=payload.get('selection_variant','initial_or_ema');del payload
        report=dict(status='complete',stage=stage,selected_epoch=selected_epoch,model_audit=model.audit(),
                    training_seed=seed,
                    selected_variant=selected_variant,
                    auxiliary_head_at_inference=False,test_selected=stage=='adapt',
                    selection_note='best EMA snapshot indexed by live training loss' if stage=='pretrain' else 'target test Hit@1 then mAP; accepted best included')
        report.update(training_only_teacher=teacher_audit,teacher_at_inference=False,training_selection=selection_audit,
                      training_only_vision_teacher=vision_audit,
                      teacher_feature_alignment=feature_alignment_audit,
                      protected_optics_source_sha256=execution['protected_optics_source_sha256'])
        if high:report['selection_note']='Only alpha>=0.4 candidates, including converted initial checkpoint; low-alpha 70.21% is NOT a fallback'
        if general:report['selection_note']='Initial/live/EMA within this run input/readout contract only; explicit initial checkpoint is the fallback. Selected epoch -1 is not a new training improvement.'
        if stage=='adapt':
            report['metrics']=evaluate(model,processor,target_train,target_test,device,args.batch_size,output,include_train_metrics=track_clean)
            model.set_remove_optical(True)
            report['remove_optical_same_weights']=evaluate(model,processor,target_train,target_test,device,args.batch_size)
            model.set_remove_optical(False)
            report['optical_removal_hit1_drop_percentage_points']=100*(report['metrics']['hit_at_1']-report['remove_optical_same_weights']['hit_at_1'])
            if high:
                report['phase_change_from_start']=phase_change(model,initial)
                saved=phase_shuffle(model)
                try:report['phase_pixels_shuffled_seed42']=evaluate(model,processor,target_train,target_test,device,args.batch_size)
                finally:restore_phase(model,saved)
                del saved
                torch.manual_seed(123)
                for modality in (model.vision,model.language):modality.optics.eval_ccd_noise=True
                try:report['mild_expert_global_ccd_noise_seed123']=evaluate(model,processor,target_train,target_test,device,args.batch_size)
                finally:
                    for modality in (model.vision,model.language):modality.optics.eval_ccd_noise=False
                report['noise_evaluation_scope']='Expert/global CCD only; gain and truncated noise from metadata; no eval DC/bypass, router clean; single seed'
        if device.type=='cuda':report['gpu_memory_mib']=dict(peak_allocated=torch.cuda.max_memory_allocated(device)/2**20,peak_reserved=torch.cuda.max_memory_reserved(device)/2**20)
        preview(model,output);write_json(output/'final_report.json',report);print(json.dumps(report),flush=True)
        return output/'best.pt'
    except BaseException as exc:
        write_json(output/'failure.json',dict(error=str(exc),type=type(exc).__name__));raise
    finally:
        if model is not None:del model
        if device.type=='cuda':torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['pretrain','adapt','chain'],default='chain')
    p.add_argument('--profile',choices=['original','high_alpha','high_alpha_retrieval',*PROFILES],default='original')
    p.add_argument('--assets',type=Path,required=True);p.add_argument('--target',type=Path,required=True)
    p.add_argument('--abo',type=Path);p.add_argument('--pool',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--device',default='cuda',choices=['cuda','cpu'])
    p.add_argument('--teacher-cache',type=Path,help='Training-only frozen teacher vectors; not loaded for other profiles or inference')
    p.add_argument('--teacher-alignment',type=Path,help='Pinned training-only teacher basis for teacher_continue')
    p.add_argument('--vision-teacher-cache',type=Path,help='Complete train-only clean visual-token cache, only for vision_patch profile')
    p.add_argument('--pretrain-epochs',type=int);p.add_argument('--adapt-epochs',type=int);p.add_argument('--steps',type=int)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--seed',type=training_seed,default=42,help='Training RNG only; no data split or evaluation seed change')
    args=p.parse_args();verify_assets(args.assets)
    if (args.profile=='domain_distill_vision_patch')!=(args.vision_teacher_cache is not None):
        p.error('--vision-teacher-cache is required only for domain_distill_vision_patch')
    if (args.profile in PINNED_TEACHER_PROFILES) != (args.teacher_alignment is not None):
        p.error('--teacher-alignment is required only for pinned teacher continuation profiles')
    if (args.profile.startswith('high_alpha') or args.profile in PROFILES) and args.mode!='adapt':p.error('High-alpha/generalization profiles support target adapt only')
    if args.mode in ('pretrain','chain') and (args.abo is None or args.pool is None):p.error('--abo and --pool required')
    if args.profile.startswith('domain_') and (args.abo is None or args.pool is None):p.error('Domain expansion requires --abo and --pool')
    if args.mode=='adapt' and args.checkpoint is None:p.error('--checkpoint required for transfer adaptation')
    if any(x is not None and x<1 for x in (args.pretrain_epochs,args.adapt_epochs,args.steps,args.batch_size)):p.error('Positive counts required')
    if args.output.exists():raise FileExistsError(args.output)
    if args.mode=='chain':
        pretrained=run_stage(args,'pretrain',args.output/'pretrain',args.checkpoint)
        run_stage(args,'adapt',args.output/'adapt',pretrained)
    else:run_stage(args,args.mode,args.output,args.checkpoint)


if __name__=='__main__':main()
