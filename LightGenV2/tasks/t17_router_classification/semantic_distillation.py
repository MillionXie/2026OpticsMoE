"""Training-only teacher hidden-feature alignment; canonical LG inference is untouched."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .train_original_test_teacher import teacher_head
from .train import write
from LightGenV2.tasks.t11_lifelong_optics.crc9_data import sha256


def semantic_loss(features,targets,projection):
    return F.mse_loss(projection(features),targets.detach().to(features))


def ridge_projection(features,targets,ridge_ratio=.001):
    """Fit an affine TRAIN map, folding its intercept into mean-one CCD features."""
    x=features.detach().double();y=targets.detach().double()
    if not torch.allclose(x.mean(1),torch.ones(len(x),device=x.device,dtype=x.dtype),atol=2e-6,rtol=0):raise ValueError('Mean-one canonical CCD features required')
    mx=x.mean(0);my=y.mean(0);xc=x-mx;yc=y-my;gram=xc.T@xc/len(x);ridge=ridge_ratio*gram.diagonal().mean().clamp_min(1e-12)
    weight=torch.linalg.solve(gram+ridge*torch.eye(x.shape[1],device=x.device,dtype=x.dtype),xc.T@yc/len(x)).T
    intercept=my-mx@weight.T;folded=weight+intercept[:,None]/x.shape[1]
    return folded.float(),dict(ridge=float(ridge),training_images=len(x),affine_intercept_absorbed_in_mean_one_features=True)


def balanced_aux_weight(main_norm,aux_norm):
    if main_norm<=0 or aux_norm<=0:raise ValueError('Positive physical-gradient norms required')
    return min(1.,max(.001,.2*main_norm/aux_norm))


def prepare_semantic(raw,cfg,out,model,train_fields):
    info=cfg.get('semantic_distillation')
    if not info:return None
    if not cfg.get('source_test_distillation') or cfg['architecture']!='LG':raise ValueError('Explicit source-only TEST-exposed LG is required')
    task=Path(__file__).resolve().parent;teacher_dir=task/cfg['teacher']['cache_run'];manifest=json.loads((teacher_dir/'cache_manifest.json').read_text())
    if sha256(teacher_dir/'best_checkpoint.pt')!=manifest['best_checkpoint.pt']:raise ValueError('Semantic teacher PT SHA differs')
    head=teacher_head(cfg['num_classes']).cuda();ck=torch.load(teacher_dir/'best_checkpoint.pt',map_location='cpu',weights_only=False);head.load_state_dict(ck['head'],strict=True);head.requires_grad_(False).eval()
    targets={}
    for split in ('train','test'):
        path=teacher_dir/f'features_{split}.npy';ids_path=teacher_dir/f'identities_{split}.npz'
        if sha256(path)!=manifest[path.name] or sha256(ids_path)!=manifest[ids_path.name]:raise ValueError('Teacher hidden source SHA differs')
        ids=np.load(ids_path);part=raw[split];mask=part['domains']==(0 if cfg['training_domain']=='RGB' else 1)
        for key in ('ids','labels','domains'):
            if not np.array_equal(ids[key],part[key][mask]):raise ValueError('Hidden teacher domain/ID differs')
        values=np.load(path,mmap_mode='r');hidden=[]
        with torch.no_grad():
            flat=values.reshape(-1,512)
            for start in range(0,len(flat),1024):hidden.append(head[:-1](torch.from_numpy(flat[start:start+1024].copy()).cuda().float()))
        targets[split]=torch.cat(hidden).reshape(len(values),8,256).detach()
    # Statistics, projection fitting and coefficient calibration use TRAIN only.
    mean=targets['train'].mean((0,1));std=targets['train'].std((0,1)).clamp_min(.001)
    targets={key:((value-mean)/std).detach() for key,value in targets.items()}
    features=[];model.eval()
    with torch.no_grad():
        for start in range(0,len(train_fields),cfg['eval_batch']):features.append(model.optical_features(train_fields[start:start+cfg['eval_batch']])['ccd_features'])
    weight,record=ridge_projection(torch.cat(features),targets['train'][:,0]);projection=torch.nn.Linear(784,256,bias=False).cuda()
    with torch.no_grad():projection.weight.copy_(weight)
    for key,value in targets.items():np.save(out/f'{key.upper()}_teacher_hidden.npy',value.cpu().numpy())
    torch.save(dict(projection=projection.state_dict(),teacher_mean=mean.cpu(),teacher_std=std.cpu()),out/'semantic_initialization.pt')
    record.update(target='teacher GELU256 before final classifier; frozen teacher',extra_training_parameters=projection.weight.numel(),inference_parameters=sum(p.numel() for p in model.parameters()),projection_absent_from_inference=True,teacher_checkpoint_sha256=sha256(teacher_dir/'best_checkpoint.pt'),teacher_inference_absent=True,statistics_parts=['source TRAIN only'],calibration_rule='clip(.2 * physical main-gradient norm / feature-gradient norm, .001, 1) on20 class-balanced source TRAIN originals')
    write(out/'semantic_manifest.json',record)
    del head,features
    return dict(projection=projection,targets=targets,optimizer=torch.optim.Adam(projection.parameters(),lr=info['projection_lr']),record=record)


def calibrate_semantic(aux,model,train_fields,labels,teacher_logits,phases,anchor,cfg):
    from .alternating_readout import anchored_classification
    chosen=torch.cat([torch.where(labels==c)[0][:2] for c in range(cfg['num_classes'])]);output=model(train_fields[chosen])
    main=anchored_classification(output['raw_logits'],labels[chosen],model.shared_head.weight,anchor,cfg['alternating_head']['anchor_strength'],cfg['label_smoothing'],teacher_logits[chosen,0],cfg['teacher']['weight'],cfg['teacher']['temperature'])
    hidden=semantic_loss(output['ccd_features'],aux['targets']['train'][chosen,0],aux['projection'])
    gradients=[torch.autograd.grad(loss,phases,retain_graph=True) for loss in (main,hidden)];norms=[]
    for values in gradients:
        total=0.
        for p,g in zip(phases,values):
            fraction=p.detach().double().sigmoid();jac=(2*torch.pi*fraction*(1-fraction)).clamp_min(1e-20);total+=float((g.double()/jac).square().sum())
        norms.append(total**.5)
    aux['weight']=balanced_aux_weight(*norms);aux['record'].update(calibration_TRAIN_images=len(chosen),physical_main_gradient_norm=norms[0],physical_feature_gradient_norm=norms[1],auxiliary_weight=aux['weight'])


def semantic_checkpoint(aux):
    if aux is None:return {}
    return dict(training_only_semantic_projection={k:v.detach().cpu().clone() for k,v in aux['projection'].state_dict().items()},training_only_semantic_optimizer=aux['optimizer'].state_dict())
