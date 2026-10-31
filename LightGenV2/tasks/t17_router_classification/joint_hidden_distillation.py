"""Training-only shared joint-teacher hidden targets; never an inference module."""
import json
from pathlib import Path
import numpy as np
import torch
from .train_original_test_teacher import teacher_head
from .semantic_distillation import semantic_loss,balanced_aux_weight
from .train_figure_revision import training_forward
from .train import write
from LightGenV2.tasks.t11_lifelong_optics.crc9_data import sha256


def hidden_payload(aux):
    if aux is None:return {}
    return dict(training_only_joint_hidden_projection={k:v.detach().cpu().clone() for k,v in aux['projection'].state_dict().items()},training_only_joint_hidden_optimizer=aux['optimizer'].state_dict())


def add_hidden_objective(base,features,aux):
    if aux is None:return base,None
    value=semantic_loss(features,aux['target'],aux['projection'])
    return base+aux['weight']*value,float(value.detach())


def prepare_joint_hidden(raw,cfg,out):
    info=cfg.get('joint_hidden_distillation')
    if not info:return None
    if cfg.get('training_domain') or not cfg['teacher'].get('joint_cache_run') or not cfg.get('source_test_distillation'):raise ValueError('Explicit joint teacher and TEST-soft protocol required')
    task=Path(__file__).resolve().parent;teacher=task/cfg['teacher']['joint_cache_run'];manifest=json.loads((teacher/'cache_manifest.json').read_text());origin=task/info['initialization_run'];diagnosis=json.loads((origin/'diagnosis.json').read_text());initial_manifest=json.loads((origin/'manifest.json').read_text())
    if diagnosis['parent_checkpoint_sha256']!=cfg['parent_checkpoint_sha256'] or diagnosis['teacher_checkpoint_sha256']!=manifest['best_checkpoint.pt']:raise ValueError('Joint hidden initialization parent/teacher mismatch')
    if sha256(origin/'training_only_projection_initialization.pt')!=initial_manifest['files']['training_only_projection_initialization.pt'] or sha256(teacher/'best_checkpoint.pt')!=manifest['best_checkpoint.pt']:raise ValueError('Hidden initialization/teacher SHA mismatch')
    initialization=torch.load(origin/'training_only_projection_initialization.pt',map_location='cpu',weights_only=False);head=teacher_head(cfg['num_classes']).eval();head.load_state_dict(torch.load(teacher/'best_checkpoint.pt',map_location='cpu',weights_only=False)['head'],strict=True);head.requires_grad_(False);mean=initialization['teacher_TRAIN8_mean'];std=initialization['teacher_TRAIN8_std'];targets={};sources={}
    for split in ('train','test'):
        feature=teacher/f'features_{split}.npy';identity=teacher/f'identities_{split}.npz'
        if sha256(feature)!=manifest[feature.name] or sha256(identity)!=manifest[identity.name]:raise ValueError('Joint hidden target source SHA')
        ids=np.load(identity)
        if any(not np.array_equal(ids[k],raw[split][k]) for k in ('ids','labels','domains')):raise ValueError('Joint hidden original identity/order mismatch')
        values=np.load(feature,mmap_mode='r');flat=values.reshape(-1,512);hidden=[]
        with torch.no_grad():
            for start in range(0,len(flat),1024):hidden.append(head[:-1](torch.from_numpy(flat[start:start+1024].copy()).float()))
            normalized=((torch.cat(hidden).reshape(len(values),8,256)-mean)/std).detach()
        if not torch.isfinite(normalized).all():raise RuntimeError('Nonfinite shared hidden targets')
        name=f'{split.upper()}_joint_teacher_hidden.npy';np.save(out/name,normalized.numpy());targets[split]=normalized.cuda();sources[split]=dict(source_features_sha256=manifest[feature.name],source_ID_sha256=manifest[identity.name],target_sha256=sha256(out/name),shape=list(normalized.shape))
    projection=torch.nn.Linear(784,256,bias=False).cuda()
    with torch.no_grad():projection.weight.copy_(initialization['training_only_projection_weight'])
    torch.save(dict(projection=projection.state_dict(),teacher_TRAIN8_mean=mean,teacher_TRAIN8_std=std),out/'joint_hidden_initialization.pt')
    record=dict(teacher_checkpoint_sha256=manifest['best_checkpoint.pt'],parent_checkpoint_sha256=cfg['parent_checkpoint_sha256'],target='one jointly trained teacher GELU256 before final Linear',normalization='original TRAIN all8 views only; frozen diagnostic mean/std',initialization='TRAIN-code0 affine Ridge, intercept folded into mean-one CCD',initialization_sha256=sha256(origin/'training_only_projection_initialization.pt'),extra_training_parameters=200704,projection_absent_from_inference=True,teacher_absent_from_inference=True,TEST_hard_GT_in_loss=False,targets=sources)
    return dict(targets=targets,projection=projection,optimizer=torch.optim.Adam(projection.parameters(),lr=info['projection_lr']),record=record)


def calibrate_joint_hidden(aux,model,fields,labels,domains,teacher_logits,anchor,cfg):
    from .train_f_balanced_joint import balanced_objective
    selected=torch.cat([torch.where((labels==c)&(domains==d))[0][:1] for d in (0,1) for c in range(10)])
    if len(selected)!=20:raise ValueError('Both TRAIN domains/classes required')
    snapshot={k:v.detach().clone() for k,v in model.state_dict().items()};model.train();output=training_forward(model,fields[selected],cfg,True)
    main,_=balanced_objective(output,labels[selected],domains[selected],teacher_logits[selected,0],model.shared_head,anchor,cfg)
    feature=semantic_loss(output['ccd_features'],aux['targets']['train'][selected,0],aux['projection']);phases=[p for n,p in model.named_parameters() if n!='shared_head.weight'];norms=[]
    for loss in (main,feature):
        grads=torch.autograd.grad(loss,phases,retain_graph=True);total=0.
        for p,g in zip(phases,grads):
            fraction=p.detach().double().sigmoid();jacobian=(2*torch.pi*fraction*(1-fraction)).clamp_min(1e-20);total+=float((g.double()/jacobian).square().sum())
        norms.append(total**.5)
    aux['weight']=balanced_aux_weight(*norms);aux['record'].update(calibration_TRAIN_images=20,physical_main_gradient_norm=norms[0],physical_hidden_gradient_norm=norms[1],weight=aux['weight'],calibration='20% main physical phase norm, clip .001..1; TRAIN only')
    if not all(torch.equal(v,snapshot[k]) for k,v in model.state_dict().items()):raise RuntimeError('Calibration changed student')
    model.eval()
