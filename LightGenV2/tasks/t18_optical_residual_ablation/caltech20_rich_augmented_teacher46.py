"""Frozen public TRAIN teacher on the student's actual augmented RGB view."""
import types,inspect,hashlib
import caltech20_rich_view_target_consistency46 as baseline
import caltech20_rich_input_dropout46 as source_closure
for n,v in vars(baseline).items():
    if not n.startswith('__') and n not in ('baseline','source_closure','original','old'):globals()[n]=v
original=baseline.original
EXTRA=('caltech20_rich_augmented_teacher46.py','check_caltech20_rich_augmented_teacher46.py','audit_caltech20_rich_augmented_teacher46.py','CALTECH20_RICH_AUGMENTED_TEACHER46_PROTOCOL_20261018.md');FILES=(EXTRA[0],baseline.FILES[1],EXTRA[1],EXTRA[2],EXTRA[3])
def config():
    c=baseline.config();case=dict(fresh=False,epochs=5,electronic_warm_epochs=0,phase_lr=.00015,full_rms_gradient=True,consistency_target='KD_both');c.update(run='caltech20_rich20_actual_augmented_TRAIN_teacher46_micro4_local5_20261018',cases={f'L{d}_augmented_teacher':dict(case,depth=d) for d in (4,6)},microbatch_size=4,online_frozen_teacher_TRAIN_only=True,teacher_and_student_same_augmented_RGB100=True,teacher_student_KD_weights=[.5,.5]);return c
OUT=HERE/'runs/simulation'/config()['run'];CPU=HERE/'runs/smoke/caltech20_rich_augmented_teacher46_CPU_20261018'
def sources():
    z=source_closure.sources()
    for n in EXTRA:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return z
for name in ('identity','claim','rates','seed_for','load_predictions','groups','js_loss','numeric_gate_kernel','campaign','feature_loss','target_data','initialize','gate_kernel','complete_gate_contract','fit_kernel','select_kernel','select','teacher_targets','gradient_record_kernel','zoom_plan','plan_hash','transform','mixed_kernel','record_kernel','gate_original','allocated_terms','ready_kernel'):
    f=getattr(baseline,name);globals()[name]=types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__)
for name in ('incumbent','prototype_data','teacher_data'):
    f=getattr(baseline,name).__wrapped__;globals()[name]=lru_cache(None)(types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__))
old=types.SimpleNamespace(**vars(baseline.old));old.kd=types.SimpleNamespace(**vars(baseline.old.kd));old.kd.teacher_targets=teacher_targets
ns=dict(baseline.old.targets.__globals__);ns['kd']=old.kd;old.targets=types.FunctionType(baseline.old.targets.__code__,ns,'unchanged_clean_teacher_targets',baseline.old.targets.__defaults__,baseline.old.targets.__closure__)
TEACHER=None;TEACHER_TRANSFORM=None;TEACHER_HEAD=None;TEACHER_INITIAL=None;TEACHER_FORWARDED_IMAGES=0;AUG_RGB=None;AUG_Q=None;AUG_MASS=None;AUG_FEATURES=None;AUG_RECORDS={}
def teacher_RGB100(images,theta):
    import resisc20_input_encoder as encoder
    rgb=torch.nn.functional.interpolate(images.float()/255,(100,100),mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
    return encoder.old.optical.b.augment(rgb,theta) if theta is not None else rgb
def capture_augmented_RGB(images,theta):
    global AUG_RGB
    AUG_RGB=teacher_RGB100(images,theta).detach()
core=types.SimpleNamespace(**vars(baseline.core));f=core.mixed;ns=dict(f.__globals__);ns['capture_augmented_RGB']=capture_augmented_RGB;text=inspect.getsource(f);site="x=base.encode(image.clamp(0,1)*255,torch.tensor(theta[idx],device=device),100)";assert text.count(site)==1;text=text.replace(site,"capture_augmented_RGB(image.clamp(0,1)*255,torch.tensor(theta[idx],device=device));"+site);exec(compile(text,__file__+':same_augmented_input_with_teacher_RGB_tap','exec'),ns);core.mixed=ns['mixed']
def teacher_runtime(device):
    global TEACHER,TEACHER_TRANSFORM,TEACHER_HEAD,TEACHER_INITIAL
    if TEACHER is None:
        from torchvision.models import resnet18,ResNet18_Weights
        cfg=teacher_source.config();assert s.sha(cfg['teacher_checkpoint'])==cfg['teacher_checkpoint_sha256'];head=teacher_source.OUT/'selected_head.pt';assert s.sha(head)==s.read(teacher_source.OUT/'fit_result.json')['head_sha256'];TEACHER_HEAD=torch.load(head,map_location='cpu',weights_only=False);TEACHER=resnet18(weights=None);TEACHER.load_state_dict(torch.load(cfg['teacher_checkpoint'],map_location='cpu',weights_only=True));TEACHER.fc=torch.nn.Identity();TEACHER.eval().requires_grad_(False);TEACHER_INITIAL=s.digest(TEACHER.state_dict());TEACHER.to(device);TEACHER_TRANSFORM=ResNet18_Weights.IMAGENET1K_V1.transforms()
    assert not TEACHER.training and all(not p.requires_grad and p.grad is None for p in TEACHER.parameters());return TEACHER
def augmented_targets(features,labels,epoch,batch,cfg,depth):
    h=TEACHER_HEAD;z=((features-h['mean'])/h['std']).astype(np.float32).astype(float);logits=z@h['model']['weight'].double().numpy().T+h['model']['bias'].double().numpy();raw=np.exp(logits-logits.max(1,keepdims=True));raw/=raw.sum(1,keepdims=True);valid=((logits.argmax(1)==labels)&(raw.max(1)>=.5)).astype(np.float32);temp=cfg['teacher_temperature_by_depth'][str(depth)];soft=np.exp(logits/temp-(logits/temp).max(1,keepdims=True));soft=(soft/soft.sum(1,keepdims=True)).astype(np.float32);rng=np.random.default_rng(np.random.SeedSequence([17,epoch,batch,20261013,4]));lam=float(rng.beta(cfg['mixup_alpha'],cfg['mixup_alpha']));lam=max(lam,1-lam);perm=rng.permutation(len(labels));mass=lam*valid+(1-lam)*valid[perm];q=(lam*valid[:,None]*soft+(1-lam)*valid[perm,None]*soft[perm])/mass[:,None].clip(1e-12);return q.astype(np.float32),mass.astype(np.float32),dict(correct_fraction=float((logits.argmax(1)==labels).mean()),gate_fraction=float(valid.mean()),temperature=temp)
f=baseline.mixed;mixed_original=types.FunctionType(f.__code__,globals(),'same_student_views_capture_matching_RGB',f.__defaults__,f.__closure__)
def mixed(data,idx,theta,epoch,batch,cfg,device):
    global AUG_Q,AUG_MASS,AUG_FEATURES,TEACHER_FORWARDED_IMAGES
    result=mixed_original(data,idx,theta,epoch,batch,cfg,device);teacher=teacher_runtime(device)
    with torch.no_grad():features=teacher(TEACHER_TRANSFORM(AUG_RGB)).detach().cpu().numpy().astype(np.float32)
    assert features.shape==(len(idx),512) and np.isfinite(features).all();TEACHER_FORWARDED_IMAGES+=len(idx);labels=data[1][idx].numpy();q,mass,info=augmented_targets(features,labels,epoch,batch,cfg,ACTIVE_DEPTH);AUG_Q=torch.tensor(q,device=device);AUG_MASS=torch.tensor(mass,device=device);AUG_FEATURES=features;AUG_RECORDS[(epoch,batch)]=dict(features=features,q=q,mass=mass,labels=labels.copy(),info=info);return result
text=inspect.getsource(baseline.gradient_kernel);site="ka=old.kd.KD_loss(pa,q[sl],mass[sl],cfg)";assert text.count(site)==1;text=text.replace(site,"ka=old.kd.KD_loss(pa,AUG_Q[sl],AUG_MASS[sl],cfg)");exec(compile(text,__file__+':actual_augmented_teacher_KD_only','exec'),globals())
f=baseline.gradients;gradients_original=types.FunctionType(f.__code__,globals(),'same_scalar_and_phase_trace',f.__defaults__,f.__closure__)
def gradients(model,x,y,q,mass,mask,weights,cfg,micro):
    result=gradients_original(model,x,y,q,mass,mask,weights,cfg,micro);assert AUG_Q.shape==q.shape and AUG_MASS.shape==mass.shape;model.view_info.update(augmented_teacher_q_SHA256=hashlib.sha256(AUG_Q.detach().cpu().numpy().astype('<f4').tobytes()).hexdigest(),augmented_teacher_mass_SHA256=hashlib.sha256(AUG_MASS.detach().cpu().numpy().astype('<f4').tobytes()).hexdigest(),augmented_teacher_features_SHA256=hashlib.sha256(AUG_FEATURES.astype('<f4').tobytes()).hexdigest(),augmented_teacher_gate_fraction=float((AUG_MASS>0).float().mean()),teacher_reference_augmented_RGB100=True);return result
def teacher_frozen():
    assert TEACHER is not None and all(not p.requires_grad and p.grad is None for p in TEACHER.parameters()) and not TEACHER.training;digest=s.digest({n:v.detach().cpu() for n,v in TEACHER.state_dict().items()});assert digest==TEACHER_INITIAL;return digest
def gate(depth,fresh,data,case):
    proof=gate_original(depth,fresh,data,case);proof.update(teacher_backbone_frozen_SHA256=teacher_frozen(),teacher_only_TRAIN=True,teacher_reference_input_remains_clean=False,clean_teacher_cached_augmented_teacher_online=True,student_target_allocation='KD_both',teacher_gate_forwarded_images=TEACHER_FORWARDED_IMAGES);return proof
f=baseline.fit;fit_original=types.FunctionType(f.__code__,globals(),'same_five_epoch_fit',f.__defaults__,f.__closure__)
def fit(key,gpu):
    global AUG_RECORDS
    AUG_RECORDS={};fit_original(key,gpu);rows=[];parts={k:[] for k in ('features','q','mass','labels')};offset=0
    for (epoch,batch),q in sorted(AUG_RECORDS.items()):
        n=len(q['labels']);assert n==(5 if batch==196 else 16);rows.append(dict(epoch=epoch,batch=batch,n=n,offset=offset,**q['info']));offset+=n
        for k in parts:parts[k].append(q[k])
    assert len(rows)==197*5 and offset==3141*5;path=OUT/key/'augmented_teacher_trace.npz';base.save_np(path,{k:np.concatenate(v) for k,v in parts.items()});s.save(OUT/key/'augmented_teacher_rows.json',rows,exclusive=True);p=OUT/key/'fit_result.json';q=s.read(p);q.update(augmented_teacher_trace_SHA256=s.sha(path),augmented_teacher_rows_SHA256=s.sha(OUT/key/'augmented_teacher_rows.json'),teacher_backbone_before=TEACHER_INITIAL,teacher_backbone_after=teacher_frozen(),teacher_optimizer_steps=0,teacher_forwarded_TRAIN_views=offset,teacher_forwarded_TRAIN_images_including_gate=TEACHER_FORWARDED_IMAGES,extra_student_inference_parameters=0);s.save(p,q)
def ready():
    q=s.read(CPU/'checks.json');assert q['independent_augmented_teacher_KD_gradient'] and q['teacher_RGB_matches_student_encoder'] and not CPU.with_name(CPU.name+'.failure.json').exists();ready_kernel()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','campaign'],required=True);p.add_argument('--key');p.add_argument('--gpu');a=p.parse_args()
    if a.key:select(a.key)
    try:campaign() if a.phase=='campaign' else fit(a.key,a.gpu)
    except BaseException as e:
        f=OUT/a.key if a.key else OUT;mark=f/'failure.json' if f.exists() else f.with_name(f.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
