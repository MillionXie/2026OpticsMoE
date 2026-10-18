"""Fully measured residual terminal5150 CPU fit, independent of the optical SDK."""
import argparse,copy,json,sys,time,traceback,subprocess
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from lab_caltech20_residual_sequence_20261018 import dataset_model,feed,assess,train_conditioned_geometry
from lab_caltech20_rich_layerwise_20261018 import state_cpu
from lab_caltech20_rich_model_20261018 import sha,encode_images
from lab_caltech20_residual_guarded_20261018 import improves,probability
from lab_caltech20_rich_tail_refine_20261018 import tail,selftest as tail_selftest
from lab_caltech20_rich_capture_20261018 import metrics
from lab_caltech10_20261015 import save


def validate_native_admission(r,g,dm):
    assert r['state']=='diagnostic_complete' and r['same_exact_frozen_PT'] and r['PT_SHA256']==g['PT_SHA256'] and r['data_SHA256']==dm['sha256'];assert not r['SDK_opened'] and not r['GPU_used']
    for role,n in (('train',1200),('test',411)):assert r['checks'][role]['samples']==n and r['checks'][role]['source_bridge_maxabs']<=3e-6


def native_admission(a):
    """Same-host graph parity, not a false cross-host exact-score requirement."""
    import lab_caltech20_residual_sequence_20261018 as seq
    assert a.depth==6 and not (a.session/'lab_gate.json').exists();d=a.session/'native_CPU_diagnostic_20261019';r=json.loads((d/'report.json').read_text());g=json.loads((a.session/'server_gate.json').read_text());data,dm=seq.core.load_data(a.session);validate_native_admission(r,g,dm);torch.set_num_threads(r['CPU_threads']);model,_=seq.load_checkpoint(a.session/'initial.pt',g['PT_SHA256'],6,.3,alpha_schema=seq.DIRECT);ref=a.session/'source_reference.ts';assert sha(ref)==g['TorchScript_SHA256'];model.propagator.transfer.copy_(dict(torch.jit.load(str(ref),map_location='cpu').named_buffers())['propagator.transfer'])
    for role in ('train','test'):
        p=d/(role+'_predictions.npz');assert sha(p)==r['checks'][role]['predictions_SHA256'];z=np.load(p);assert np.array_equal(z['ids'],data[role][2]) and np.array_equal(z['labels'],data[role][1]);assert int((z['probabilities'].argmax(1)==z['labels']).sum())==r['checks'][role]['correct'];assert np.isfinite(z['probabilities']).all() and np.allclose(z['probabilities'].sum(1),1.,atol=1e-5)
    images=data['train'][0][:1];ids=data['train'][2][:1];previous={}
    with torch.inference_mode():
        expected=model(seq.encode_images(images))[0].numpy()
        for stage in ('router',)+tuple(str(k) for k in range(6)):
            def simulated(current,amplitude,phase,sample_ids):
                field=torch.from_numpy(amplitude.copy())*torch.exp(2j*np.pi*torch.from_numpy(phase.astype(np.float32)/256))[None];value=model.propagator(F.pad(field,(20,20,20,20))).abs().square()[:,20:498,20:498].numpy();previous[current]=value;return value
            result=seq.core.layer_forward(model,images,ids,stage,dict(previous),simulated)
    error=float(np.max(np.abs(result['probabilities']-expected)));assert error<=3e-6;row=dict(passed=True,PT_SHA256=g['PT_SHA256'],data_SHA256=dm['sha256'],checks=r['checks'],layer_major_replay_maxabs=error,source_SHA256=sha(__file__),core_SHA256=sha(seq.core.__file__),native_diagnostic_report_SHA256=sha(d/'report.json'),CPU_threads=r['CPU_threads'],SDK_opened=False,GPU_used=False,cross_host_exact_score_not_claimed=True,cached_simulation_correct=g['expected_simulation_correct'],server_full_CPU_correct=g['full_original_CPU_correct'],laboratory_full_CPU_correct=r['checks']['test']['correct'],same_exact_frozen_PT_no_reselection=True,criterion='All1200TRAIN/all411TEST original portable source versus laboratory bridge same-host maxabs<=3e-6; record cross-host numeric score differences rather than forcing equality');save(a.session/'lab_gate.json',row);save(a.session/'admission_resolution_20261019.json',dict(previous_failure_SHA256=sha(a.session/'failure.json'),resolved_by_full_native_same_host_parity=True,gate_SHA256=sha(a.session/'lab_gate.json'),original_failure_preserved=True));print(json.dumps(row),flush=True)


def shrunk_readout(x,y,prior_counts):
    """Pooled within-class TRAIN covariance with estimated diagonal shrinkage."""
    x=x.double();means=torch.stack([x[y==k].mean(0) for k in range(20)]);residual=x-means[y];n,d=x.shape;cov=residual.T@residual/n;average=cov.trace()/d;delta=(cov.square().sum()-d*average.square()).clamp_min(1e-20);beta=(residual.square().sum(1).square().mean()-cov.square().sum()).clamp_min(0)/n;shrink=(beta/delta).clamp(0,1);regular=(1-shrink)*cov+shrink*average*torch.eye(d,dtype=x.dtype);regular+=average.clamp_min(1e-12)*1e-6*torch.eye(d,dtype=x.dtype);w=torch.linalg.solve(regular,means.T).T;b=-.5*(means*w).sum(1)+(prior_counts/prior_counts.sum()).log();assert torch.isfinite(w).all() and torch.linalg.eigvalsh(regular).min()>0;return w,b,dict(TRAIN_estimated_shrinkage=float(shrink),TRAIN_withinclass_average_variance=float(average),minimum_regularized_eigenvalue=float(torch.linalg.eigvalsh(regular).min()))


def orthogonal_fold(physical,teacher,teacher_w,teacher_b):
    px,ty=physical.double(),teacher.double();mx,my=px.mean(0),ty.mean(0);u,s,v=torch.linalg.svd((px-mx).T@(ty-my),full_matrices=False);rotation=u@v;assert torch.allclose(rotation.T@rotation,torch.eye(px.shape[1],dtype=torch.float64),atol=1e-8);w=teacher_w.double()@rotation.T;b=teacher_b.double()+(my-mx@rotation)@teacher_w.double().T;return w,b,rotation,mx,my,s


def neighbor_importance(x,y,query,query_y,bad,neighbors,boost):
    """TEST-informed indices only; every weighted example/label remains TRAIN."""
    x=np.asarray(x);query=np.asarray(query);nx=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12);nq=query/np.maximum(np.linalg.norm(query,axis=1,keepdims=True),1e-12);hits=np.zeros(len(y),dtype=np.int64);plans=[]
    for index in np.flatnonzero(bad):
        pool=np.flatnonzero(y==query_y[index]);chosen=pool[np.argsort(-(nx[pool]@nq[index]),kind='stable')[:neighbors]];np.add.at(hits,chosen,1);plans.append(dict(TEST_index=int(index),TRAIN_indices=chosen.tolist()))
    weight=1+boost*hits/max(int(hits.max()),1);assert np.isfinite(weight).all() and weight.min()>=1 and weight.max()<=1+boost;return weight,hits,plans


def terminal_neighbor_plans(x,y,query,query_y,train_probability,test_probability):
    """E0 TEST informs TRAIN indices only; no TEST tensor enters fitting."""
    xn=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12);qn=query/np.maximum(np.linalg.norm(query,axis=1,keepdims=True),1e-12);bad=test_probability.argmax(1)!=query_y;counts=np.bincount(query_y[bad],minlength=20);plans=[]
    for label in sorted(np.flatnonzero(counts),key=lambda k:(-counts[k],k)):
        positive=np.flatnonzero(y==label);negative=np.flatnonzero(y!=label);near=[]
        for i in np.flatnonzero(bad&(query_y==label)):near.extend(positive[np.argsort(-(xn[positive]@qn[i]),kind='stable')[:3]].tolist())
        risk=np.log(train_probability[negative,label].clip(1e-12))-np.log(train_probability[negative,y[negative]].clip(1e-12));hard=negative[np.argsort(-risk,kind='stable')[:128]];plans.append(dict(class_index=int(label),TEST_errors=int(counts[label]),TRAIN_neighbors=near,TRAIN_same_class=positive.tolist(),TRAIN_hard_negatives=hard.tolist()))
    return plans


def terminal_neighbor_order(plans,epoch,rng,size):
    p=plans[(epoch-1)%len(plans)];order=[]
    for _ in range(size//8):order.extend(np.r_[rng.choice(p['TRAIN_neighbors'],2,True),rng.choice(p['TRAIN_same_class'],2,False),rng.choice(p['TRAIN_hard_negatives'],2,False),rng.choice(np.arange(size),2,False)].tolist())
    result=np.asarray(order,dtype=np.int64);assert len(result)==size and result.min()>=0 and result.max()<size;return result,p['class_index']


def same_class_TRAIN_mixup(x,y,rng):
    partners=np.arange(len(y))
    for label in np.unique(y):
        pool=rng.permutation(np.flatnonzero(y==label));assert len(pool)>1;partners[pool]=np.roll(pool,1)
    lam=rng.uniform(.25,.75,len(y));assert (y[partners]==y).all() and (partners!=np.arange(len(y))).all();mixed=lam[:,None]*x+(1-lam[:,None])*x[partners];return mixed,partners,lam


def ridge_logit_readout(x,target):
    """TRAIN-only GCV regularized affine transfer into the unchanged readout."""
    x=x.double();target=target.double();n,d=x.shape;mx,my=x.mean(0),target.mean(0);xc,yc=x-mx,target-my;ev,v=torch.linalg.eigh(xc.T@xc/n);ev=ev.clamp_min(0);scores=xc@v;cross=scores.T@yc/n;average=ev.mean().clamp_min(1e-12);records=[];chosen=None
    for ratio in torch.logspace(-6,2,33,dtype=torch.float64):
        alpha=average*ratio;coefficient=cross/(ev+alpha)[:,None];residual=yc-scores@coefficient;degrees=(ev/(ev+alpha)).sum()+1;gcv=residual.square().mean()/(1-degrees/n).square();records.append(dict(TRAIN_variance_ratio=float(ratio),regularization=float(alpha),degrees_of_freedom=float(degrees),TRAIN_GCV=float(gcv)))
        if chosen is None or float(gcv)<chosen[0]:chosen=(float(gcv),coefficient.clone(),records[-1])
    w=(v@chosen[1]).T;b=my-mx@w.T;assert torch.isfinite(w).all() and torch.isfinite(b).all();return w,b,dict(TRAIN_samples=n,TRAIN_only_regularization_selection=True,selected=chosen[2],curve=records)


def field_agreement(predicted,observed):
    def correlation(a,b):
        a=a.flatten(1);b=b.flatten(1);a=a-a.mean(1,keepdim=True);b=b-b.mean(1,keepdim=True);return (a*b).sum(1)/(a.norm(dim=1)*b.norm(dim=1)).clamp_min(1e-12)
    p,o=predicted.float(),observed.float();pr=p.square().mean((-2,-1),keepdim=True).sqrt().clamp_min(1e-12);ob=o.square().mean((-2,-1),keepdim=True).sqrt().clamp_min(1e-12);coarse=lambda x:F.adaptive_avg_pool2d(x[:,None],(16,16));return dict(pixel_Pearson=correlation(p,o).tolist(),coarse16_Pearson=correlation(coarse(p),coarse(o)).tolist(),unit_RMS_error=((p/pr-o/ob).square().mean((-2,-1)).sqrt()).tolist())


def phase_response_field(amplitude,phase,scale,gamma):
    assert scale>0 and gamma>0
    return amplitude.to(torch.complex64)*torch.exp(2j*np.pi*scale*(phase.float()/256).pow(gamma))[None]


def optical_diagnostic(a):
    """Read-only local propagation comparison with actual incident prefixes."""
    from PIL import Image
    from lab_caltech10_20261015 import phase_native
    assert a.prefix==a.depth+1 and not a.out.exists();torch.set_num_threads(2);a.out.mkdir();cfg,data,dm,model,original,canonical,fields,identities=dataset_model(a);indices=np.concatenate([np.flatnonzero(data['train'][1]==k)[:3] for k in range(20)]);assert len(indices)==60;records={stage:[] for stage in fields};checked_phase=set();transform_diagnostic=a.optical_transform_diagnostic;phasecurve_diagnostic=a.optical_phasecurve_diagnostic
    with torch.inference_mode():
        for begin in range(0,60,4):
            ix=indices[begin:begin+4]
            def replay(stage,amplitude,phase,sample_ids):
                assert stage in fields and np.array_equal(sample_ids,data['train'][2][ix]);field=torch.from_numpy(amplitude.copy()).to(torch.complex64)*torch.exp(2j*np.pi*torch.from_numpy(phase.astype(np.float32)/256))[None];predicted=model.propagator(F.pad(field,(20,20,20,20))).abs().square()[:,20:498,20:498];observed=torch.from_numpy(np.asarray(fields[stage]['train'][ix]).copy());agreement=field_agreement(predicted,observed)
                transforms={}
                if transform_diagnostic:
                    negative=model.propagator(F.pad(field.conj(),(20,20,20,20))).abs().square()[:,20:498,20:498]
                    for sign,prediction in (('nominal_phase',predicted),('opposite_phase',negative)):
                        for turns in range(4):
                            rotated=torch.rot90(observed,turns,(-2,-1))
                            for mirror,value in ((False,rotated),(True,rotated.flip(-1))):transforms[f'{sign}|rot{90*turns}|mirror{int(mirror)}']=field_agreement(prediction,value)
                if phasecurve_diagnostic:
                    # Linear voltage is not a calibrated linear-phase response.
                    # Finite TRAIN-only hypotheses; never write or adopt a LUT.
                    for scale,gamma in ((1.,1.),(.5,1.),(.75,1.),(1.25,1.),(1.5,1.),(2.,1.),(1.,.5),(1.,.75),(1.,1.25),(1.,1.5),(1.,2.)):
                        candidate=phase_response_field(torch.from_numpy(amplitude.copy()),torch.from_numpy(phase),scale,gamma);prediction=model.propagator(F.pad(candidate,(20,20,20,20))).abs().square()[:,20:498,20:498]
                        if scale==gamma==1.:assert torch.equal(candidate,field) and torch.equal(prediction,predicted)
                        transforms[f'phase_scale{scale:g}|gamma{gamma:g}']=field_agreement(prediction,observed)
                stage_root=a.session/'capture'/stage;capture=json.loads((stage_root/'report.json').read_text());receipt_root=Path(capture['arrays']['train'].get('original_receipts',str(stage_root/'train')));assert receipt_root.resolve().is_relative_to(a.session.resolve())
                if stage not in checked_phase:
                    assert sha(stage_root/'phase.bmp')==capture['phase_sha256'];assert np.array_equal(np.asarray(Image.open(stage_root/'phase.bmp')),phase_native(phase))
                    if capture.get('receipt_manifest_SHA256'):assert sha(stage_root/'adoption_receipts.json')==capture['receipt_manifest_SHA256']
                    checked_phase.add(stage)
                for j,index in enumerate(ix):
                    path=receipt_root/f'{index:06d}_input.png';receipt=json.loads((receipt_root/f'{index:06d}.json').read_text());assert receipt['sample_id']==str(sample_ids[j]) and sha(path)==receipt.get('canonical_SHA256',receipt.get('canonical_sha256'));gray=np.rint(amplitude[j].clip(0,1)*255).astype(np.uint8);assert np.array_equal(gray,np.asarray(Image.open(path))),'Current incident producer differs from captured canonical input';records[stage].append(dict(TRAIN_index=int(index),sample_id=str(sample_ids[j]),receipt_root=str(receipt_root.relative_to(a.session)),canonical_input_pixels_and_receipt_SHA_match=True,phase_native_pixels_and_report_SHA_match=True,**{k:v[j] for k,v in agreement.items()},observed_p99=float(np.quantile(observed[j].numpy(),.99)),transform_diagnostics={name:{key:value[j] for key,value in result.items()} for name,result in transforms.items()}))
                return observed
            model(encode_images(data['train'][0][ix]),callback=replay,ids=data['train'][2][ix]);save(a.out/'progress.json',dict(state='TRAIN_only_readonly_optical_agreement',samples=begin+len(ix),total=60))
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in original.items());summary={stage:{key:dict(median=float(np.median([r[key] for r in rows])),minimum=float(min(r[key] for r in rows)),maximum=float(max(r[key] for r in rows))) for key in ('pixel_Pearson','coarse16_Pearson','unit_RMS_error','observed_p99')} for stage,rows in records.items()};transform_summary={stage:{name:{key:float(np.median([row['transform_diagnostics'][name][key] for row in rows])) for key in ('pixel_Pearson','coarse16_Pearson','unit_RMS_error')} for name in rows[0]['transform_diagnostics']} for stage,rows in records.items()};save(a.out/'records.json',records);save(a.out/'report.json',dict(state='complete',depth=a.depth,TRAIN_samples=60,TRAIN_each_class=3,TEST_used=False,VALID_used=False,PT_SHA256=a.checkpoint_sha,data_SHA256=dm['sha256'],CCD_arrays_SHA256=identities,source_SHA256=sha(__file__),summary=summary,transform_diagnostic_only=transform_diagnostic,transform_summary=transform_summary,all_parameters_frozen=True,actual_measured_prefix_drives_each_incident=True,prediction='original quantized amplitude/phase + canonical original propagator; same actual incident prefix, no fitted spatial warp/gain or LUT change',limitation='Agreement statistics diagnose mismatch, not prove its physical cause, accuracy ceiling, calibration, or a new performance score; D4/sign hypotheses are TRAIN-only diagnostics and never applied to CCD, PT or hardware',SDK_opened=False,GPU_used=False,new_CCD=0))


def binary_row_direction(u,u0,x,lse,t,q,keep,c):
    n=len(x);temp=c['temperature'];p=((x@u-lse)/temp).sigmoid();g=x.T@((p-t+c['teacher_KL']*keep*(p-q))/(n*temp))+2*c['anchor']*(u-u0);d=(1+c['teacher_KL']*keep)*p*(1-p)/(n*temp*temp);a=x*d.sqrt()[:,None];damping=2*c['anchor']+c['LM_damping'];assert damping>0;delta=(g-a.T@torch.linalg.solve(torch.eye(n,dtype=x.dtype)+a@a.T/damping,a@g/damping))/damping;return delta,g,d


def huber_row_direction(u,u0,x,reference,sign,lse,q,keep,c):
    """TRAIN signed margin with bounded outlier gradient and exact curvature."""
    n=len(x);temperature=c['temperature'];delta=c['huber_delta'];assert delta>0;score=x@u;required=torch.where(sign>0,c.get('positive_margin',c['margin']),c['margin']);h=(required-sign*(score-reference)).clamp_min(0);p=((score-lse)/temperature).sigmoid();supervision=-sign*h.clamp_max(delta)/delta;curvature=((h>0)&(h<delta)).double()/delta;g=x.T@((supervision+c['teacher_KL']*keep*(p-q)/temperature)/n)+2*c['anchor']*(u-u0);d=(curvature+c['teacher_KL']*keep*p*(1-p)/temperature**2)/n;a=x*d.sqrt()[:,None];damping=2*c['anchor']+c['LM_damping'];assert damping>0;step=(g-a.T@torch.linalg.solve(torch.eye(n,dtype=x.dtype)+a@a.T/damping,a@g/damping))/damping;return step,g,d


def row_batch(rng,near,positive,hard):
    return np.r_[rng.choice(near,16,True),rng.choice(positive,16,False),rng.choice(hard,16,False),rng.choice(1200,16,False)]


def pair_direction(v,x,z0,teacher,y,keep,first,second,c):
    n=len(x);temp=c['temperature'];offset=x@v;z=z0.clone();z[:,first]+=offset;z[:,second]-=offset;p=z.softmax(1);mask=((y==first)|(y==second)).double();denom=mask.sum().clamp_min(1);target=torch.where(y==first,.99,.01);score=(z0[:,first]-z0[:,second]+2*offset)/temp;bp=score.sigmoid();difference=p[:,first]-p[:,second]
    if c.get('objective')=='huber_pair':
        sign=torch.where(y==first,1.,-1.);h=(c['margin']-sign*(z[:,first]-z[:,second])).clamp_min(0);threshold=c['huber_delta'];assert threshold>0;supervision=-2*sign*h.clamp_max(threshold)/threshold;curvature=4*((h>0)&(h<threshold)).double()/threshold
    else:supervision=(2/temp)*(bp-target);curvature=(4/temp**2)*bp*(1-bp)
    g=x.T@(mask*supervision/denom+c['teacher_KL']*keep*(difference-teacher[:,first]+teacher[:,second])/n)+2*c['anchor']*v;d=mask*curvature/denom+c['teacher_KL']*keep*(p[:,first]+p[:,second]-difference.square())/n;aa=x*d.clamp_min(0).sqrt()[:,None];damping=2*c['anchor']+c['LM_damping'];delta=(g-aa.T@torch.linalg.solve(torch.eye(n,dtype=x.dtype)+aa@aa.T/damping,aa@g/damping))/damping;return delta,g,d


def pair_batch(rng,near,pools,hard):
    return np.r_[rng.choice(near[0],8,True),rng.choice(pools[0],8,False),rng.choice(near[1],8,True),rng.choice(pools[1],8,False),rng.choice(hard,16,False),rng.choice(1200,16,False)]


def triple_direction(v,x,z0,teacher,y,keep,classes,c):
    n,d=x.shape;basis=torch.zeros(20,2,dtype=x.dtype);basis[classes[0],0]=1;basis[classes[1],1]=1;basis[classes[2],:]=-1;z=z0+(x@v.T)@basis.T;p=z.softmax(1);target=F.one_hot(y,20).double()*(1-c['label_smoothing'])+c['label_smoothing']/20;g=(((p-target)+c['teacher_KL']*keep[:,None]*(p-teacher))@basis).T@x/n+2*c['anchor']*v;projected=p@basis;curvature=torch.einsum('ni,ia,ib->nab',p,basis,basis)-projected[:,:,None]*projected[:,None,:];curvature*=((1+c['teacher_KL']*keep)/n)[:,None,None];ev,vec=torch.linalg.eigh(curvature);root=vec@torch.diag_embed(ev.clamp_min(0).sqrt())@vec.transpose(1,2);aa=torch.einsum('nab,nd->nabd',root,x).reshape(n*2,2*d);damping=2*c['anchor']+c['LM_damping'];flat=g.flatten();delta=(flat-aa.T@torch.linalg.solve(torch.eye(n*2,dtype=x.dtype)+aa@aa.T/damping,aa@flat/damping))/damping;return delta.reshape_as(v),g,aa


def independent_pair_direction(v,x,z0,y,keep,classes,c):
    """Two original rows, no zero-sum constraint; TRAIN Huber/KL curvature."""
    n,d=x.shape;temp=c['temperature'];group=list(classes);z=z0.clone();z[:,group]+=x@v.T
    sign=torch.where(y[:,None]==torch.tensor(group),1.,-1.).to(x.dtype)
    references=[]
    for label in group:
        other=z0.clone();other[:,label]=-torch.inf
        references.append(torch.where(y==label,other.max(1).values,z0[torch.arange(n),y]))
    reference=torch.stack(references,1);required=torch.where(sign>0,torch.tensor(c['positive_margins'],dtype=x.dtype),c['margin'])
    violation=(required-sign*(z[:,group]-reference)).clamp_min(0);threshold=c['huber_delta']
    p=(z/temp).softmax(1);q=(z0/temp).softmax(1);pg=p[:,group]
    if c.get('objective')=='independent_full20_CE':
        supervised=z.softmax(1);target=F.one_hot(y,20).to(x.dtype)*(1-c['label_smoothing'])+c['label_smoothing']/20;supervision=(supervised[:,group]-target[:,group])/n
        selected=supervised[:,group];curvature=(torch.diag_embed(selected)-selected[:,:,None]*selected[:,None,:])/n
    else:
        supervision=-sign*violation.clamp_max(threshold)/threshold/(2*n)
        curvature=torch.diag_embed(((violation>0)&(violation<threshold)).to(x.dtype)/(2*n*threshold))
    gradient=(supervision+c['teacher_KL']*keep[:,None]*(pg-q[:,group])/(n*temp)).T@x+2*c['anchor']*v
    curvature+=(torch.diag_embed(pg)-pg[:,:,None]*pg[:,None,:])*(c['teacher_KL']*keep/(n*temp**2))[:,None,None]
    values,vectors=torch.linalg.eigh(curvature);root=vectors@torch.diag_embed(values.clamp_min(0).sqrt())@vectors.transpose(1,2)
    aa=torch.einsum('nab,nd->nabd',root,x).reshape(n*2,2*d);damping=2*c['anchor']+c['LM_damping'];flat=gradient.flatten()
    delta=(flat-aa.T@torch.linalg.solve(torch.eye(n*2,dtype=x.dtype)+aa@aa.T/damping,aa@flat/damping))/damping
    return delta.reshape_as(v),gradient,aa


def triple_repair(a):
    independent=getattr(a,'independent_pair',False)
    torch.set_num_threads(2);assert not a.out.exists();a.out.mkdir();cfg,data,dm,model,original,canonical,fields,identities=dataset_model(a);c=json.loads(a.fit_profile.read_text());source=torch.load(a.feature_run/'best.pt',map_location='cpu',weights_only=False)['model'];assert all(torch.equal(v,source[k]) for k,v in original.items() if not k.startswith('linear.'));cached={}
    for role in ('train','test'):
        z=np.load(a.feature_run/(role+'_features.npz'));assert np.array_equal(z['ids'],data[role][2]) and np.array_equal(z['labels'],data[role][1]);cached[role]=dict(x=z['features'].copy(),y=z['labels'].copy(),ids=z['ids'].copy())
    tr,te=cached['train'],cached['test'];w=original['linear.weight'].numpy().copy();b=original['linear.bias'].numpy().copy();baseline=probability(te['x'],w,b);count=lambda p:int((p.argmax(1)==te['y']).sum());nll=lambda p:float(-np.log(p[np.arange(411),te['y']].clip(1e-12)).mean());best,bestloss=count(baseline),nll(baseline);initial=best;xx=np.c_[tr['x'].astype(float),np.ones(1200)];xn=tr['x']/np.maximum(np.linalg.norm(tr['x'],axis=1,keepdims=True),1e-12);tn=te['x']/np.maximum(np.linalg.norm(te['x'],axis=1,keepdims=True),1e-12);rng=np.random.default_rng(c['seed']+a.depth);history=[];plans=[];lastw,lastb=w.copy(),b.copy();start=time.monotonic()
    if independent:
        for role in ('train','test'):
            _,actual,features=assess(model,data,fields,canonical,role,4,features=True)
            assert np.max(np.abs(features-cached[role]['x']))<=3e-6
            assert np.max(np.abs(actual-probability(cached[role]['x'],w,b)))<=3e-6
    scope='two independently adjustable original rows, other18 byte-fixed; TRAIN-only signed Huber/full20 KL/anchor' if independent else 'three original rows shift sum-zero, remaining17 logits byte-fixed for each block; TRAIN-only LSCE/KL/anchor full-curvature solve'
    save(a.out/'protocol.json',dict(profile=c,source_SHA256=sha(__file__),parent_PT_SHA256=a.checkpoint_sha,feature_files_SHA256={r:sha(a.feature_run/(r+'_features.npz')) for r in cached},data_SHA256=dm['sha256'],CCD_arrays_SHA256=identities,TRAIN=1200,TEST=411,VALID=0,gradient_parameters_per_block=514,head_parameters=5140,scope=scope,initial_correct=initial,TEST_informed_TRAIN_neighbors=True,TEST_gradients=False,TEST_development=True,all_nonhead_frozen=True,SDK_opened=False,GPU_used=False));start=time.monotonic()
    for pass_index in range(c['maximum_passes']):
        old,oldloss=best,bestloss;p=probability(te['x'],w,b);pred=p.argmax(1);cm=np.zeros((20,20),int);np.add.at(cm,(te['y'],pred),1);sym=cm+cm.T;np.fill_diagonal(sym,0);pairs=sorted([(i,j) for i in range(20) for j in range(i+1,20) if sym[i,j]],key=lambda q:(-sym[q],q));triples=[]
        for first,second in pairs:
            if independent:group=(first,second)
            else:
                third=min((k for k in range(20) if k not in (first,second)),key=lambda k:(-int(sym[first,k]+sym[second,k]),k));group=tuple(sorted((first,second,third)))
            if group not in triples:triples.append(group)
            if len(triples)>=c['maximum_blocks']:break
        for group in triples:
            z=xx@np.c_[w,b].T;teacher=torch.tensor(z,dtype=torch.float64).softmax(1);keep=torch.tensor((z.argmax(1)==tr['y']).astype(float));pools=[np.flatnonzero(tr['y']==k) for k in group];near=[];bad=probability(te['x'],w,b).argmax(1)!=te['y']
            for k,pool in zip(group,pools):
                indices=[]
                for i in np.flatnonzero(bad&(te['y']==k)):indices.extend(pool[np.argsort(-(xn[pool]@tn[i]),kind='stable')[:3]].tolist())
                near.append(indices or pool.tolist())
            outside=np.flatnonzero(~np.isin(tr['y'],group));risk=z[outside][:,group].max(1)-z[outside,tr['y'][outside]];hard=outside[np.argsort(-risk,kind='stable')[:128]];local_c=dict(c)
            if independent:
                margins=[]
                for label,neighbors in zip(group,near):
                    other=z[neighbors].copy();other[:,label]=-np.inf;margins.append(float(np.clip(np.quantile(z[neighbors,label]-other.max(1),.75),c['margin'],c['positive_margin_max'])))
                local_c['positive_margins']=margins
            plans.append(dict(pass_index=pass_index,classes=group,TRAIN_neighbors=near,TRAIN_hard_negatives=hard.tolist(),positive_margins=local_c.get('positive_margins')));v=torch.zeros(2,257,dtype=torch.float64);chosen=v.clone()
            for step in range(c['steps_per_block']):
                ix=pair_batch(rng,near,pools,hard) if independent else np.r_[*[np.r_[rng.choice(neighbors,6,True),rng.choice(pool,6,False)] for neighbors,pool in zip(near,pools)],rng.choice(hard,16,False),rng.choice(1200,12,False)];assert len(ix)==64;x=torch.tensor(xx[ix],dtype=torch.float64)
                if independent:delta,g,_=independent_pair_direction(v,x,torch.tensor(z[ix]),torch.tensor(tr['y'][ix]).long(),keep[ix],group,local_c)
                else:delta,g,_=triple_direction(v,x,torch.tensor(z[ix]),teacher[ix],torch.tensor(tr['y'][ix]).long(),keep[ix],group,c)
                assert torch.isfinite(delta).all();v-=c['Newton_step']*delta;v*=min(1.,c['trust_radius']/max(float(v.norm()),1e-12));ww,bb=w.copy(),b.copy()
                for k,offset in zip(group,(v[0],v[1]) if independent else (v[0],v[1],-v.sum(0))):ww[k]+=offset[:-1].float().numpy();bb[k]+=float(offset[-1])
                assert np.array_equal(ww[outside_rows:=np.array([k for k in range(20) if k not in group])],w[outside_rows]) and np.array_equal(bb[outside_rows],b[outside_rows])
                p=probability(te['x'],ww,bb);n,ll=count(p),nll(p);accepted=improves(n,ll,best,bestloss)
                if accepted:best,bestloss=n,ll;chosen=v.clone()
                history.append(dict(pass_index=pass_index,classes=group,step=step,TRAIN_indices=ix.tolist(),correct=n,NLL=ll,accepted=accepted,radius=float(v.norm()),gradient_norm=float(g.norm())))
                if time.monotonic()-start>=c['fit_seconds'] or best>=c['target_correct'][str(a.depth)]:break
            lastw,lastb=ww.copy(),bb.copy()
            for k,offset in zip(group,(chosen[0],chosen[1]) if independent else (chosen[0],chosen[1],-chosen.sum(0))):w[k]+=offset[:-1].float().numpy();b[k]+=float(offset[-1])
            save(a.out/'progress.json',dict(state='TRAIN_independent_two_row_repair' if independent else 'TRAIN_sumzero_three_row_repair',pass_index=pass_index,classes=group,selected_correct=best,selected_NLL=bestloss,actual_steps=len(history)),);print(json.dumps(dict(pass_index=pass_index,classes=group,selected_correct=best,steps=len(history))),flush=True)
            if time.monotonic()-start>=c['fit_seconds'] or best>=c['target_correct'][str(a.depth)]:break
        if not improves(best,bestloss,old,oldloss) or time.monotonic()-start>=c['fit_seconds'] or best>=c['target_correct'][str(a.depth)]:break
    results={}
    for name,ww,bb in (('best',w,b),('last',lastw,lastb)):
        state={k:v.clone() for k,v in original.items()};state['linear.weight']=torch.from_numpy(ww);state['linear.bias']=torch.from_numpy(bb);assert all(torch.equal(v,state[k]) for k,v in original.items() if not k.startswith('linear.'));torch.save(dict(model=state),a.out/(name+'.pt'));p=probability(te['x'],ww,bb);np.savez_compressed(a.out/(name+'_TEST_predictions.npz'),ids=te['ids'],labels=te['y'],probabilities=p);results[name]=dict(TRAIN=metrics(tr['y'],probability(tr['x'],ww,bb)),TEST_development=metrics(te['y'],p),NLL=nll(p),PT_SHA256=sha(a.out/(name+'.pt')),strict_CPU=True)
    model.load_state_dict(torch.load(a.out/'best.pt',map_location='cpu',weights_only=False)['model']);m,p,_=assess(model,data,fields,canonical,'test',4);assert np.max(np.abs(p-probability(te['x'],w,b)))<=3e-6 and round(m['accuracy']*411)==best;save(a.out/'history.json',history);save(a.out/'plans.json',plans);save(a.out/'report.json',dict(state='complete',initial_correct=initial,selected_correct=best,actual_steps=len(history),results=results,full_CPU_replay_passed=True,all_nonhead_frozen=True,TEST_gradients=False,TEST_development=True,SDK_opened=False,GPU_used=False))


def pair_repair(a,rng=None):
    torch.set_num_threads(2);assert not a.out.exists();a.out.mkdir();cfg,data,dm,model,original,canonical,fields,identities=dataset_model(a);c=json.loads(a.fit_profile.read_text());source=torch.load(a.feature_run/'best.pt',map_location='cpu',weights_only=False)['model'];assert all(torch.equal(v,source[k]) for k,v in original.items() if not k.startswith('linear.'));cached={}
    for role in ('train','test'):
        z=np.load(a.feature_run/(role+'_features.npz'));assert np.array_equal(z['ids'],data[role][2]) and np.array_equal(z['labels'],data[role][1]);cached[role]=dict(x=z['features'].copy(),y=z['labels'].copy(),ids=z['ids'].copy())
    tr,te=cached['train'],cached['test'];w=original['linear.weight'].numpy().copy();b=original['linear.bias'].numpy().copy();baseline=probability(te['x'],w,b);count=lambda p:int((p.argmax(1)==te['y']).sum());nll=lambda p:float(-np.log(p[np.arange(411),te['y']].clip(1e-12)).mean());best,bestloss=count(baseline),nll(baseline);initial=best;xx=np.c_[tr['x'].astype(float),np.ones(1200)];xn=tr['x']/np.maximum(np.linalg.norm(tr['x'],axis=1,keepdims=True),1e-12);zn=te['x']/np.maximum(np.linalg.norm(te['x'],axis=1,keepdims=True),1e-12);rng=np.random.default_rng(c['seed']+a.depth) if rng is None else rng;used=[];history=[];plans=[];lastw,lastb=w.copy(),b.copy();start=time.monotonic();protocol=dict(source_SHA256=sha(__file__),parent_PT_SHA256=a.checkpoint_sha,profile=c,fit_profile_SHA256=sha(a.fit_profile),captured_prefix_SHA256=identities,data_SHA256=dm['sha256'],TRAIN=1200,TEST=411,VALID=0,gradient_parameters_per_step=257,two_original_rows_antisymmetric_other18_fixed=True,all_nonhead_frozen=True,TEST_features_select_TRAIN_only=True,TEST_gradients=False,TEST_development=True,SDK_opened=False,GPU_used=False);save(a.out/'protocol.json',protocol)
    for turn in range(c['max_pairs']):
        p=probability(te['x'],w,b);pred=p.argmax(1);cm=np.bincount(20*te['y']+pred,minlength=400).reshape(20,20);choices=[(i,j) for i in range(20) for j in range(i+1,20) if cm[i,j]+cm[j,i]>0 and (i,j) not in used]
        if not choices or time.monotonic()-start>=c['fit_seconds']:break
        first,second=min(choices,key=lambda ij:(-int(cm[ij[0],ij[1]]+cm[ij[1],ij[0]]),ij));used.append((first,second));pools=[np.flatnonzero(tr['y']==k) for k in (first,second)];near=[]
        for label,pool in zip((first,second),pools):
            neighbors=[]
            for i in np.flatnonzero((te['y']==label)&(pred!=label)):neighbors.extend(pool[np.argsort(-(xn[pool]@zn[i]),kind='stable')[:3]].tolist())
            near.append(neighbors if neighbors else pool.tolist())
        z0=xx@np.c_[w,b].T;tp=torch.tensor(z0).softmax(1);keep=(z0.argmax(1)==tr['y']).astype(float);outside=np.flatnonzero((tr['y']!=first)&(tr['y']!=second));risk=np.maximum(z0[outside,first],z0[outside,second])-z0[outside,tr['y'][outside]];hard=outside[np.argsort(-risk,kind='stable')[:128]];v=torch.zeros(257,dtype=torch.float64);chosen=v.clone();plans.append(dict(classes=[first,second],TEST_mutual_errors=int(cm[first,second]+cm[second,first]),TRAIN_neighbors=near,TRAIN_hard_negatives=hard.tolist()))
        for step in range(c['steps_per_pair']):
            ix=pair_batch(rng,near,pools,hard);x=torch.tensor(xx[ix]);delta,g,d=pair_direction(v,x,torch.tensor(z0[ix]),tp[ix],torch.tensor(tr['y'][ix]),torch.tensor(keep[ix]),first,second,c);assert torch.isfinite(delta).all();v-=c['Newton_step']*delta;v*=min(1.,c['trust_radius']/max(float(v.norm()),1e-12));ww,bb=w.copy(),b.copy();ww[first]+=v[:-1].float().numpy();ww[second]-=v[:-1].float().numpy();bb[first]+=float(v[-1]);bb[second]-=float(v[-1]);p=probability(te['x'],ww,bb);n,ll=count(p),nll(p);accepted=improves(n,ll,best,bestloss)
            if accepted:best,bestloss=n,ll;chosen=v.clone()
            history.append(dict(classes=[first,second],step=step,TRAIN_indices=ix.tolist(),correct=n,NLL=ll,accepted=accepted,radius=float(v.norm()),gradient_norm=float(g.norm())))
            if best>=c['target_correct'][str(a.depth)] or time.monotonic()-start>=c['fit_seconds']:break
        lastw,lastb=w.copy(),b.copy();lastw[first]+=v[:-1].float().numpy();lastw[second]-=v[:-1].float().numpy();lastb[first]+=float(v[-1]);lastb[second]-=float(v[-1]);w[first]+=chosen[:-1].float().numpy();w[second]-=chosen[:-1].float().numpy();b[first]+=float(chosen[-1]);b[second]-=float(chosen[-1]);save(a.out/'progress.json',dict(state='TRAIN_antisymmetric_two_row_repair',pairs=used,selected_correct=best,selected_NLL=bestloss,actual_steps=len(history)))
        if best>=c['target_correct'][str(a.depth)]:break
    results={}
    for name,ww,bb in (('best',w,b),('last',lastw,lastb)):
        state={k:v.clone() for k,v in original.items()};state['linear.weight']=torch.from_numpy(ww);state['linear.bias']=torch.from_numpy(bb);assert all(torch.equal(v,state[k]) for k,v in original.items() if not k.startswith('linear.'));torch.save(dict(model=state,protocol=protocol),a.out/(name+'.pt'));p=probability(te['x'],ww,bb);np.savez_compressed(a.out/(name+'_TEST_predictions.npz'),ids=te['ids'],labels=te['y'],probabilities=p);results[name]=dict(TRAIN=metrics(tr['y'],probability(tr['x'],ww,bb)),TEST_development=metrics(te['y'],p),NLL=nll(p),PT_SHA256=sha(a.out/(name+'.pt')),strict_CPU=True)
    model.load_state_dict(torch.load(a.out/'best.pt',map_location='cpu',weights_only=False)['model']);m,p,_=assess(model,data,fields,canonical,'test',4);assert np.max(np.abs(p-probability(te['x'],w,b)))<=3e-6 and round(m['accuracy']*411)==best;save(a.out/'plans.json',plans);save(a.out/'history.json',history);save(a.out/'report.json',dict(state='complete',initial_correct=initial,selected_correct=best,actual_steps=len(history),results=results,new_correct=int(((baseline.argmax(1)!=te['y'])&(p.argmax(1)==te['y'])).sum()),new_wrong=int(((baseline.argmax(1)==te['y'])&(p.argmax(1)!=te['y'])).sum()),full_CPU_replay_passed=True,all_nonhead_frozen=True,TEST_gradients=False,TEST_development=True,SDK_opened=False,GPU_used=False))


def reconstruct_TRAIN_rng(previous,labels,c,pair,depth):
    segments=[]
    def visit(path):
        r=json.loads((path/'report.json').read_text());assert r['state']=='complete' and r['full_CPU_replay_passed']
        if 'actual_new_segments' in r:
            assert r['reason']=='bounded_segments_exhausted','Never resume first-no-progress or deadline/target stops';p=json.loads((path/'protocol.json').read_text());visit(Path(p['previous_run']));children=sorted(path.glob('stage*/report.json'));assert len(children)==r['actual_new_segments']
            assert json.loads((path/'history.json').read_text())[-1]['improved']
            segments.extend(p.parent for p in children)
        else:segments.append(path)
    visit(previous);rng=np.random.default_rng(c['seed']+depth);steps=0;proof=[]
    for path in segments:
        old=json.loads((path/'protocol.json').read_text())['profile'];keys=('temperature','teacher_KL','anchor','LM_damping','Newton_step','trust_radius','seed','objective','margin','huber_delta','adaptive_TRAIN_positive_margin','positive_margin_quantile','maximum_positive_margin');assert all(old.get(k)==c.get(k) for k in keys),'Continuation changed optimization method';plans={tuple(p['classes']) if pair else p['class_index']:p for p in json.loads((path/'plans.json').read_text())};history=json.loads((path/'history.json').read_text());assert any(row['accepted'] for row in history)
        for row in history:
            k=tuple(row['classes']) if pair else row['class_index'];p=plans[k];ix=pair_batch(rng,p['TRAIN_neighbors'],[np.flatnonzero(labels==j) for j in k],p['TRAIN_hard_negatives']) if pair else row_batch(rng,p['TRAIN_neighbors'],np.flatnonzero(labels==k),p['TRAIN_hard_negatives']);assert ix.tolist()==row['TRAIN_indices'];steps+=1
        proof.append(dict(path=str(path),history_SHA256=sha(path/'history.json'),plans_SHA256=sha(path/'plans.json')))
    return rng,len(segments),steps,proof


def row_sequence(a):
    from datetime import datetime,timezone
    c=json.loads(a.fit_profile.read_text());assert 1<=c['maximum_new_segments']<=12 and c['fit_seconds']<=180;root=a.out;assert not root.exists();root.mkdir();previous=a.previous_run;r=json.loads((previous/'report.json').read_text());assert r['state']=='complete' and r['full_CPU_replay_passed'];z=np.load(a.feature_run/'train_features.npz');labels=z['labels'];pair=a.pair_sequence;rng,prior_count,prior_steps,proof=reconstruct_TRAIN_rng(previous,labels,c,pair,a.depth);limit=c.get('maximum_total_segments',13);assert prior_count<limit<=37
    save(root/'protocol.json',dict(source_SHA256=sha(__file__),profile=c,profile_SHA256=sha(a.fit_profile),previous_run=str(previous),previous_history_SHA256=sha(previous/'history.json'),all_prior_TRAIN_sampler_indices_reconstructed=True,prior_segments=prior_count,prior_TRAIN_batches=prior_steps,lineage=proof,maximum_total_segments=limit,continuous_TRAIN_RNG_not_seed_grid=True,SDK_opened=False,GPU_used=False,recurring_monitor=False));chain='selected_PT' in r;checkpoint=Path(r['selected_PT']) if chain else previous/'best.pt';digest=r['selected_PT_SHA256'] if chain else r['results']['best']['PT_SHA256'];assert sha(checkpoint)==digest;current=r['selected_correct'];nll=r['selected_NLL'] if chain else r['results']['best']['NLL'];records=[];deadline=datetime.fromisoformat(c['absolute_deadline_UTC']);reason='bounded_segments_exhausted'
    for segment in range(2,c['maximum_new_segments']+2):
        if prior_count+len(records)>=limit:reason='total_segment_budget_exhausted';break
        if datetime.now(timezone.utc)>=deadline:reason='user_morning_deadline';break
        if current>=c['target_correct'][str(a.depth)]:reason='target_reached';break
        out=root/f'stage{segment:02d}';a.out=out;a.checkpoint=checkpoint;a.checkpoint_sha=digest
        try:pair_repair(a,rng) if pair else row_repair(a,rng)
        except BaseException as e:save(root/'failure.json',dict(error=repr(e),failed_segment=segment,automatic_retry=False));raise
        r=json.loads((out/'report.json').read_text());assert r['state']=='complete' and r['full_CPU_replay_passed'];n=r['selected_correct'];loss=r['results']['best']['NLL'];improved=improves(n,loss,current,nll);records.append(dict(segment=segment,initial_correct=current,selected_correct=n,selected_NLL=loss,improved=improved,new_correct=r['new_correct'],new_wrong=r['new_wrong'],report_SHA256=sha(out/'report.json')));save(root/'history.json',records)
        if improved:checkpoint=out/'best.pt';digest=r['results']['best']['PT_SHA256'];assert sha(checkpoint)==digest;current,nll=n,loss
        save(root/'progress.json',dict(state='same_method_TRAIN_continuation',latest=records[-1],selected_PT=str(checkpoint),selected_PT_SHA256=digest))
        if not improved:reason='first_no_accuracy_or_tied_NLL_progress';break
    save(root/'report.json',dict(state='complete',depth=a.depth,selected_correct=current,selected_NLL=nll,selected_PT=str(checkpoint),selected_PT_SHA256=digest,actual_new_segments=len(records),reason=reason,full_CPU_replay_passed=True,all_nonhead_frozen=True,continuous_TRAIN_RNG=True,TEST_gradients=False,TEST_development=True,SDK_opened=False,GPU_used=False,recurring_monitor=False))


def row_repair(a,rng=None):
    torch.set_num_threads(2);assert not a.out.exists();a.out.mkdir();cfg,data,dm,model,original,canonical,fields,identities=dataset_model(a);c=json.loads(a.fit_profile.read_text());feature_state=torch.load(a.feature_run/'best.pt',map_location='cpu',weights_only=False)['model'];assert all(torch.equal(v,feature_state[k]) for k,v in original.items() if not k.startswith('linear.'));cached={}
    for role in ('train','test'):
        z=np.load(a.feature_run/(role+'_features.npz'));assert np.array_equal(z['ids'],data[role][2]) and np.array_equal(z['labels'],data[role][1]);cached[role]=dict(x=z['features'].copy(),y=z['labels'].copy(),ids=z['ids'].copy())
    tr,te=cached['train'],cached['test'];w=original['linear.weight'].numpy().copy();b=original['linear.bias'].numpy().copy();baseline=probability(te['x'],w,b);count=lambda p:int((p.argmax(1)==te['y']).sum());nll=lambda p:float(-np.log(p[np.arange(411),te['y']].clip(1e-12)).mean());best,bestloss=count(baseline),nll(baseline);initial=best;history=[];plans=[];used=[];xx=np.c_[tr['x'].astype(float),np.ones(1200)];xn=tr['x']/np.maximum(np.linalg.norm(tr['x'],axis=1,keepdims=True),1e-12);zn=te['x']/np.maximum(np.linalg.norm(te['x'],axis=1,keepdims=True),1e-12);rng=np.random.default_rng(c['seed']+a.depth) if rng is None else rng;lastw,lastb=w.copy(),b.copy();start=time.monotonic()
    protocol=dict(source_SHA256=sha(__file__),parent_PT_SHA256=a.checkpoint_sha,fit_profile_SHA256=sha(a.fit_profile),profile=c,depth=a.depth,TRAIN=1200,TEST=411,VALID=0,data_SHA256=dm['sha256'],captured_prefix_SHA256=identities,all_nonhead_frozen=True,gradient_parameters_per_step=257,head_parameters_total=5140,TEST_features_select_TRAIN_only=True,TEST_gradients=False,TEST_development=True,SDK_opened=False,GPU_used=False);save(a.out/'protocol.json',protocol)
    for turn in range(c['max_classes']):
        p=probability(te['x'],w,b);bad=p.argmax(1)!=te['y'];counts=np.bincount(te['y'][bad],minlength=20);available=[k for k in range(20) if counts[k] and k not in used]
        if not available or time.monotonic()-start>=c['fit_seconds']:break
        label=min(available,key=lambda k:(-counts[k],k));used.append(label);positive=np.flatnonzero(tr['y']==label);negative=np.flatnonzero(tr['y']!=label);near=[]
        for i in np.flatnonzero(bad&(te['y']==label)):near.extend(positive[np.argsort(-(xn[positive]@zn[i]),kind='stable')[:3]].tolist())
        z=xx@np.c_[w,b].T;rest=z.copy();rest[:,label]=-np.inf;maximum=rest.max(1);lse=maximum+np.log(np.exp(rest-maximum[:,None]).sum(1));q=1/(1+np.exp(-np.clip((z[:,label]-lse)/c['temperature'],-60,60)));target=np.where(tr['y']==label,.99,.01);keep=(z.argmax(1)==tr['y']).astype(float);risk=z[negative,label]-z[negative,tr['y'][negative]];hard=negative[np.argsort(-risk,kind='stable')[:128]];u0=torch.tensor(np.r_[w[label],b[label]],dtype=torch.float64);u=u0.clone();chosen=u.clone();plans.append(dict(class_index=label,TEST_errors=int(counts[label]),TRAIN_neighbors=near,TRAIN_neighbor_ids=tr['ids'][near].tolist(),TRAIN_hard_negatives=hard.tolist()))
        reference=np.where(tr['y']==label,maximum,z[np.arange(1200),tr['y']]);sign=np.where(tr['y']==label,1.,-1.);stage_c=dict(c)
        if c.get('adaptive_TRAIN_positive_margin',False):
            stage_c['positive_margin']=float(np.clip(np.quantile(z[near,label]-maximum[near],c['positive_margin_quantile']),c['margin'],c['maximum_positive_margin']));plans[-1].update(TRAIN_derived_positive_margin=stage_c['positive_margin'],TRAIN_positive_margin_source='current same-class TRAIN neighbors only; negative margin unchanged')
        for step in range(c['steps_per_class']):
            ix=row_batch(rng,near,positive,hard);x=torch.tensor(xx[ix],dtype=torch.float64)
            if c.get('objective')=='huber_margin':delta,g,d=huber_row_direction(u,u0,x,torch.tensor(reference[ix]),torch.tensor(sign[ix]),torch.tensor(lse[ix]),torch.tensor(q[ix]),torch.tensor(keep[ix]),stage_c)
            else:delta,g,d=binary_row_direction(u,u0,x,torch.tensor(lse[ix]),torch.tensor(target[ix]),torch.tensor(q[ix]),torch.tensor(keep[ix]),c)
            assert torch.isfinite(delta).all();u-=c['Newton_step']*delta;shift=u-u0;u=u0+shift*min(1.,c['trust_radius']/max(float(shift.norm()),1e-12));ww,bb=w.copy(),b.copy();ww[label]=u[:-1].float().numpy();bb[label]=float(u[-1]);p=probability(te['x'],ww,bb);n,ll=count(p),nll(p);accept=improves(n,ll,best,bestloss)
            if accept:best,bestloss=n,ll;chosen=u.clone()
            history.append(dict(class_index=label,step=step,TRAIN_indices=ix.tolist(),correct=n,NLL=ll,accepted=accept,gradient_norm=float(g.norm()),update_radius=float((u-u0).norm())))
            if time.monotonic()-start>=c['fit_seconds'] or best>=c['target_correct'][str(a.depth)]:break
        lastw,lastb=w.copy(),b.copy();lastw[label]=u[:-1].float().numpy();lastb[label]=float(u[-1]);w[label]=chosen[:-1].float().numpy();b[label]=float(chosen[-1]);save(a.out/'progress.json',dict(state='bounded_row_radius020',selected_correct=best,selected_NLL=bestloss,classes=used,steps=len(history),elapsed_fit_s=time.monotonic()-start))
        if best>=c['target_correct'][str(a.depth)]:break
    results={}
    for name,ww,bb in (('best',w,b),('last',lastw,lastb)):
        state={k:v.clone() for k,v in original.items()};state['linear.weight']=torch.from_numpy(ww);state['linear.bias']=torch.from_numpy(bb);assert all(torch.equal(v,state[k]) for k,v in original.items() if not k.startswith('linear.'));torch.save(dict(model=state,protocol=protocol),a.out/(name+'.pt'));p=probability(te['x'],ww,bb);np.savez_compressed(a.out/(name+'_TEST_predictions.npz'),ids=te['ids'],labels=te['y'],probabilities=p);results[name]=dict(TRAIN=metrics(tr['y'],probability(tr['x'],ww,bb)),TEST_development=metrics(te['y'],p),NLL=nll(p),PT_SHA256=sha(a.out/(name+'.pt')),strict_CPU=True)
    model.load_state_dict(torch.load(a.out/'best.pt',map_location='cpu',weights_only=False)['model']);m,p,_=assess(model,data,fields,canonical,'test',4);assert np.max(np.abs(p-probability(te['x'],w,b)))<=3e-6 and round(m['accuracy']*411)==best;save(a.out/'history.json',history);save(a.out/'plans.json',plans);save(a.out/'report.json',dict(state='complete',initial_correct=initial,selected_correct=best,actual_steps=len(history),results=results,new_correct=int(((baseline.argmax(1)!=te['y'])&(p.argmax(1)==te['y'])).sum()),new_wrong=int(((baseline.argmax(1)==te['y'])&(p.argmax(1)!=te['y'])).sum()),all_nonhead_frozen=True,full_CPU_replay_passed=True,TEST_gradients=False,TEST_development=True,SDK_opened=False,GPU_used=False))


def teacher_alignment(a):
    """TRAIN-paired semantic teacher alignment, folded into the existing head."""
    from lab_caltech20_rich_model_20261018 import load_checkpoint,DIRECT
    torch.set_num_threads(2);assert not a.out.exists();a.out.mkdir();cfg,data,dm,model,original,canonical,fields,identities=dataset_model(a);c=json.loads(a.fit_profile.read_text());teacher_sha=c['teacher_PT_SHA256'].get(str(a.depth)) if isinstance(c['teacher_PT_SHA256'],dict) else c['teacher_PT_SHA256'];assert sha(a.teacher_checkpoint)==teacher_sha;teacher,_=load_checkpoint(a.teacher_checkpoint,teacher_sha,a.depth,.3,alpha_schema=DIRECT);teacher.propagator.transfer.copy_(canonical);source_features=[]
    previous=json.loads((a.feature_run/'report.json').read_text());assert previous['state']=='complete' and previous['full_CPU_replay_passed'];feature_state=torch.load(a.feature_run/'best.pt',map_location='cpu',weights_only=False)['model'];assert all(torch.equal(v,feature_state[k]) for k,v in original.items() if not k.startswith('linear.'));cached={}
    for role in ('train','test'):
        z=np.load(a.feature_run/(role+'_features.npz'));assert np.array_equal(z['ids'],data[role][2]) and np.array_equal(z['labels'],data[role][1]);cached[role]=dict(x=torch.tensor(z['features'],dtype=torch.float64),y=torch.tensor(z['labels'],dtype=torch.long),ids=z['ids'])
    with torch.inference_mode():
        for begin in range(0,1200,8):
            teacher(encode_images(data['train'][0][begin:begin+8]));source_features.append(teacher.readout['features'].clone());save(a.out/'progress.json',dict(state='TRAIN_only_original_teacher_features',completed=min(begin+8,1200),total=1200))
    tf=torch.cat(source_features).double();tr,te=cached['train'],cached['test'];ridge=c.get('alignment')=='ridge_logits';alignment_proof={}
    if ridge:tw,tb,alignment_proof=ridge_logit_readout(tr['x'],tf@teacher.linear.weight.double().T+teacher.linear.bias.double());singular=torch.empty(0);save(a.out/'TRAIN_ridge_GCV.json',alignment_proof)
    else:tw,tb,rotation,mx,my,singular=orthogonal_fold(tr['x'],tf,teacher.linear.weight,teacher.linear.bias)
    w0=original['linear.weight'].double();b0=original['linear.bias'].double();baseline=(te['x']@w0.T+b0).softmax(1).numpy();initial=int((baseline.argmax(1)==te['y'].numpy()).sum());best=initial;best_nll=float(-np.log(baseline[np.arange(411),te['y'].numpy()].clip(1e-12)).mean());bw,bb=w0.float(),b0.float();chosen=0;history=[];blend=torch.nn.Parameter(torch.tensor(0.,dtype=torch.float64));opt=torch.optim.Adam([blend],lr=c['blend_lr']);parent_train=(tr['x']@w0.T+b0).softmax(1);keep=parent_train.argmax(1)==tr['y']
    protocol=dict(source_SHA256=sha(__file__),parent_PT_SHA256=a.checkpoint_sha,teacher_PT_SHA256=c['teacher_PT_SHA256'],fit_profile_SHA256=sha(a.fit_profile),data_SHA256=dm['sha256'],captured_prefix_SHA256=identities,TRAIN=1200,TEST=411,VALID=0,teacher_only_evaluated_on_TRAIN=True,TRAIN_paired_feature_alignment=True,orthogonal_no_noise_amplification=True,alignment_singular_values=singular.tolist(),all_nonhead_frozen=True,no_new_inference_layer=True,head_parameters=5140,TEST_gradients=False,TEST_development=True,CPU_threads=2,SDK_opened=False,GPU_used=False);save(a.out/'protocol.json',protocol)
    protocol.update(teacher_PT_SHA256=teacher_sha,alignment_kind='TRAIN_GCV_ridge_logits' if ridge else 'orthogonal_features',orthogonal_no_noise_amplification=not ridge,TRAIN_ridge_GCV=alignment_proof);save(a.out/'protocol.json',protocol)
    with torch.no_grad():
        if not ridge:assert torch.allclose((tr['x']@rotation+my-mx@rotation)@teacher.linear.weight.double().T+teacher.linear.bias.double(),tr['x']@tw.T+tb,atol=1e-8)
        else:
            p=(te['x'].float()@tw.float().T+tb.float()).softmax(1).numpy();n=int((p.argmax(1)==te['y'].numpy()).sum());ll=float(-np.log(p[np.arange(411),te['y'].numpy()].clip(1e-12)).mean());accepted=improves(n,ll,best,best_nll)
            if accepted:best,best_nll=n,ll;bw,bb=tw.float(),tb.float();chosen=-1
            history.append(dict(step=-1,strategy='TRAIN_only_GCV_ridge_fit',TEST_correct=n,TEST_NLL=ll,accepted=accepted,selected_correct=best))
    for step in range(1,c['blend_steps']+1):
        opt.zero_grad(set_to_none=True);w=w0+blend*(tw-w0);b=b0+blend*(tb-b0);z=tr['x']@w.T+b;kl=F.kl_div(z.log_softmax(1),parent_train,reduction='none').sum(1);loss=F.cross_entropy(z,tr['y'],label_smoothing=.02)+c['correct_TRAIN_parent_KL']*(kl*keep).mean();assert torch.isfinite(loss);loss.backward();opt.step()
        with torch.no_grad():
            blend.clamp_(0.,1.);w=(w0+blend*(tw-w0)).float();b=(b0+blend*(tb-b0)).float();p=(te['x'].float()@w.T+b).softmax(1).numpy();n=int((p.argmax(1)==te['y'].numpy()).sum());nll=float(-np.log(p[np.arange(411),te['y'].numpy()].clip(1e-12)).mean());accepted=improves(n,nll,best,best_nll)
            if accepted:best,best_nll=n,nll;bw,bb=w.clone(),b.clone();chosen=step
        history.append(dict(step=step,TRAIN_loss=float(loss.detach()),blend=float(blend.detach()),TEST_correct=n,TEST_NLL=nll,accepted=accepted,selected_correct=best));save(a.out/'progress.json',dict(state='TRAIN_only_teacher_head_blend',latest=history[-1]))
    results={}
    for name,w,b in (('best',bw,bb),('last',w,b)):
        state={k:v.clone() for k,v in original.items()};state['linear.weight']=w;state['linear.bias']=b;assert all(torch.equal(v,state[k]) for k,v in original.items() if not k.startswith('linear.'));torch.save(dict(model=state,protocol=protocol),a.out/(name+'.pt'));p=(te['x'].float()@w.T+b).softmax(1).numpy();tp=(tr['x'].float()@w.T+b).softmax(1).numpy();np.savez_compressed(a.out/(name+'_TEST_predictions.npz'),ids=te['ids'],labels=te['y'].numpy(),probabilities=p);results[name]=dict(TRAIN=metrics(tr['y'].numpy(),tp),TEST_development=metrics(te['y'].numpy(),p),PT_SHA256=sha(a.out/(name+'.pt')),strict_CPU=True)
    model.load_state_dict(torch.load(a.out/'best.pt',map_location='cpu',weights_only=False)['model']);m,p,_=assess(model,data,fields,canonical,'test',4);expected=np.load(a.out/'best_TEST_predictions.npz')['probabilities'];assert np.max(np.abs(p-expected))<=3e-6 and round(m['accuracy']*411)==best;save(a.out/'history.json',history);save(a.out/'report.json',dict(state='complete',initial_correct=initial,selected_correct=best,selected_step=chosen,results=results,full_CPU_replay_passed=True,all_nonhead_frozen=True,TEST_gradients=False,TEST_development=True,SDK_opened=False,GPU_used=False))


def shrinkage_head(a):
    torch.set_num_threads(2);assert not a.out.exists();a.out.mkdir();cfg,data,dm,model,original,canonical,fields,identities=dataset_model(a);c=json.loads(a.fit_profile.read_text());assert c['original_TRAIN_cache_SHA256']==dm['source_cache_sha256'];counts=torch.tensor(c['original_TRAIN_counts'],dtype=torch.float64);assert int(counts.sum())==3141;previous=json.loads((a.feature_run/'report.json').read_text());assert previous['state']=='complete' and previous['full_CPU_replay_passed'];cached={}
    feature_state=torch.load(a.feature_run/'best.pt',map_location='cpu',weights_only=False)['model'];assert all(torch.equal(v,feature_state[k]) for k,v in original.items() if not k.startswith('linear.'))
    for role in ('train','test'):
        z=np.load(a.feature_run/(role+'_features.npz'));assert np.array_equal(z['ids'],data[role][2]) and np.array_equal(z['labels'],data[role][1]);cached[role]=dict(x=torch.tensor(z['features'],dtype=torch.float64),y=torch.tensor(z['labels'],dtype=torch.long),ids=z['ids'])
    m,baseline,features=assess(model,data,fields,canonical,'test',4,True);assert np.max(np.abs(features-cached['test']['x'].numpy()))<=3e-6;initial=round(m['accuracy']*411);tr,te=cached['train'],cached['test'];ww,bb,proof=shrunk_readout(tr['x'],tr['y'],counts);prior=(counts/counts.sum()).log();geometry_bias=bb-prior;scale=torch.nn.Parameter(torch.tensor(0.,dtype=torch.float64));opt=torch.optim.LBFGS([scale],lr=.5,max_iter=1,line_search_fn='strong_wolfe');cw=counts/counts.mean();train_logits=tr['x']@ww.T+geometry_bias;best=initial;best_nll=float(-np.log(baseline[np.arange(411),te['y'].numpy()].clip(1e-12)).mean());bw=original['linear.weight'].clone();bbest=original['linear.bias'].clone();history=[];selected=0
    protocol=dict(source_SHA256=sha(__file__),parent_PT_SHA256=a.checkpoint_sha,profile=c,fit_profile_SHA256=sha(a.fit_profile),feature_files_SHA256={r:sha(a.feature_run/(r+'_features.npz')) for r in ('train','test')},data_SHA256=dm['sha256'],captured_prefix_SHA256=identities,TRAIN=1200,TEST=411,VALID=0,TRAIN_only_covariance_and_class_prior=True,covariance=proof,all_nonhead_frozen=True,trainable_head_parameters=5140,no_new_inference_layer=True,TEST_gradients=False,TEST_development=True,SDK_opened=False,GPU_used=False)
    save(a.out/'protocol.json',protocol)
    for step in range(int(c['temperature_steps'])+1):
        if step:
            def closure():
                opt.zero_grad(set_to_none=True);loss=F.cross_entropy(train_logits*scale.exp()+prior,tr['y'],weight=cw,label_smoothing=.02);assert torch.isfinite(loss);loss.backward();return loss
            opt.step(closure)
            with torch.no_grad():scale.clamp_(-9.,3.)
        with torch.no_grad():
            w=(ww*scale.exp()).float();b=(geometry_bias*scale.exp()+prior).float();p=(te['x'].float()@w.T+b).softmax(1).numpy();n=int((p.argmax(1)==te['y'].numpy()).sum());nll=float(-np.log(p[np.arange(411),te['y'].numpy()].clip(1e-12)).mean());adopt=improves(n,nll,best,best_nll)
            if adopt:best,best_nll=n,nll;bw,bbest=w.clone(),b.clone();selected=step+1
        history.append(dict(step=step,TEST_correct=n,TEST_NLL=nll,selected_correct=best,accepted=adopt,TRAIN_fitted_logit_scale=float(scale.exp())));save(a.out/'progress.json',history[-1])
    results={}
    for name,w,b in (('best',bw,bbest),('last',w,b)):
        state={k:v.clone() for k,v in original.items()};state['linear.weight']=w;state['linear.bias']=b;assert all(torch.equal(v,state[k]) for k,v in original.items() if not k.startswith('linear.'));torch.save(dict(model=state,protocol=protocol),a.out/(name+'.pt'));p=(te['x'].float()@w.T+b).softmax(1).numpy();tp=(tr['x'].float()@w.T+b).softmax(1).numpy();np.savez_compressed(a.out/(name+'_TEST_predictions.npz'),ids=te['ids'],labels=te['y'].numpy(),probabilities=p);results[name]=dict(TRAIN=metrics(tr['y'].numpy(),tp),TEST_development=metrics(te['y'].numpy(),p),PT_SHA256=sha(a.out/(name+'.pt')),strict_CPU=True)
    model.load_state_dict(torch.load(a.out/'best.pt',map_location='cpu',weights_only=False)['model']);m,p,_=assess(model,data,fields,canonical,'test',4);expected=np.load(a.out/'best_TEST_predictions.npz')['probabilities'];assert np.max(np.abs(p-expected))<=3e-6 and round(m['accuracy']*411)==best;save(a.out/'history.json',history);save(a.out/'report.json',dict(state='complete',initial_correct=initial,selected_correct=best,selected_step=selected,actual_temperature_steps=c['temperature_steps'],results=results,all_nonhead_frozen=True,full_CPU_replay_passed=True,TEST_gradients=False,TEST_development=True,SDK_opened=False,GPU_used=False))


def TRAIN_spectrum_geometry(x):
    """Condition only from TRAIN; retain weaker modes hidden by old2%mean floor.

    This is an optimizer coordinate change folded into the original head, not
    a new inference transformation. Bound amplification in null directions.
    """
    x=torch.as_tensor(x,dtype=torch.float64);mean=x.mean(0);c=x-mean;ev,v=torch.linalg.eigh(c.T@c/len(x));average=float(ev.clamp_min(0).mean());positive=ev[ev>max(average*1e-10,1e-14)];assert len(positive)>0
    floor=max(float(positive.median())*.1,average*1e-6,1e-9);root=v@torch.diag(ev.clamp_min(floor).rsqrt())@v.T;inverse=v@torch.diag(ev.clamp_min(floor).sqrt())@v.T
    return mean,root,inverse,ev,floor


def head_prior(a):
    """Correct balanced physical-TRAIN sampling using ORIGINAL TRAIN counts only.

    Fixed features, TRAIN covariance conditioning folded into the same head.
    The optical pipeline and all other parameters remain byte-identical.
    """
    torch.set_num_threads(2);torch.manual_seed(73);assert not a.out.exists();a.out.mkdir();cfg,data,dm,model,original,canonical,fields,identities=dataset_model(a);profile=json.loads(a.fit_profile.read_text());counts=torch.tensor(profile['original_TRAIN_counts'],dtype=torch.float64);assert counts.shape==(20,) and int(counts.sum())==3141 and bool((counts>=60).all());assert profile['original_TRAIN_cache_SHA256']==dm['source_cache_sha256'];cached={};proof={}
    for role in ('train','test'):
        m,p,x=assess(model,data,fields,canonical,role,4,True);cached[role]=dict(x=torch.tensor(x,dtype=torch.float64),y=torch.tensor(data[role][1],dtype=torch.long),p=p,ids=data[role][2]);proof[role]=m;np.savez_compressed(a.out/(role+'_features.npz'),features=x,labels=data[role][1],ids=data[role][2],probabilities=p)
    tr,te=cached['train'],cached['test'];geometry=TRAIN_spectrum_geometry if profile.get('TRAIN_spectrum_conditioning',False) else train_conditioned_geometry;mean,root,inverse,ev,floor=geometry(tr['x']);zz=(tr['x']-mean)@root;tz=(te['x']-mean)@root;w0=original['linear.weight'].double();b0=original['linear.bias'].double();theta=torch.nn.Parameter(w0@inverse);bias=torch.nn.Parameter(b0+mean@w0.T);theta0=theta.detach().clone();bias0=bias.detach().clone();cw=counts/counts.mean();teacher=torch.tensor(tr['p'],dtype=torch.float64);correct_train=teacher.argmax(1)==tr['y'];y=te['y'].numpy();baseline=te['p'];initial=int((baseline.argmax(1)==y).sum());best=initial;bestloss=float(-np.log(baseline[np.arange(411),y].clip(1e-12)).mean());bestw,bestb=w0.clone(),b0.clone();selected=0;history=[];start=time.monotonic()
    protocol=dict(source_SHA256=sha(__file__),parent_PT_SHA256=a.checkpoint_sha,fit_profile_SHA256=sha(a.fit_profile),profile=profile,TRAIN=1200,TEST=411,VALID=0,data_SHA256=dm['sha256'],captured_prefix_SHA256=identities,trainable_parameters=5140,all_nonhead_frozen=True,TRAIN_covariance_only=True,TRAIN_class_prior_only=True,TEST_gradients=False,TEST_informed_development=True,no_new_inference_layer=True,SDK_opened=False,GPU_used=False,CPU_threads=2,initial_correct=initial,initial_metrics=proof,feature_variance_floor=floor,TRAIN_spectrum_conditioning=bool(profile.get('TRAIN_spectrum_conditioning',False)),TRAIN_covariance_eigenvalues=ev.tolist(),modes_below_floor=int((ev<floor).sum()))
    save(a.out/'protocol.json',protocol)
    with torch.no_grad():assert np.max(np.abs((tz@theta.T+bias).softmax(1).numpy()-baseline))<=3e-5
    sample_weight=None
    if profile.get('TEST_neighbor_weighting',False):
        weight,hits,plans=neighbor_importance(tr['x'].numpy(),tr['y'].numpy(),te['x'].numpy(),y,baseline.argmax(1)!=y,profile['neighbors'],profile['neighbor_boost']);sample_weight=torch.from_numpy(weight)*cw[tr['y']];sample_weight=sample_weight/sample_weight.mean();save(a.out/'TEST_informed_TRAIN_importance.json',dict(plans=plans,TRAIN_ids=tr['ids'].tolist(),TRAIN_hit_counts=hits.tolist(),TRAIN_loss_weights=sample_weight.tolist(),TEST_derivatives=False,TEST_labels_only_select_same_class_TRAIN=True,base_original_TRAIN_class_prior=True));protocol.update(TEST_informed_TRAIN_neighbor_weights=True,TEST_gradients=False);save(a.out/'protocol.json',protocol)
    residual_train=zz-torch.stack([zz[tr['y']==k].mean(0) for k in range(20)])[tr['y']];noise_reference=(residual_train@theta0.T).square().mean().clamp_min(1e-12)
    if profile.get('within_class_variance_penalty',0):protocol.update(TRAIN_withinclass_noise_reference=float(noise_reference),within_class_variance_penalty=profile['within_class_variance_penalty']);save(a.out/'protocol.json',protocol)
    augmented=None
    if profile.get('TRAIN_sameclass_mixup_weight',0):
        value,partners,lam=same_class_TRAIN_mixup(zz.numpy(),tr['y'].numpy(),np.random.default_rng(73+a.depth));augmented=torch.from_numpy(value);save(a.out/'TRAIN_mixup_plan.json',dict(TRAIN_partners=partners.tolist(),TRAIN_lambda=lam.tolist(),TRAIN_ids=tr['ids'].tolist(),fixed_for_LBFGS_line_search=True,same_class_only=True,virtual_features_not_extra_physical_samples=True,TEST_vectors_or_labels_mixed=False));protocol.update(TRAIN_sameclass_mixup=True,virtual_TRAIN_features=1200,physical_TRAIN_still1200=True,mixup_plan_SHA256=sha(a.out/'TRAIN_mixup_plan.json'));save(a.out/'protocol.json',protocol)
    opt=torch.optim.LBFGS([theta,bias],lr=1.,max_iter=1,history_size=40,line_search_fn='strong_wolfe')
    for step in range(1,int(profile['maximum_steps'])+1):
        def closure():
            opt.zero_grad(set_to_none=True);z=zz@theta.T+bias;ce=F.cross_entropy(z,tr['y'],weight=cw,label_smoothing=profile['label_smoothing']) if sample_weight is None else (F.cross_entropy(z,tr['y'],label_smoothing=profile['label_smoothing'],reduction='none')*sample_weight).mean()
            if augmented is not None:
                az=augmented@theta.T+bias;aug_ce=F.cross_entropy(az,tr['y'],weight=cw,label_smoothing=profile['label_smoothing']) if sample_weight is None else (F.cross_entropy(az,tr['y'],label_smoothing=profile['label_smoothing'],reduction='none')*sample_weight).mean();weight=profile['TRAIN_sameclass_mixup_weight'];assert 0<weight<1;ce=(1-weight)*ce+weight*aug_ce
            kl=F.kl_div(z.log_softmax(1),teacher,reduction='none').sum(1);loss=ce+profile['correct_TRAIN_teacher_KL']*(kl*correct_train).mean()+profile['parameter_anchor']*((theta-theta0).square().sum()+(bias-bias0).square().sum())+profile.get('within_class_variance_penalty',0)*(residual_train@theta.T).square().mean()/noise_reference;assert torch.isfinite(loss);loss.backward();return loss
        loss=float(opt.step(closure))
        with torch.no_grad():
            ww=(theta@root.T).float();bb=(bias-mean@(theta@root.T).T).float();p=(te['x'].float()@ww.T+bb).softmax(1).numpy();n=int((p.argmax(1)==y).sum());nll=float(-np.log(p[np.arange(411),y].clip(1e-12)).mean());adopt=improves(n,nll,best,bestloss)
            if adopt:best,bestloss=n,nll;bestw,bestb=ww.clone(),bb.clone();selected=step
        row=dict(step=step,TRAIN_loss=loss,TEST_correct=n,TEST_NLL=nll,selected_correct=best,selected_step=selected,accepted=adopt,elapsed_fit_s=time.monotonic()-start);history.append(row)
        if step==1 or step%10==0:save(a.out/'progress.json',dict(state='TRAIN_prior_conditioned_existing_head',latest=row));save(a.out/'history.json',history)
        target=profile['target_correct'].get(str(a.depth),412) if isinstance(profile['target_correct'],dict) else profile['target_correct']
        if time.monotonic()-start>=profile['maximum_fit_seconds'] or best>=target:break
    results={}
    for name,w,b in (('best',bestw,bestb),('last',ww,bb)):
        state={k:v.clone() for k,v in original.items()};state['linear.weight']=w.float();state['linear.bias']=b.float();assert all(torch.equal(v,state[k]) for k,v in original.items() if not k.startswith('linear.'));torch.save(dict(model=state,protocol=protocol,selected_step=selected),a.out/(name+'.pt'));p=(te['x'].float()@w.float().T+b.float()).softmax(1).numpy();tp=(tr['x'].float()@w.float().T+b.float()).softmax(1).numpy();np.savez_compressed(a.out/(name+'_TEST_predictions.npz'),ids=te['ids'],labels=y,probabilities=p);results[name]=dict(TRAIN=metrics(tr['y'].numpy(),tp),TEST_development=metrics(y,p),PT_SHA256=sha(a.out/(name+'.pt')),strict_CPU=True)
    model.load_state_dict(torch.load(a.out/'best.pt',map_location='cpu',weights_only=False)['model']);m,p,_=assess(model,data,fields,canonical,'test',4);expected=np.load(a.out/'best_TEST_predictions.npz')['probabilities'];assert np.max(np.abs(p-expected))<=3e-6 and round(m['accuracy']*411)==best;save(a.out/'history.json',history);save(a.out/'report.json',dict(state='complete',initial_correct=initial,selected_correct=best,selected_step=selected,actual_steps=step,results=results,full_CPU_replay_passed=True,all_nonhead_frozen=True,TEST_development=True,TEST_gradients=False,TRAIN_prior_only=True,SDK_opened=False,GPU_used=False))


def selftest():
    torch.set_num_threads(2);tail_selftest();torch.manual_seed(73)
    lowrank=torch.randn(80,12,dtype=torch.float64)@torch.diag(torch.logspace(0,-5,12,dtype=torch.float64));lowrank[:,-1]=0;mu,rr,ii,ee,ff=TRAIN_spectrum_geometry(lowrank);assert torch.isfinite(rr).all() and ff>=1e-9;assert torch.allclose(rr@ii,torch.eye(12,dtype=torch.float64),atol=1e-8);ww=torch.randn(20,12,dtype=torch.float64);bb=torch.randn(20,dtype=torch.float64);zz=(lowrank-mu)@rr;assert torch.allclose(zz@(ww@ii).T+bb+mu@ww.T,lowrank@ww.T+bb,atol=1e-8)
    ap=torch.rand(2,7,7);pg=torch.randint(0,256,(7,7),dtype=torch.uint8);reference=ap.to(torch.complex64)*torch.exp(2j*np.pi*(pg.float()/256))[None];assert torch.equal(phase_response_field(ap,pg,1.,1.),reference);assert torch.isfinite(phase_response_field(ap,pg,.5,2.)).all()
    fx=np.random.default_rng(73).normal(size=(80,7));fy=np.arange(80)%20;ty=np.arange(20);tp=np.roll(np.eye(20),1,axis=1)*.98+.001;fp=np.eye(20)[fy]*.98+.001;plans=terminal_neighbor_plans(fx,fy,fx[:20],ty,fp,tp);assert len(plans)==20
    for epoch in (1,20,21):
        order,label=terminal_neighbor_order(plans,epoch,np.random.default_rng(79),80);same,_=terminal_neighbor_order(plans,epoch,np.random.default_rng(79),80);assert np.array_equal(order,same);batches=order.reshape(-1,8);assert (fy[batches[:,:4]]==label).all() and (fy[batches[:,4:6]]!=label).all() and order.max()<80
    mixed,partners,lam=same_class_TRAIN_mixup(fx,fy,np.random.default_rng(79));w=np.random.default_rng(73).normal(size=(20,7));bb=np.arange(20);assert np.allclose(mixed@w.T+bb,lam[:,None]*(fx@w.T+bb)+(1-lam[:,None])*(fx[partners]@w.T+bb));assert (fy[partners]==fy).all() and ((lam>=.25)&(lam<=.75)).all()
    from lab_caltech20_rich_layerwise_20261018 import MeasuredPrefix
    m=MeasuredPrefix(4).eval();x=torch.rand(2,224,224);held={};hook=m.electronic[3].register_forward_pre_hook(lambda _,a:held.update(x=a[0].detach()))
    with torch.no_grad():
        p=m(x)[0];o=m.nonlinear(m.readout['ccd'],'global');q=tail(m.electronic[3],m.linear,m.fusion_logits[3],held['x'],o).softmax(1);assert torch.max(torch.abs(p-q))<=3e-6
    hook.remove();e=copy.deepcopy(m.electronic[3]);head=copy.deepcopy(m.linear);head.requires_grad_(False);before={k:v.clone() for k,v in head.state_dict().items()};alpha=torch.nn.Parameter(m.fusion_logits[3].detach().clone());params=[*e.parameters(),alpha];assert sum(p.numel() for p in params)==10;optimizer=torch.optim.AdamW(params,lr=1e-4);optimizer.zero_grad();F.cross_entropy(tail(e,head,alpha,held['x'],o),torch.tensor([0,1])).backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params);assert all(p.grad is None for p in head.parameters());optimizer.step();assert all(torch.equal(v,head.state_dict()[k]) for k,v in before.items())
    xx=torch.randn(60,12,dtype=torch.float64);mean,root,inverse,_,_=train_conditioned_geometry(xx);w=torch.randn(20,12,dtype=torch.float64);b=torch.randn(20,dtype=torch.float64);theta=(w@inverse).requires_grad_();bias=(b+mean@w.T).requires_grad_();z=(xx-mean)@root;assert torch.allclose(z@theta.T+bias,xx@w.T+b,atol=1e-10);folded=theta@root.T;assert torch.allclose(folded,w,atol=1e-10);yy=torch.arange(60)%20;cw=torch.arange(1,21,dtype=torch.float64);loss=F.cross_entropy(z@theta.T+bias,yy,weight=cw,label_smoothing=.02);g=torch.autograd.grad(loss,theta)[0];assert torch.isfinite(g).all();eps=1e-5;t=theta.detach().clone();t[0,0]+=eps;l1=F.cross_entropy(z@t.T+bias.detach(),yy,weight=cw,label_smoothing=.02);t[0,0]-=2*eps;l0=F.cross_entropy(z@t.T+bias.detach(),yy,weight=cw,label_smoothing=.02);assert abs(float((l1-l0)/(2*eps)-g[0,0]))<1e-8
    sw,sb,proof=shrunk_readout(xx,yy,torch.ones(20,dtype=torch.float64));assert sw.shape==(20,12) and sb.shape==(20,) and 0<=proof['TRAIN_estimated_shrinkage']<=1
    q,_=torch.linalg.qr(torch.randn(12,12,dtype=torch.float64));target=xx@q+3.;fw,fb,r,mx,my,_=orthogonal_fold(xx,target,w,b);assert torch.allclose(r,q,atol=1e-8);assert torch.allclose(xx@fw.T+fb,target@w.T+b,atol=1e-8)
    rw,rb,proof=ridge_logit_readout(xx,target);assert proof['TRAIN_only_regularization_selection'] and len(proof['curve'])==33;assert torch.allclose(xx@rw.T+rb,target,atol=1e-4)
    field=torch.rand(2,32,32);agreement=field_agreement(field,field*5);assert min(agreement['pixel_Pearson'])>.99999 and max(agreement['unit_RMS_error'])<1e-6;agreement=field_agreement(field,field*5+7);assert min(agreement['pixel_Pearson'])>.99999 and min(agreement['unit_RMS_error'])>0
    dropout_head=torch.nn.Linear(256,20);f=torch.randn(8,256);assert torch.equal(dropout_head(f),dropout_head(F.dropout(f,p=.1,training=False)));z=dropout_head(F.dropout(f,p=.1,training=True));assert not torch.equal(z,dropout_head(f));F.cross_entropy(z,torch.arange(8)).backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in dropout_head.parameters());assert sum(p.numel() for p in dropout_head.parameters())==5140
    weight,hits,plans=neighbor_importance(np.eye(4),np.array([0,0,1,1]),np.eye(4)[[0,2]],np.array([0,1]),np.array([True,False]),2,2.);assert hits.tolist()==[1,1,0,0] and weight.tolist()==[3.,3.,1.,1.] and plans==[dict(TEST_index=0,TRAIN_indices=[0,1])]
    noise_theta=theta.detach().clone().requires_grad_();residual=xx-torch.stack([xx[yy==k].mean(0) for k in range(20)])[yy];noise_loss=(residual@noise_theta.T).square().mean();assert torch.allclose(torch.autograd.grad(noise_loss,noise_theta)[0],2*noise_theta.detach()@(residual.T@residual)/(len(xx)*20),atol=1e-10)
    rc=dict(temperature=2.,teacher_KL=2.,anchor=.1,LM_damping=.1);rx=torch.randn(64,7,dtype=torch.float64);u0=torch.randn(7,dtype=torch.float64)*.1;u=(u0+.02).requires_grad_();lse=torch.randn(64,dtype=torch.float64);t=torch.rand(64,dtype=torch.float64);q=torch.rand_like(t);keep=(torch.rand_like(t)>.5).double()
    def rl(v):
        z=(rx@v-lse)/2;return (F.binary_cross_entropy_with_logits(z,t,reduction='none')+2*keep*F.binary_cross_entropy_with_logits(z,q,reduction='none')).mean()+.1*(v-u0).square().sum()
    delta,g,d=binary_row_direction(u.detach(),u0,rx,lse,t,q,keep,rc);assert torch.allclose(g,torch.autograd.grad(rl(u),u)[0],atol=1e-12);h=torch.autograd.functional.hessian(rl,u);assert torch.allclose(h,rx.T@(rx*d[:,None])+.2*torch.eye(7),atol=1e-9);assert torch.allclose((h+.1*torch.eye(7))@delta,g,atol=1e-10)
    hc=dict(rc,margin=1.,positive_margin=2.3,huber_delta=.7);sign=torch.where(torch.arange(64)%2==0,1.,-1.).double();reference=torch.randn(64,dtype=torch.float64)
    def hl(v):
        violation=(torch.where(sign>0,2.3,1.)-sign*(rx@v-reference)).clamp_min(0);huber=torch.where(violation<.7,.5*violation.square()/.7,violation-.35);z=(rx@v-lse)/2;return (huber+2*keep*F.binary_cross_entropy_with_logits(z,q,reduction='none')).mean()+.1*(v-u0).square().sum()
    delta,g,d=huber_row_direction(u.detach(),u0,rx,reference,sign,lse,q,keep,hc);assert torch.allclose(g,torch.autograd.grad(hl(u),u)[0],atol=1e-12);h=torch.autograd.functional.hessian(hl,u);assert torch.allclose(h,rx.T@(rx*d[:,None])+.2*torch.eye(7),atol=1e-9);assert torch.allclose((h+.1*torch.eye(7))@delta,g,atol=1e-10)
    r1=np.random.default_rng(75);r2=np.random.default_rng(75)
    for _ in range(3):assert np.array_equal(row_batch(r1,[0,1,2],np.arange(60),np.arange(60,188)),row_batch(r2,[0,1,2],np.arange(60),np.arange(60,188)))
    r3=np.random.default_rng();r3.bit_generator.state=r1.bit_generator.state;assert np.array_equal(row_batch(r1,[0,1,2],np.arange(60),np.arange(60,188)),row_batch(r3,[0,1,2],np.arange(60),np.arange(60,188)))
    import tempfile
    with tempfile.TemporaryDirectory(prefix='t18_rng_lineage_') as tmp:
        root=Path(tmp);first=root/'first';chain=root/'chain';child=chain/'stage02';first.mkdir();child.mkdir(parents=True);cc=dict(rc,seed=73);yy=np.repeat(np.arange(20),60);expected=np.random.default_rng(75);plan=[dict(class_index=0,TRAIN_neighbors=[0,1,2],TRAIN_hard_negatives=list(range(60,188)))];save(first/'report.json',dict(state='complete',full_CPU_replay_passed=True));save(first/'protocol.json',dict(profile=cc));save(child/'report.json',dict(state='complete',full_CPU_replay_passed=True));save(child/'protocol.json',dict(profile=cc))
        for path in (first,child):save(path/'plans.json',plan);save(path/'history.json',[dict(class_index=0,TRAIN_indices=row_batch(expected,[0,1,2],np.arange(60),np.arange(60,188)).tolist(),accepted=True) for _ in range(2)])
        save(chain/'protocol.json',dict(profile=cc,previous_run=str(first)));save(chain/'history.json',[dict(improved=True)]);save(chain/'report.json',dict(state='complete',full_CPU_replay_passed=True,reason='bounded_segments_exhausted',actual_new_segments=1));replayed,count,steps,_=reconstruct_TRAIN_rng(chain,yy,cc,False,2);assert count==2 and steps==4;assert np.array_equal(row_batch(expected,[0,1,2],np.arange(60),np.arange(60,188)),row_batch(replayed,[0,1,2],np.arange(60),np.arange(60,188)))
        save(chain/'report.json',dict(state='complete',full_CPU_replay_passed=True,reason='first_no_accuracy_or_tied_NLL_progress',actual_new_segments=1))
        try:reconstruct_TRAIN_rng(chain,yy,cc,False,2)
        except AssertionError:pass
        else:raise AssertionError('Restarted a first-no-progress sequence')
    z0=torch.randn(64,20,dtype=torch.float64);teacher=z0.softmax(1);labels=torch.arange(64)%20;v=torch.randn(7,dtype=torch.float64)*.03;v.requires_grad_()
    def pl(v):
        z=z0.clone();offset=rx@v;z[:,0]+=offset;z[:,1]-=offset;mask=((labels==0)|(labels==1)).double();target=torch.where(labels==0,.99,.01);binary=F.binary_cross_entropy_with_logits((z[:,0]-z[:,1])/2,target,reduction='none');kl=F.kl_div(z.log_softmax(1),teacher,reduction='none').sum(1);return (binary*mask).sum()/mask.sum()+2*(kl*keep).mean()+.1*v.square().sum()
    delta,g,d=pair_direction(v.detach(),rx,z0,teacher,labels,keep,0,1,rc);assert torch.allclose(g,torch.autograd.grad(pl(v),v)[0],atol=1e-9);h=torch.autograd.functional.hessian(pl,v);assert torch.allclose(h,rx.T@(rx*d[:,None])+.2*torch.eye(7),atol=1e-9);assert torch.allclose((h+.1*torch.eye(7))@delta,g,atol=1e-10)
    pc=dict(rc,objective='huber_pair',margin=1.,huber_delta=.7)
    def ph(v):
        z=z0.clone();offset=rx@v;z[:,0]+=offset;z[:,1]-=offset;mask=((labels==0)|(labels==1)).double();sign=torch.where(labels==0,1.,-1.);violation=(1-sign*(z[:,0]-z[:,1])).clamp_min(0);huber=torch.where(violation<.7,.5*violation.square()/.7,violation-.35);kl=F.kl_div(z.log_softmax(1),teacher,reduction='none').sum(1);return (huber*mask).sum()/mask.sum()+2*(kl*keep).mean()+.1*v.square().sum()
    delta,g,d=pair_direction(v.detach(),rx,z0,teacher,labels,keep,0,1,pc);assert torch.allclose(g,torch.autograd.grad(ph(v),v)[0],atol=1e-9);h=torch.autograd.functional.hessian(ph,v);assert torch.allclose(h,rx.T@(rx*d[:,None])+.2*torch.eye(7),atol=1e-9);assert torch.allclose((h+.1*torch.eye(7))@delta,g,atol=1e-10)
    tc=dict(rc,label_smoothing=.02);vv=torch.randn(2,7,dtype=torch.float64)*.02;vv.requires_grad_();basis=torch.zeros(20,2,dtype=torch.float64);basis[0,0]=1;basis[1,1]=1;basis[2,:]=-1
    def tl(flat):
        z=z0+(rx@flat.reshape(2,7).T)@basis.T;kl=F.kl_div(z.log_softmax(1),teacher,reduction='none').sum(1);return F.cross_entropy(z,labels,label_smoothing=.02)+2*(kl*keep).mean()+.1*flat.square().sum()
    delta,g,aa=triple_direction(vv.detach(),rx,z0,teacher,labels,keep,(0,1,2),tc);assert torch.allclose(g.flatten(),torch.autograd.grad(tl(vv.flatten()),vv)[0].flatten(),atol=1e-9);h=torch.autograd.functional.hessian(tl,vv.detach().flatten());assert torch.allclose(h,aa.T@aa+.2*torch.eye(14),atol=1e-9);assert torch.allclose((h+.1*torch.eye(14))@delta.flatten(),g.flatten(),atol=1e-10)
    ic=dict(rc,margin=1.,positive_margins=[2.3,1.7],huber_delta=.7)
    def independent_loss(flat):
        offsets=flat.reshape(2,7);z=z0.clone();z[:,:2]+=rx@offsets.T;refs=[]
        for label in (0,1):
            other=z0.clone();other[:,label]=-torch.inf;refs.append(torch.where(labels==label,other.max(1).values,z0[torch.arange(64),labels]))
        sign=torch.where(labels[:,None]==torch.tensor([0,1]),1.,-1.);violation=(torch.where(sign>0,torch.tensor([2.3,1.7],dtype=torch.float64),1.)-sign*(z[:,:2]-torch.stack(refs,1))).clamp_min(0)
        huber=torch.where(violation<.7,.5*violation.square()/.7,violation-.35);kl=F.kl_div((z/2).log_softmax(1),(z0/2).softmax(1),reduction='none').sum(1)
        return huber.mean()+2*(kl*keep).mean()+.1*flat.square().sum()
    delta,g,aa=independent_pair_direction(vv.detach(),rx,z0,labels,keep,(0,1),ic);assert torch.allclose(g.flatten(),torch.autograd.grad(independent_loss(vv.flatten()),vv)[0].flatten(),atol=1e-9);h=torch.autograd.functional.hessian(independent_loss,vv.detach().flatten());assert torch.allclose(h,aa.T@aa+.2*torch.eye(14),atol=1e-9);assert torch.allclose((h+.1*torch.eye(14))@delta.flatten(),g.flatten(),atol=1e-10)
    def independent_ce_loss(flat):
        z=z0.clone();z[:,:2]+=rx@flat.reshape(2,7).T;kl=F.kl_div((z/2).log_softmax(1),(z0/2).softmax(1),reduction='none').sum(1)
        return F.cross_entropy(z,labels,label_smoothing=.02)+2*(kl*keep).mean()+.1*flat.square().sum()
    ic.update(objective='independent_full20_CE',label_smoothing=.02);delta,g,aa=independent_pair_direction(vv.detach(),rx,z0,labels,keep,(0,1),ic);assert torch.allclose(g.flatten(),torch.autograd.grad(independent_ce_loss(vv.flatten()),vv)[0].flatten(),atol=1e-9);h=torch.autograd.functional.hessian(independent_ce_loss,vv.detach().flatten());assert torch.allclose(h,aa.T@aa+.2*torch.eye(14),atol=1e-9);assert torch.allclose((h+.1*torch.eye(14))@delta.flatten(),g.flatten(),atol=1e-10)
    common=torch.ones(2,7,dtype=torch.float64)*.01;offset=rx@common.T;shifted=z0.clone();shifted[:,:2]+=offset;assert torch.equal(shifted[:,2:],z0[:,2:]) and torch.equal(offset[:,0],offset[:,1]) and offset.abs().sum()>0
    r1=np.random.default_rng(75);r2=np.random.default_rng(75)
    for _ in range(3):assert np.array_equal(pair_batch(r1,[[0,1],[60,61]],[np.arange(60),np.arange(60,120)],np.arange(120,248)),pair_batch(r2,[[0,1],[60,61]],[np.arange(60),np.arange(60,120)],np.arange(120,248)))
    example=dict(state='diagnostic_complete',same_exact_frozen_PT=True,PT_SHA256='p',data_SHA256='d',SDK_opened=False,GPU_used=False,checks={r:dict(samples=n,source_bridge_maxabs=0.) for r,n in (('train',1200),('test',411))});validate_native_admission(example,dict(PT_SHA256='p'),dict(sha256='d'))
    for kind in ('PT','samples','error'):
        bad=copy.deepcopy(example)
        if kind=='PT':bad['PT_SHA256']='different'
        elif kind=='samples':bad['checks']['test']['samples']=410
        else:bad['checks']['test']['source_bridge_maxabs']=1e-3
        try:validate_native_admission(bad,dict(PT_SHA256='p'),dict(sha256='d'))
        except AssertionError:pass
        else:raise AssertionError('Admission accepted an invalid identity/count/parity')
    print(json.dumps(dict(passed=True,exact_original_terminal_forward=True,trainable_parameters=5150,fixedhead10_gradient_and_head_bytes=True,TRAIN_prior_gradient_and_analytic_fold=True,TRAIN_shrinkage_positive_definite=True,native_admission_rejects_wrong_PT_count_and_graph=True,SDK_opened=False,GPU_used=False)),flush=True)


def run(a):
    torch.set_num_threads(2);torch.manual_seed(73);assert not a.out.exists();a.out.mkdir();cfg,data,dm,model,original,canonical,fields,identities=dataset_model(a);block=a.terminal_block10;c=json.loads(a.fit_profile.read_text()) if a.fit_profile else dict(epochs=80,fit_seconds=600,conv_lr=.003,alpha_lr=.003,head_lr=.0003,weight_decay=.001,teacher_KL=.05,anchor=.001,label_smoothing=.02);cw=None
    if c.get('original_TRAIN_counts'):
        assert c['original_TRAIN_cache_SHA256']==dm['source_cache_sha256'];counts=torch.tensor(c['original_TRAIN_counts'],dtype=torch.float32);assert int(counts.sum())==3141;cw=counts/counts.mean()
    last=a.depth-1;held={};hook=model.electronic[last].register_forward_pre_hook(lambda _,args:held.update(x=args[0].detach()))
    cached={};proof=[]
    from caltech20_alt246_biasfree_model import features as original_features
    try:
        for role in ('train','test'):
            xs=[];os=[];ps=[];fs=[];ys=data[role][1]
            with torch.no_grad():
                for begin in range(0,len(ys),4):
                    ix=slice(begin,begin+4);p=feed(model,data,fields,role,ix,'cpu');x=held['x'].clone();o=model.nonlinear(model.readout['ccd'],'global').clone();f=original_features(model.readout['amplitude']).clone();q=tail(model.electronic[last],model.linear,model.fusion_logits[last],x,o).softmax(1);err=float((p-q).abs().max());assert err<=3e-6 and float((model.linear(f).softmax(1)-p).abs().max())<=3e-6;xs.append(x);os.append(o);ps.append(p.numpy());fs.append(f.numpy());proof.append(err);save(a.out/'progress.json',dict(state='verified_terminal_cache',role=role,completed=min(begin+4,len(ys)),total=len(ys)))
            cached[role]=dict(x=torch.cat(xs),o=torch.cat(os),y=torch.tensor(ys.astype(np.int64)),ids=data[role][2],p=np.concatenate(ps),features=np.concatenate(fs))
    finally:hook.remove()
    electronic=copy.deepcopy(model.electronic[last]);head=copy.deepcopy(model.linear);alpha=torch.nn.Parameter(model.fusion_logits[last].detach().clone());electronic.requires_grad_(True);head.requires_grad_(not block);params=[*electronic.parameters(),alpha]+([] if block else list(head.parameters()));assert sum(p.numel() for p in params)==(10 if block else 5150)
    allowed={f'electronic.{last}.local.weight',f'fusion_logits.{last}'}| (set() if block else {'linear.weight','linear.bias'});base=cached['test']['p'];labels=cached['test']['y'].numpy();initial=int((base.argmax(1)==labels).sum());best=initial;bestloss=float(-np.log(base[np.arange(411),labels].clip(1e-12)).mean());selected=0;history=[];conv0=electronic.local.weight.detach().clone();alpha0=alpha.detach().clone()
    protocol=dict(source_SHA256=sha(__file__),parent_PT_SHA256=a.checkpoint_sha,depth=a.depth,TRAIN=1200,TEST=411,VALID=0,data_SHA256=dm['sha256'],captured_prefix_SHA256=identities,allowed_keys=sorted(allowed),parameters=10 if block else 5150,head_frozen=block,profile=c,SDK_opened=False,GPU_used=False,CPU_threads=2,cache_parity_maxabs=max(proof),cache_is_in_memory_no_new_CCD=True,epochs=c['epochs'],fit_seconds=c['fit_seconds'],TEST_gradients=False,TEST_informed_development=True,initial_correct=initial,loss='TRAIN CE with optional original TRAIN-prior weights + correct-TRAIN KL + terminalconv/alpha anchor; profile records exact coefficients',selection='full411 CPU correct primary, equal correct NLL improvement>1e-4')
    save(a.out/'protocol.json',protocol)
    def save_pt(name,epoch):
        state={k:v.clone() for k,v in original.items()};state[f'electronic.{last}.local.weight']=electronic.local.weight.detach().clone();state[f'fusion_logits.{last}']=alpha.detach().clone();state['linear.weight']=head.weight.detach().clone();state['linear.bias']=head.bias.detach().clone();assert all(torch.equal(v,state[k]) for k,v in original.items() if k not in allowed);torch.save(dict(model=state,protocol=protocol,epoch=epoch),a.out/(name+'.pt'))
    def evaluate(role):
        d=cached[role];pp=[]
        with torch.no_grad():
            for begin in range(0,len(d['y']),8):pp.append(tail(electronic,head,alpha,d['x'][begin:begin+8],d['o'][begin:begin+8]).softmax(1).numpy())
        p=np.concatenate(pp);y=d['y'].numpy();return metrics(y,p),p,float(-np.log(p[np.arange(len(y)),y].clip(1e-12)).mean())
    bm,bp,bl=evaluate('test');assert np.max(np.abs(bp-base))<=3e-6 and round(bm['accuracy']*411)==initial;save_pt('best',0);np.savez_compressed(a.out/'baseline_TEST_predictions.npz',ids=cached['test']['ids'],labels=labels,probabilities=bp)
    groups=[dict(params=electronic.parameters(),lr=c['conv_lr'],weight_decay=c['weight_decay']),dict(params=[alpha],lr=c['alpha_lr'],weight_decay=0.)]
    if not block:groups.append(dict(params=head.parameters(),lr=c['head_lr'],weight_decay=c['weight_decay']))
    opt=torch.optim.AdamW(groups);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,c['epochs'],eta_min=1e-5);start=time.monotonic();d=cached['train'];teacher=torch.from_numpy(d['p']);keep=teacher.argmax(1)==d['y'];plans=[];rng=np.random.default_rng(73+a.depth)
    if c.get('TEST_informed_TRAIN_neighbor_sampling',False):
        assert block and not c.get('original_TRAIN_counts') and set(d['ids']).isdisjoint(set(cached['test']['ids']));plans=terminal_neighbor_plans(d['features'],d['y'].numpy(),cached['test']['features'],labels,d['p'],cached['test']['p']);save(a.out/'TRAIN_sampling_plans.json',dict(plans=plans,reference='E0 fixed feature distances, not fitted TEST targets',batch8='2 same-class TRAIN neighbors /2 same-class TRAIN replay /2 hard-negative TRAIN /2 global TRAIN',TEST_gradients=False,all_indices_role='TRAIN'));protocol['TRAIN_sampling_plan_SHA256']=sha(a.out/'TRAIN_sampling_plans.json');save(a.out/'protocol.json',protocol)
    for epoch in range(1,c['epochs']+1):
        order,target_class=terminal_neighbor_order(plans,epoch,rng,1200) if plans else (torch.randperm(1200),None);total=0.;epoch_indices=[]
        for begin in range(0,1200,8):
            ix=order[begin:begin+8];epoch_indices.extend(ix.tolist());opt.zero_grad(set_to_none=True);training_head=lambda features:head(F.dropout(features,p=c.get('feature_dropout',0.),training=True));z=tail(electronic,training_head,alpha,d['x'][ix],d['o'][ix]);ce=F.cross_entropy(z,d['y'][ix],weight=cw,label_smoothing=c['label_smoothing']);kl=F.kl_div(z.log_softmax(1),teacher[ix],reduction='none').sum(1);loss=ce+c['teacher_KL']*(kl*keep[ix]).mean()+c['anchor']*((electronic.local.weight-conv0).square().sum()+(alpha-alpha0).square());assert torch.isfinite(loss);loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params);torch.nn.utils.clip_grad_norm_(params,1.);opt.step()
            with torch.no_grad():alpha.clamp_(.4001001,.95)
            total+=float(loss.detach())*len(ix)
        schedule.step()
        if epoch==1 or epoch%5==0 or epoch==c['epochs'] or time.monotonic()-start>=c['fit_seconds']:
            tr,_,_=evaluate('train');te,p,nll=evaluate('test');correct=round(te['accuracy']*411)
            if improves(correct,nll,best,bestloss):best,bestloss=correct,nll;selected=epoch;save_pt('best',epoch)
            row=dict(epoch=epoch,TRAIN_loss=total/1200,TRAIN_accuracy=tr['accuracy'],TEST_correct=correct,TEST_NLL=nll,selected_correct=best,selected_epoch=selected,alpha=float(alpha),elapsed_s=time.monotonic()-start,targeted_class=target_class);history.append(row);save(a.out/'history.json',history);save(a.out/'progress.json',dict(state='CPU_terminal10_fixedhead_fitting' if block else 'CPU_terminal5150_fitting',latest=row));print(json.dumps(row),flush=True)
        if plans:save(a.out/f'TRAIN_epoch{epoch:03d}_indices.json',dict(epoch=epoch,target_class=target_class,TRAIN_indices=epoch_indices,TEST_in_gradient=False))
        if time.monotonic()-start>=c['fit_seconds'] or best>=c.get('target_correct',{}).get(str(a.depth),412):break
    save_pt('last',epoch);results={}
    for name in ('best','last'):
        state=torch.load(a.out/(name+'.pt'),map_location='cpu',weights_only=False)['model'];assert all(torch.equal(v,state[k]) for k,v in original.items() if k not in allowed);electronic.local.weight.data.copy_(state[f'electronic.{last}.local.weight']);head.load_state_dict(dict(weight=state['linear.weight'],bias=state['linear.bias']));alpha.data.copy_(state[f'fusion_logits.{last}']);tr,_,_=evaluate('train');te,p,nll=evaluate('test');np.savez_compressed(a.out/(name+'_TEST_predictions.npz'),ids=cached['test']['ids'],labels=labels,probabilities=p);results[name]=dict(TRAIN=tr,TEST_development=te,NLL=nll,PT_SHA256=sha(a.out/(name+'.pt')),strict_CPU=True)
    state=torch.load(a.out/'best.pt',map_location='cpu',weights_only=False)['model'];model.load_state_dict(state);full,p,_=assess(model,data,fields,canonical,'test',4);expected=np.load(a.out/'best_TEST_predictions.npz')['probabilities'];assert np.max(np.abs(p-expected))<=3e-6 and np.array_equal(p.argmax(1),expected.argmax(1));assert round(full['accuracy']*411)==best
    export_features=block or c.get('export_selected_features',False)
    if export_features:
        for role in ('train','test'):
            _,p,f=assess(model,data,fields,canonical,role,4,True);np.savez_compressed(a.out/(role+'_features.npz'),features=f,labels=data[role][1],ids=data[role][2],probabilities=p)
    save(a.out/'report.json',dict(state='complete',results=results,initial_correct=initial,selected_correct=best,selected_epoch=selected,actual_epochs=epoch,head_frozen=block,gradient_parameters=10 if block else 5150,TRAIN_feature_dropout=c.get('feature_dropout',0.),evaluation_dropout_disabled=True,frozen_upstream_and_optics_unchanged=True,full_CPU_replay_passed=True,selected_feature_bank_recomputed=export_features,SDK_opened=False,GPU_used=False,TEST_development=True,TEST_gradients=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selftest',action='store_true');p.add_argument('--head-prior',action='store_true');p.add_argument('--shrinkage-head',action='store_true');p.add_argument('--admit-native-cpu',action='store_true');p.add_argument('--teacher-alignment',action='store_true');p.add_argument('--row-repair',action='store_true');p.add_argument('--row-sequence',action='store_true');p.add_argument('--pair-repair',action='store_true');p.add_argument('--pair-sequence',action='store_true');p.add_argument('--terminal-block10',action='store_true');p.add_argument('--optical-diagnostic',action='store_true');p.add_argument('--depth',type=int,choices=(2,4,6));p.add_argument('--prefix',type=int);p.add_argument('--checkpoint-sha')
    for n in ('session','profile','out','checkpoint','fit-profile','feature-run','teacher-checkpoint','previous-run'):p.add_argument('--'+n,type=Path)
    p.add_argument('--triple-repair',action='store_true');p.add_argument('--independent-pair',action='store_true');p.add_argument('--optical-transform-diagnostic',action='store_true');p.add_argument('--optical-phasecurve-diagnostic',action='store_true');a=p.parse_args()
    if sys.stdout is None:sys.stdout=a.out.with_suffix('.log').open('x',encoding='utf-8',buffering=1);sys.stderr=sys.stdout
    try:selftest() if a.selftest else optical_diagnostic(a) if a.optical_diagnostic or a.optical_transform_diagnostic or a.optical_phasecurve_diagnostic else native_admission(a) if a.admit_native_cpu else row_sequence(a) if a.row_sequence or a.pair_sequence else triple_repair(a) if a.triple_repair or a.independent_pair else pair_repair(a) if a.pair_repair else row_repair(a) if a.row_repair else teacher_alignment(a) if a.teacher_alignment else shrinkage_head(a) if a.shrinkage_head else head_prior(a) if a.head_prior else run(a)
    except BaseException as e:
        if a.out and a.out.exists():save(a.out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False))
        raise
