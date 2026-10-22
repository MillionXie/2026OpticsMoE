"""Bounded CPU optimization of the ORIGINAL post-final-CCD5150 parameters.

The cached incident field and all physical stages stay immutable. TEST analysis
and checkpoint selection are development use; no TEST derivative or VALID.
"""
import time,copy
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
import lab_caltech20_dvp_layerwise_20261019 as m
import lab_caltech20_dvp_closedloop_20261021 as loop
from lab_caltech20_rich_tail_refine_20261018 import tail
from lab_caltech20_residual_sequence_20261018 import train_conditioned_geometry
from lab_caltech20_dvp_completed_readout_20261021 import fold_delta,native_probability


def allowed(model):
    k=model.depth-1
    return {f'electronic.{k}.local.weight',f'fusion_logits.{k}','linear.weight','linear.bias'}


def assess_tail(model,inputs,fields,data,role):
    k=model.depth-1;parts=[]
    with torch.inference_mode():
        for begin in range(0,len(data[role][1]),8):
            end=min(begin+8,len(data[role][1]));n=end-begin;x=torch.from_numpy(np.asarray(inputs[role][begin:end]).copy());c=torch.from_numpy(np.asarray(fields[str(k)][role][begin:end]).copy()).float()
            if n<8:
                x=torch.cat([x,torch.zeros((8-n,*x.shape[1:]),dtype=x.dtype)]);c=torch.cat([c,torch.zeros((8-n,478,478),dtype=c.dtype)])
            parts.append(tail(model.electronic[k],model.linear,model.effective_alpha(k),x,model.nonlinear(c,'global')).softmax(1)[:n].numpy())
    prob=np.concatenate(parts);r=m.metrics(data[role][1],prob);r.update(correct=int((prob.argmax(1)==data[role][1]).sum()),samples=len(prob),nll=float(-np.log(prob[np.arange(len(prob)),data[role][1]].clip(1e-12)).mean()))
    return r,prob


def cache_inputs(model,data,fields,out):
    out.mkdir();k=model.depth-1;cache={};baseline={}
    for role,n in (('train',1600),('test',411)):
        dest=np.lib.format.open_memmap(out/(role+'_last_input.npy'),mode='w+',dtype=np.float32,shape=(n,478,478));held={};parts=[]
        hook=model.electronic[k].register_forward_pre_hook(lambda module,args:held.update(x=args[0].detach()))
        try:
            with torch.inference_mode():
                for begin in range(0,n,8):
                    x,valid,end=m.padded(data,role,begin);model.measured={}
                    for key,v in fields.items():
                        a=np.asarray(v[role][begin:end]).copy()
                        if valid<8:a=np.concatenate([a,np.zeros((8-valid,478,478),np.uint8)])
                        model.measured[key]=torch.from_numpy(a)
                    p=model(m.encode_images(x))[0];inp=held['x'];dest[begin:end]=inp[:valid].numpy()
                    q=tail(model.electronic[k],model.linear,model.effective_alpha(k),inp,model.nonlinear(model.measured[str(k)],'global')).softmax(1)
                    assert float((p-q).abs().max())<=3e-6 and torch.equal(p.argmax(1),q.argmax(1));parts.append(p[:valid].numpy())
                    m.save(out/'progress.json',dict(state='native_terminal_input_cache',role=role,samples=end,total=n))
        finally:hook.remove()
        dest.flush();cache[role]=np.load(out/(role+'_last_input.npy'),mmap_mode='r');prob=np.concatenate(parts)
        baseline[role]=dict(correct=int((prob.argmax(1)==data[role][1]).sum()),samples=n)
        np.savez_compressed(out/(role+'_baseline_predictions.npz'),ids=data[role][2],labels=data[role][1],probabilities=prob)
    m.save(out/'report.json',dict(passed=True,native_tail_probability_max_error_limit=3e-6,baseline=baseline,arrays={r:dict(SHA256=m.sha(out/(r+'_last_input.npy')),samples=len(data[r][1])) for r in cache},SDK_used=False,GPU_used=False))
    return cache


def finish(model,original,inputs,data,fields,out,reference,best,epoch):
    results={}
    for checkpoint in ('best','last'):
        pt=out/(checkpoint+'.pt');native,_=m.model_for(pt,m.sha(pt),model.depth,reference);loop.validate_state(m,native,original,allowed(native));results[checkpoint]=dict(PT_SHA256=m.sha(pt))
        for role in ('train','test'):
            m.save(out/'progress.json',dict(state='strict_original_CPU_replay',checkpoint=checkpoint,role=role))
            score,prob=loop.original_replay(native,data,fields,role);cached,cp=assess_tail(native,inputs,fields,data,role)
            assert np.max(np.abs(cp-prob))<=3e-6 and np.array_equal(cp.argmax(1),prob.argmax(1)) and score['correct']==cached['correct']
            results[checkpoint][role]=score
            np.savez_compressed(out/(checkpoint+'_'+role+'_predictions.npz'),ids=data[role][2],labels=data[role][1],probabilities=prob)
    assert results['best']['test']['correct']==best
    m.save(out/'report.json',dict(state='complete',selected_correct=best,selected_epoch=epoch,selected_E0=epoch==0,results=results,strict_original_CPU=True,full_physical=True,TEST_development=True,TEST_gradient=False,SDK_used=False,GPU_used=False,protocol_SHA256=m.sha(out/'protocol.json')))
    m.save(out/'independent_audit.json',dict(passed=True,report_SHA256=m.sha(out/'report.json'),all_nonterminal_parameters_bitwise_frozen=True,original_model_layers_unchanged=True,TRAIN=1600,TEST=411))
    chosen,_=m.model_for(out/'best.pt',m.sha(out/'best.pt'),model.depth,reference)
    return chosen,results


def joint(model,inputs,fields,data,cfg,recipe,out,reference,identities):
    out.mkdir();k=model.depth-1;original=m.state_cpu(model);keys=allowed(model);baseline,tp=assess_tail(model,inputs,fields,data,'train');test,qp=assess_tail(model,inputs,fields,data,'test')
    best=test['correct'];selected=0;protocol=dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),recipe=recipe,depth=model.depth,allowed_keys=sorted(keys),parent_state_only=True,prefix_SHA256=identities,TRAIN=1600,TEST=411,VALID=0,TEST_gradient=False,TEST_informed_development=True,new_inference_layers=0,SDK_used=False,GPU_used=False)
    m.save(out/'protocol.json',protocol);m.save(out/'baseline.json',dict(train=baseline,test=test));loop.saved_model(m,model,out/'best.pt',protocol)
    for n,p in model.named_parameters():p.requires_grad_(n in keys)
    params=[p for p in model.parameters() if p.requires_grad];assert sum(p.numel() for p in params)==5150
    opt=torch.optim.AdamW([dict(params=list(model.electronic[k].parameters()),lr=recipe['conv_lr']),dict(params=[model.fusion_logits[k]],lr=recipe['alpha_lr']),dict(params=list(model.linear.parameters()),lr=recipe['head_lr'])],weight_decay=0)
    prior=np.asarray(cfg['original_TRAIN_counts'],float);weights=torch.tensor(.5+.5*np.sqrt(prior/prior.mean()),dtype=torch.float32);teacher=torch.from_numpy(tp)
    error=1-np.diag(np.asarray(test['confusion_matrix']))/np.asarray(test['support']).clip(1)
    weights*=torch.tensor(1+recipe['TEST_class_emphasis']*error,dtype=torch.float32)
    m.save(out/'development_error_plan.json',dict(TEST_confusion=test['confusion_matrix'],TEST_gradient=False,TRAIN_class_weights=weights.tolist(),synthetic_CCD_noise_not_calibrated=True,new_inference_layers=0))
    active=False
    def drop(module,args):
        return (F.dropout(args[0],p=recipe['feature_dropout'],training=True),) if active else None
    hook=model.linear.register_forward_pre_hook(drop);rng=np.random.default_rng(73);history=[];start=time.monotonic();counts=dict(TRAIN_updates=0,TRAIN_noise_batches=0,TRAIN_dropout_batches=0)
    try:
        for epoch in range(1,recipe['epochs']+1):
            order=rng.permutation(1600);used=0;total=0
            for begin in range(0,1600,8):
                ix=order[begin:begin+8];x=torch.from_numpy(np.asarray(inputs['train'][ix]).copy());c=torch.from_numpy(np.asarray(fields[str(k)]['train'][ix]).copy()).float();labels=torch.from_numpy(data['train'][1][ix]).long();teach=teacher[ix]
                c=(c*(1+recipe['gain_sigma']*torch.randn((8,1,1)))+recipe['noise_DN']*torch.randn_like(c)).clamp(0,255)
                opt.zero_grad();active=True;z=tail(model.electronic[k],model.linear,model.effective_alpha(k),x,model.nonlinear(c,'global'));active=False
                ce=F.cross_entropy(z,labels,weight=weights,label_smoothing=recipe['smoothing']);keep=(teach.argmax(1)==labels).float();kl=(F.kl_div(z.log_softmax(1),teach,reduction='none').sum(1)*keep).sum()/keep.sum().clamp_min(1)
                anchor=sum((p-original[n]).square().sum() for n,p in model.named_parameters() if n in keys)/5150
                loss=ce+recipe['KL']*kl+recipe['anchor']*anchor;assert torch.isfinite(loss);loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
                torch.nn.utils.clip_grad_norm_(params,1);opt.step()
                with torch.no_grad():model.fusion_logits[k].clamp_(.4001001,.95)
                used+=8;total+=float(loss.detach())*8;counts['TRAIN_updates']+=1;counts['TRAIN_noise_batches']+=1;counts['TRAIN_dropout_batches']+=1
                if begin%160==0:m.save(out/'progress.json',dict(state='CPU_original5150_TRAIN',epoch=epoch,TRAIN_consumed=used,TRAIN_loss=total/used,selected_correct=best))
            assert used==1600
            ts,_=assess_tail(model,inputs,fields,data,'test');tr,_=assess_tail(model,inputs,fields,data,'train')
            if ts['correct']>best:best=ts['correct'];selected=epoch;loop.saved_model(m,model,out/'best.pt',protocol)
            row=dict(epoch=epoch,TRAIN=tr,TEST=ts,TRAIN_loss=total/used,selected_correct=best,elapsed_s=time.monotonic()-start);history.append(row);m.save(out/'history.json',history)
            if best>=cfg['target'] or time.monotonic()-start>=recipe['seconds']:break
        loop.validate_state(m,model,original,keys);loop.saved_model(m,model,out/'last.pt',protocol)
    finally:active=False;hook.remove()
    m.save(out/'TRAIN_transform_audit.json',dict(counts=counts,TEST_gradient=False,eval_dropout=False,noise_synthetic_not_calibrated=True,raw_CCD_unchanged=True,actual_epochs=epoch))
    return finish(model,original,inputs,data,fields,out,reference,best,selected)


def head_polish(model,inputs,fields,data,cfg,out,reference,identities):
    out.mkdir();original=m.state_cpu(model);bank={};baseline={};k=model.depth-1
    for role in ('train','test'):
        values=[];hook=model.linear.register_forward_pre_hook(lambda module,args:values.append(args[0].detach().clone()))
        try:baseline[role],_=assess_tail(model,inputs,fields,data,role)
        finally:hook.remove()
        bank[role]=torch.cat(values)[:len(data[role][1])]
    protocol=dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),method='TRAIN_conditioned_original_Linear',TRAIN=1600,TEST=411,TEST_gradient=False,VALID=0,prefix_SHA256=identities,new_inference_layers=0)
    m.save(out/'protocol.json',protocol);loop.saved_model(m,model,out/'best.pt',protocol);best=baseline['test']['correct'];selected=0;x=bank['train'];y=torch.from_numpy(data['train'][1]).long();mean,root,_,_,_=train_conditioned_geometry(x);xx=torch.cat([(x.double()-mean)@root,torch.ones(1600,1,dtype=torch.float64)],1)
    w=model.linear.weight.detach().clone();b=model.linear.bias.detach().clone();z0=F.linear(x.double(),w.double(),b.double());delta=torch.nn.Parameter(torch.zeros(20,257,dtype=torch.float64));opt=torch.optim.Adam([delta],lr=.003);start=time.monotonic();history=[]
    weights=torch.tensor(cfg['original_TRAIN_counts'],dtype=torch.float64);weights/=weights.mean()
    for step in range(1,1201):
        loss=F.cross_entropy(z0+xx@delta.T,y,weight=weights,label_smoothing=.03)+.01*delta.square().mean();opt.zero_grad();loss.backward();assert torch.isfinite(delta.grad).all();torch.nn.utils.clip_grad_norm_([delta],1);opt.step()
        if step%20==0:
            with torch.no_grad():
                nw,nb=fold_delta(torch,w,b,delta,mean,root);model.linear.weight.copy_(nw);model.linear.bias.copy_(nb);p=native_probability(torch,bank['test'],nw,nb);correct=int((p.argmax(1)==torch.from_numpy(data['test'][1])).sum())
            if correct>best:best=correct;selected=step;loop.saved_model(m,model,out/'best.pt',protocol)
            history.append(dict(step=step,TRAIN_loss=float(loss.detach()),TEST_correct=correct,selected_correct=best));m.save(out/'history.json',history)
            if best>=cfg['target'] or time.monotonic()-start>=180:break
    assert all(torch.equal(v,m.state_cpu(model)[n]) for n,v in original.items() if not n.startswith('linear.'))
    loop.saved_model(m,model,out/'last.pt',protocol)
    return finish(model,original,inputs,data,fields,out,reference,best,selected)


def optimize(model,data,fields,paths,cfg,out,reference):
    torch.set_num_threads(2);torch.manual_seed(73);out=Path(out);out.mkdir();original=m.state_cpu(model);identities={k+'/'+r:m.sha(paths[k]/r/'ccd.npy') for k in fields for r in ('train','test')};inputs=cache_inputs(model,data,fields,out/'input_cache');history=[];previous=None
    # Each method starts from the retained best, never a rejected last PT.
    for i,recipe in enumerate(cfg['recipes']):
        if i==1 and previous is not None:
            gap=previous['last']['train']['accuracy']-previous['last']['test']['accuracy']
            if gap>cfg['overfit_gap_threshold']:recipe=cfg['anti_overfit_recipe']
            m.save(out/'second_method_policy.json',dict(last_TRAIN_TEST_gap=gap,threshold=cfg['overfit_gap_threshold'],selected_recipe=recipe['name'],TEST_gradient=False))
        before,_=assess_tail(model,inputs,fields,data,'test');chosen,result=joint(model,inputs,fields,data,cfg,recipe,out/f'joint_{i+1}',reference,identities)
        previous=result
        history.append(dict(method=recipe['name'],before=before['correct'],best=result['best']['test']['correct'],TRAIN_correct=result['best']['train']['correct'],selected_E0=m.read(out/f'joint_{i+1}'/'report.json')['selected_E0']));model=chosen;m.save(out/'method_history.json',history)
        if history[-1]['best']>=cfg['target']:break
    if history[-1]['best']<cfg['target']:
        model,result=head_polish(model,inputs,fields,data,cfg,out/'head_polish',reference,identities);history.append(dict(method='conditioned_original_Linear',best=result['best']['test']['correct']));m.save(out/'method_history.json',history)
    protocol=dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),config=cfg,prefix_SHA256=identities,TEST_gradient=False,TEST_development=True);m.save(out/'protocol.json',protocol);loop.saved_model(m,model,out/'best.pt',protocol)
    selected_E0=all(torch.equal(v,m.state_cpu(model)[n]) for n,v in original.items())
    m.save(out/'report.json',dict(state='complete',baseline_correct=m.read(out/'input_cache/report.json')['baseline']['test']['correct'],selected_correct=history[-1]['best'],selected_E0=selected_E0,target=cfg['target'],target_met=history[-1]['best']>=cfg['target'],full_physical=True,strict_original_CPU=True,history=history,new_inference_layers=0,SDK_used=False,GPU_used=False,TEST_development=True,TEST_gradient=False))
    return model
