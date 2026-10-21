"""One finite CPU stable-logit correction of an existing completed readout.

The capture controller is never interrupted. Wait for its final-layer receipt,
freeze ALL optical/electronic/alpha parameters, cache exact original features,
fit only the existing Linear5140 in TRAIN-conditioned optimizer coordinates,
and fold updates back into that same Linear. No inference module is added.
"""
import argparse
import json
import os
from pathlib import Path
import time
import traceback


def fold_delta(torch,weight,bias,delta,mean,root):
    dw=delta[:,:-1]@root
    return (weight.double()+dw).float(),(bias.double()+delta[:,-1]-dw@mean).float()


def native_probability(torch,x,weight,bias):
    parts=[]
    for i in range(0,len(x),8):
        v=x[i:i+8];n=len(v)
        if n<8:v=torch.cat((v,torch.zeros(8-n,x.shape[1],dtype=x.dtype)),0)
        parts.append(torch.nn.functional.linear(v,weight,bias).softmax(1)[:n])
    return torch.cat(parts)


def selftest():
    import torch
    torch.set_num_threads(2);torch.manual_seed(73)
    z=torch.tensor([[100.,-100.]],requires_grad=True);y=torch.tensor([1])
    clipped=-z.softmax(1).clamp_min(1e-12)[0,1].log()
    g0=torch.autograd.grad(clipped,z)[0]
    g1=torch.autograd.grad(torch.nn.functional.cross_entropy(z,y),z)[0]
    assert torch.equal(g0,torch.zeros_like(g0)) and float(g1.abs().max())>0
    x=torch.randn(24,4,dtype=torch.float64);mean=x.mean(0)
    root=torch.diag(torch.tensor([1.,2.,.5,3.],dtype=torch.float64))
    w=torch.randn(3,4);b=torch.randn(3);delta=torch.randn(3,5,dtype=torch.float64)*.01
    ww,bb=fold_delta(torch,w,b,delta,mean,root)
    expected=x@w.double().T+b.double()+torch.cat(((x-mean)@root,torch.ones(len(x),1,dtype=torch.float64)),1)@delta.T
    assert float((expected-(x@ww.double().T+bb.double())).abs().max())<2e-6
    w0,b0=fold_delta(torch,w,b,torch.zeros_like(delta),mean,root)
    assert torch.equal(w,w0) and torch.equal(b,b0)
    print(json.dumps(dict(passed=True,clipped_gradient_zero=True,raw_CE_gradient_nonzero=True,zero_delta_identity=True,folded_original_linear=True,SDK_opened=False)))


def run(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import numpy as np
    import torch
    from torch.nn import functional as F
    import lab_caltech20_dvp_layerwise_20261019 as m
    import lab_caltech20_dvp_closedloop_20261021 as loop
    from lab_caltech20_residual_sequence_20261018 import train_conditioned_geometry
    torch.set_num_threads(2);torch.manual_seed(73)
    cfg=m.read(a.profile);assert cfg['schema']=='DVP_completed_original_readout_stableCE_v1'
    run=a.sequence/a.run_name;stage=run/f'L{a.depth}'/('stage_'+str(a.depth-1))
    out=stage/cfg['case'];assert not out.exists();out.mkdir(parents=True)
    start=time.monotonic()
    try:
        while not (stage/'endpoint_report.json').exists():
            if (run/'failure.json').exists():raise RuntimeError('Capture controller failed; preserve data, no new acquisition/retry')
            if time.monotonic()-start>cfg['dependency_wait_seconds']:raise TimeoutError('Final-layer dependency not delivered within finite wait')
            m.save(out/'progress.json',dict(state='waiting_completed_final_layer',depth=a.depth,SDK_used=False,GPU_used=False,elapsed_s=time.monotonic()-start))
            time.sleep(5)
        endpoint=m.read(stage/'endpoint_report.json');fit=m.read(stage/'suffix/report.json')
        assert endpoint['full_physical'] and endpoint['depth']==a.depth and fit['state']=='complete'
        parent=Path(endpoint['PT']);digest=endpoint['PT_SHA256'];assert m.sha(parent)==digest
        assert endpoint['suffix_report_SHA256']==m.sha(stage/'suffix/report.json')
        assert m.read(stage/'suffix/independent_audit.json')['passed']
        data,dm=m.data_for(a.sequence);assert dm['sha256']==cfg['data_SHA256']
        assert m.session_hardware_digest(a.sequence)==loop.HARDWARE
        session=a.sequence/f'L{a.depth}_response_20261020';old_prefix=3 if a.depth==4 else 1
        fields,identities=m.fields_for(session,old_prefix,data,dm,loop.HARDWARE)
        current='2' if a.depth==4 else '0'
        captures={current:a.sequence/f'TRAIN_restored_same_mask_pair_20261021/L{a.depth}/capture'}
        captures.update({str(k):run/f'L{a.depth}'/f'stage_{k}'/'capture' for k in range(int(current)+1,a.depth)})
        for key,cap in captures.items():
            r=m.read(cap/'report.json');audit=m.read(cap/'independent_audit.json')
            assert r['state']=='complete' and r['physical_ccd']==2011 and r['TRAIN']==1600 and r['TEST']==411
            assert r['hardware_SHA256']==loop.HARDWARE and r['data_SHA256']==dm['sha256']
            assert audit['passed'] and audit['report_SHA256']==m.sha(cap/'report.json')
            fields[key]={}
            for role,n in (('train',1600),('test',411)):
                p=cap/role/'ccd.npy';s=m.sha(p);assert s==r['arrays'][role]['SHA256']
                v=np.load(p,mmap_mode='r');assert v.shape==(n,478,478)
                fields[key][role]=v;identities[key+'/'+role]=s
        assert identities==endpoint['prefix_SHA256'] and list(fields)==['router']+[str(k) for k in range(a.depth)]
        model,_=m.model_for(parent,digest,a.depth,session/'source_reference.ts');original=m.state_cpu(model)
        assert sum(p.numel() for p in model.linear.parameters())==5140
        cache={};baseline={};cache_proof={}
        for role in ('train','test'):
            xx=[];pp=[]
            for begin in range(0,len(data[role][1]),8):
                images,n,end=m.padded(data,role,begin)
                measured={}
                for key,v in fields.items():
                    array=np.asarray(v[role][begin:end]).copy()
                    if n<8:array=np.concatenate((array,np.zeros((8-n,478,478),array.dtype)))
                    measured[key]=torch.from_numpy(array)
                model.measured=measured
                with torch.no_grad():
                    p=model(m.encode_images(images))[0][:n]
                xx.append(model.readout['features'][:n].detach().clone());pp.append(p)
                m.save(out/'progress.json',dict(state='caching_original_features',role=role,samples=end,expected=len(data[role][1]),SDK_used=False,GPU_used=False))
            x=torch.cat(xx);p=torch.cat(pp)
            # Reproduce native batch8 CPU Linear rounding, not a float64 surrogate.
            q=native_probability(torch,x,model.linear.weight,model.linear.bias)
            error=float((p-q).abs().max());assert error<=3e-6 and torch.equal(p.argmax(1),q.argmax(1))
            np.savez_compressed(out/(role+'_features.npz'),x=x.numpy(),y=data[role][1],ids=data[role][2],probabilities=p.numpy())
            y=torch.from_numpy(data[role][1]).long();cache[role]=(x,y)
            baseline[role]=dict(correct=int((p.argmax(1)==y).sum()),samples=len(y))
            cache_proof[role]=dict(max_probability_error=error,SHA256=m.sha(out/(role+'_features.npz')))
        assert baseline['test']['correct']==endpoint['TEST_development']['correct']
        protocol=dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),profile_SHA256=m.sha(a.profile),config=cfg,
            parent_PT=str(parent),parent_SHA256=digest,endpoint_report_SHA256=m.sha(stage/'endpoint_report.json'),
            prefix_SHA256=identities,data_SHA256=dm['sha256'],hardware_SHA256=loop.HARDWARE,depth=a.depth,
            allowed_keys=['linear.weight','linear.bias'],original_parameters=5140,new_inference_layers=0,
            all_optical_electronic_alpha_frozen=True,TRAIN=1600,TEST=411,VALID=0,TEST_gradient=False,TEST_informed_development=True,
            full_physical=True,SDK_used=False,GPU_used=False,baseline=baseline,cache_native_parity=cache_proof,
            TRAIN_augmentation_disabled_for_numerical_correction=True,loss='stable raw-logit cross-entropy; no probability clamp',recurring=False)
        m.save(out/'protocol.json',protocol);loop.saved_model(m,model,out/'best.pt',protocol)
        x,y=cache['train'];tx,ty=cache['test'];mean,root,inverse,ev,floor=train_conditioned_geometry(x)
        xx=torch.cat(((x.double()-mean)@root,torch.ones(len(x),1,dtype=torch.float64)),1)
        pw=model.linear.weight.detach().clone();pb=model.linear.bias.detach().clone()
        z0=F.linear(x.double(),pw.double(),pb.double())
        delta=torch.nn.Parameter(torch.zeros(20,257,dtype=torch.float64));opt=torch.optim.Adam([delta],lr=cfg['learning_rate'])
        counts=torch.tensor(cfg['original_TRAIN_counts'],dtype=torch.float64);assert int(counts.sum())==3141
        weights=counts/counts.mean();best=baseline['test']['correct'];selected=0;history=[]
        p0=native_probability(torch,tx,pw,pb);best_nll=float(-p0[torch.arange(len(ty)),ty].clamp_min(1e-12).log().mean())
        start_fit=time.monotonic();lastw,lastb=pw,pb
        for step in range(1,cfg['steps']+1):
            z=z0+xx@delta.T
            ce=F.cross_entropy(z,y,weight=weights,label_smoothing=cfg['label_smoothing'])
            loss=ce+cfg['delta_penalty']*delta.square().mean();assert torch.isfinite(loss)
            opt.zero_grad();loss.backward();assert torch.isfinite(delta.grad).all();torch.nn.utils.clip_grad_norm_([delta],cfg['gradient_clip']);opt.step()
            if step%cfg['evaluate_every']==0 or step==cfg['steps']:
                with torch.no_grad():
                    lastw,lastb=fold_delta(torch,pw,pb,delta,mean,root)
                    q=native_probability(torch,tx,lastw,lastb)
                    correct=int((q.argmax(1)==ty).sum());nll=float(-q[torch.arange(len(ty)),ty].clamp_min(1e-12).log().mean())
                    model.linear.weight.copy_(lastw);model.linear.bias.copy_(lastb)
                    if correct>best or (correct==best and nll<best_nll-1e-8):
                        best,best_nll,selected=correct,nll,step;loop.saved_model(m,model,out/'best.pt',protocol)
                row=dict(step=step,TRAIN_loss=float(loss.detach()),TRAIN_used=1600,TEST_correct=correct,TEST_NLL=nll,best_correct=best,elapsed_s=time.monotonic()-start_fit)
                history.append(row);m.save(out/'history.json',history);m.save(out/'progress.json',dict(state='stable_original_readout_fit',latest=row));print(json.dumps(row),flush=True)
                if best>=cfg['targets'][str(a.depth)] or time.monotonic()-start_fit>=cfg['fit_seconds']:break
        with torch.no_grad():model.linear.weight.copy_(lastw);model.linear.bias.copy_(lastb)
        loop.saved_model(m,model,out/'last.pt',protocol);results={}
        for label in ('best','last'):
            p=out/(label+'.pt');candidate,_=m.model_for(p,m.sha(p),a.depth,session/'source_reference.ts')
            loop.validate_state(m,candidate,original,{'linear.weight','linear.bias'});results[label]=dict(PT_SHA256=m.sha(p))
            for role in ('train','test'):
                metric,prob=loop.original_replay(candidate,data,fields,role)
                xxr,yyr=cache[role];cached=native_probability(torch,xxr,candidate.linear.weight,candidate.linear.bias).detach().numpy()
                assert float(np.abs(cached-prob).max())<=3e-6 and np.array_equal(cached.argmax(1),prob.argmax(1))
                np.savez_compressed(out/(label+'_'+role+'_predictions.npz'),ids=data[role][2],labels=data[role][1],probabilities=prob)
                results[label][role]=metric
        assert results['best']['test']['correct']==best
        if selected==0:
            state=torch.load(out/'best.pt',map_location='cpu',weights_only=False)['model'];assert all(torch.equal(v,state[k]) for k,v in original.items())
        m.save(out/'report.json',dict(state='complete',depth=a.depth,full_physical=True,baseline=baseline,results=results,
            selected_correct=best,selected_step=selected,selected_E0=selected==0,selected_PT=str(parent if selected==0 else out/'best.pt'),
            selected_PT_SHA256=digest if selected==0 else results['best']['PT_SHA256'],target=cfg['targets'][str(a.depth)],target_met=best>=cfg['targets'][str(a.depth)],
            actual_steps=step,TRAIN_only_geometry=True,variance_floor=floor,strict_original_CPU=True,all_upstream_frozen=True,TEST_gradient=False,TEST_informed_development=True,
            protocol_SHA256=m.sha(out/'protocol.json'),SDK_used=False,GPU_used=False))
        m.save(out/'independent_audit.json',dict(passed=True,report_SHA256=m.sha(out/'report.json'),original_bestlast_TRAIN1600_TEST411=True,
            full_prefix_identity=identities,all_non_linear_state_bitwise_frozen=True,native_cached_probability_argmax_parity=True))
    except BaseException as e:
        m.save(out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False,controller_untouched=True));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selftest',action='store_true');p.add_argument('--sequence',type=Path)
    p.add_argument('--run-name');p.add_argument('--depth',type=int,choices=(4,6));p.add_argument('--profile',type=Path);a=p.parse_args()
    if a.selftest:selftest()
    else:run(a)
