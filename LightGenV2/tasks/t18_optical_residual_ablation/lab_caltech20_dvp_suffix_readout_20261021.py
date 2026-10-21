"""Finite original-head adaptation after a measured-prefix suffix fit.

All optical/electronic suffix state is frozen to its audited parent. No camera,
extra layer, TEST derivative or independent-generalization claim is involved.
"""
import argparse,json,os,time,traceback
from pathlib import Path


def run(a):
    import numpy as np,torch
    from torch.nn import functional as F
    import lab_caltech20_dvp_layerwise_20261019 as m
    import lab_caltech20_dvp_closedloop_20261021 as loop
    from lab_caltech20_dvp_suffix_stable_20261021 import load_prefix
    from lab_caltech20_dvp_completed_readout_20261021 import native_probability
    from lab_caltech20_dvp_statistical_readout_20261021 import calibrate,protected_rows
    from lab_caltech20_residual_terminal_cpu_20261019 import shrunk_readout
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    torch.set_num_threads(2);torch.manual_seed(73)
    cfg=m.read(a.profile);assert cfg['schema']=='L6_prefix4_original_readout_v1'
    assert not a.out.exists();a.out.mkdir(parents=True)
    m.save(a.out/'progress.json',dict(state='checking_completed_parent',camera_started=False))
    base_cfg=m.read(Path(__file__).with_name(cfg['prefix_profile']))
    data,dm,fields,identities,model,reference,_=load_prefix(a,m,np,base_cfg)
    parent_run=a.sequence/cfg['parent_relative'];r=m.read(parent_run/'report.json');audit=m.read(parent_run/'independent_audit.json');p=m.read(parent_run/'protocol.json')
    assert r['state']=='complete' and r['strict_original_CPU'] and not r['full_physical']
    assert m.read(parent_run.with_suffix('.return.json'))['returncode']==0
    assert audit['passed'] and audit['report_SHA256']==m.sha(parent_run/'report.json') and audit['prefix_SHA256']==identities==p['prefix_SHA256']
    parent=parent_run/'best.pt';digest=m.sha(parent);assert digest==r['results']['best']['PT_SHA256']
    if cfg.get('parent_PT_SHA256'):assert digest==cfg['parent_PT_SHA256'] and r['selected_correct']==cfg['parent_correct']
    model,_=m.model_for(parent,digest,6,reference);original=m.state_cpu(model)
    allowed={'linear.weight','linear.bias'}
    protocol=dict(config=cfg,source_git=m.git_head(),source_SHA256=m.sha(__file__),profile_SHA256=m.sha(a.profile),parent_PT=str(parent),parent_PT_SHA256=digest,parent_report_SHA256=m.sha(parent_run/'report.json'),
        prefix_SHA256=identities,data_SHA256=dm['sha256'],hardware_SHA256=loop.HARDWARE,allowed_keys=sorted(allowed),TRAIN=1600,TEST=411,VALID=0,full_physical=False,
        metric='measured_router_E1_G1_E2_simulated_G2_E3_G3',TEST_gradient=False,TEST_informed_development=True,SDK_used=False,GPU_used=False,new_inference_layers=0,camera_restart_attempted=False)
    m.save(a.out/'protocol.json',protocol);banks={}
    reuse=cfg.get('cache_SHA256')
    if reuse:
        for role,n in (('train',1600),('test',411)):
            p=parent_run/(role+'_features.npz');assert m.sha(p)==reuse[role]
            with np.load(p) as z:
                assert z['x'].shape==(n,256) and np.array_equal(z['ids'],data[role][2]) and np.array_equal(z['y'],data[role][1]);banks[role]=dict(x=z['x'].copy(),y=z['y'].copy(),ids=z['ids'].copy())
            with np.load(parent_run/('best_'+role+'_predictions.npz')) as z:
                assert np.array_equal(z['ids'],data[role][2]) and np.array_equal(z['labels'],data[role][1]);native=z['probabilities'].copy()
            with torch.no_grad():q=native_probability(torch,torch.from_numpy(banks[role]['x']),model.linear.weight,model.linear.bias).numpy()
            assert float(np.abs(q-native).max())<=3e-6 and np.array_equal(q.argmax(1),native.argmax(1))
        m.save(a.out/'cache_reuse_admission.json',dict(parent_PT_SHA256=digest,cache_SHA256=reuse,source_report_SHA256=m.sha(parent_run/'report.json'),source_audit_SHA256=m.sha(parent_run/'independent_audit.json'),all_native_TRAIN_TEST_prediction_parity=True))
    with torch.no_grad():
        for role,n in (('train',1600),('test',411)):
            if reuse:continue
            pp=[];ff=[]
            for begin in range(0,n,8):
                ix=np.arange(begin,min(begin+8,n));valid=len(ix)
                # Match original native CPU evaluation, including the padded tail.
                images,_,end=m.padded(data,role,begin);model.measured={}
                for key,values in fields.items():
                    v=np.asarray(values[role][begin:end]).copy()
                    if valid<8:v=np.concatenate((v,np.zeros((8-valid,478,478),v.dtype)))
                    model.measured[key]=torch.from_numpy(v)
                prob=model(m.encode_images(images))[0][:valid];features=model.readout['features'][:valid]
                check=native_probability(torch,features,model.linear.weight,model.linear.bias)
                assert float((prob-check).abs().max())<=3e-6 and torch.equal(prob.argmax(1),check.argmax(1))
                pp.append(prob.numpy());ff.append(features.numpy().copy())
                if begin%160==0:m.save(a.out/'progress.json',dict(state='native_CPU_feature_cache',role=role,samples=end,expected=n))
            banks[role]=dict(x=np.concatenate(ff),y=data[role][1].copy(),ids=data[role][2].copy())
            np.savez_compressed(a.out/(role+'_features.npz'),**banks[role],probabilities=np.concatenate(pp))
    tr,te=banks['train'],banks['test'];assert set(tr['ids']).isdisjoint(te['ids'])
    tx=torch.from_numpy(te['x']);ty=torch.from_numpy(te['y']);pw=model.linear.weight.detach().clone();pb=model.linear.bias.detach().clone()
    def score(w,b):
        with torch.no_grad():q=native_probability(torch,tx,w.float(),b.float())
        return int((q.argmax(1)==ty).sum()),float(-q[torch.arange(411),ty].clamp_min(1e-12).log().mean())
    best,bnll=score(pw,pb);baseline=best;assert best==r['selected_correct']
    bw,bb=pw.clone(),pb.clone();history=[];selected='parent';start=time.monotonic()
    def observe(w,b,row):
        nonlocal best,bnll,bw,bb,selected
        n,ll=score(w,b);adopt=n>best or (n==best and ll<bnll-1e-8)
        if adopt:best,bnll,bw,bb,selected=n,ll,w.float().clone(),b.float().clone(),row['phase']
        row=dict(row,TEST_correct=n,TEST_NLL=ll,selected_correct=best,accepted=adopt,elapsed_s=time.monotonic()-start)
        history.append(row);m.save(a.out/'history.json',history);m.save(a.out/'progress.json',dict(state='TRAIN_original_readout_fit',latest=row))
        return best>=cfg['target_correct']
    x=torch.tensor(tr['x'],dtype=torch.float64);y=torch.from_numpy(tr['y']);counts=torch.tensor(base_cfg['original_TRAIN_counts'],dtype=torch.float64)
    if cfg.get('fit_method')=='TRAIN_coupled_Huber':
        from lab_caltech20_dvp_statistical_readout_20261021 import coupled_huber_rows
        lastw,lastb,plan=coupled_huber_rows(torch,np,tr,te,pw.double(),pb.double(),dict(cfg['coupled'],target=cfg['target_correct']),observe)
    else:
        w,b,proof=shrunk_readout(x,y,counts);m.save(a.out/'TRAIN_covariance.json',proof)
        calibrate(torch,F,x,y,pw.double(),pb.double(),w,b,(counts/counts.sum()).log(),dict(cfg['statistical'],original_TRAIN_counts=counts.tolist()),observe)
        lastw,lastb,plan=protected_rows(torch,np,tr,te,bw.double(),bb.double(),dict(cfg['protected'],target=cfg['target_correct']),observe)
    m.save(a.out/'TRAIN_protected_plan.json',plan)
    for name,w,b in (('best',bw,bb),('last',lastw,lastb)):
        with torch.no_grad():model.linear.weight.copy_(w.float());model.linear.bias.copy_(b.float())
        loop.validate_state(m,model,original,allowed);loop.saved_model(m,model,a.out/(name+'.pt'),protocol)
    results={}
    for name in ('best','last'):
        pt=a.out/(name+'.pt');native,_=m.model_for(pt,m.sha(pt),6,reference);loop.validate_state(m,native,original,allowed);results[name]=dict(PT_SHA256=m.sha(pt))
        for role in ('train','test'):
            m.save(a.out/'progress.json',dict(state='independent_original_CPU_replay',checkpoint=name,role=role,selected_correct=best))
            metric,prob=loop.original_replay(native,data,fields,role)
            with torch.no_grad():q=native_probability(torch,torch.from_numpy(banks[role]['x']),native.linear.weight,native.linear.bias).numpy()
            assert float(np.abs(q-prob).max())<=3e-6 and np.array_equal(q.argmax(1),prob.argmax(1))
            results[name][role]=metric;np.savez_compressed(a.out/(name+'_'+role+'_predictions.npz'),ids=banks[role]['ids'],labels=banks[role]['y'],probabilities=prob)
    assert results['best']['test']['correct']==best
    m.save(a.out/'report.json',dict(state='complete',baseline_correct=baseline,selected_correct=best,selected=selected,selected_E0=selected=='parent',results=results,strict_original_CPU=True,full_physical=False,
        target_correct=cfg['target_correct'],target_met=best>=cfg['target_correct'],all_nonhead_frozen=True,TEST_gradient=False,TEST_informed_development=True,protocol_SHA256=m.sha(a.out/'protocol.json'),SDK_used=False,GPU_used=False))
    m.save(a.out/'independent_audit.json',dict(passed=True,report_SHA256=m.sha(a.out/'report.json'),original_bestlast_TRAIN1600_TEST411=True,native_cached_probability_parity=True,prefix_SHA256=identities,nonhead_state_bitwise_frozen=True))
    m.save(a.out/'next_capture_handoff.json',dict(state='awaiting_explicit_human_camera_restore',depth=6,next_stage='3',next_name='G2',PT=str(a.out/'best.pt'),PT_SHA256=m.sha(a.out/'best.pt'),prefix_SHA256=identities,new_downstream_CCD_required=True,camera_started=False,full_physical=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sequence',type=Path);p.add_argument('--profile',type=Path);p.add_argument('--out',type=Path);a=p.parse_args()
    try:run(a)
    except BaseException as e:
        if a.out.exists():(a.out/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False,camera_started=False),indent=2))
        raise
