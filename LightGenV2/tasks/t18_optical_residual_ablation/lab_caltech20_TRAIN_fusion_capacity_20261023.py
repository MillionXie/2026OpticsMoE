"""One TRAIN-only fusion/readout capacity check of existing parameters.

Does not use the legacy DVP captures or its numeric conclusion. Fresh features
are required at BOTH existing alpha settings and this exact current parent.
No inference transform, measured-mask update, SDK call or TEST fit is added.
"""
import math
import lab_caltech20_tucam_layerwise_20261023 as base
import lab_caltech20_tucam_TRAIN_linear_20261023 as linear


def capacity_better(parent,candidate):
    assert all(0<=row['TRAIN_capacity_correct']<=1600 and math.isfinite(row['TRAIN_capacity_NLL']) for row in (parent,candidate))
    return candidate['TRAIN_capacity_correct']>=parent['TRAIN_capacity_correct'] and candidate['TRAIN_capacity_NLL']<parent['TRAIN_capacity_NLL']-1e-4


def probe(model,data,dm,fields,paths,parent,cfg,counts,out):
    np,torch,m,loop=base.numerical_modules()
    from torch.nn import functional as F
    assert cfg==dict(stage=2,samples=1600,proposed_alpha=.4001001,estimators_per_setting=1)
    assert list(fields)==list(base.STAGES[:4]) and len(counts)==20 and sum(counts)==3141
    assert not out.exists() and not m.LOCK.exists();out.mkdir();model.cpu().eval().requires_grad_(False)
    original=m.state_cpu(model);allowed={'linear.weight','linear.bias','fusion_logits.2'};lineage=base.identities(m,paths)
    origin=float(model.fusion_logits[2]);labels=data['train'][1];ids=data['train'][2]
    assert np.bincount(labels,minlength=20).tolist()==[80]*20
    rows=[];proposal=None
    for name,value in (('parent',origin),('rebalanced',cfg['proposed_alpha'])):
        model.load_state_dict(original)
        with torch.no_grad():model.fusion_logits[2].fill_(value)
        parts=[]
        with torch.inference_mode():
            for begin in range(0,1600,8):
                ix=np.arange(begin,begin+8);model(loop.put_fields(model,data,fields,'train',ix,'cpu'));parts.append(model.readout['features'].numpy().copy())
                if begin%160==0:m.save(out/'progress.json',dict(state='TRAIN_capacity_features',setting=name,TRAIN_consumed=begin+8,TEST_used=False))
        x=np.concatenate(parts);assert x.shape==(1600,256) and np.isfinite(x).all()
        cache=out/('TRAIN_'+name+'_features.npz');np.savez_compressed(cache,features=x,labels=labels,ids=ids)
        if name=='parent':
            with np.load(parent.parent/'best_train_predictions.npz',allow_pickle=False) as z:
                assert np.array_equal(z['ids'],ids) and np.array_equal(z['labels'],labels)
                with torch.inference_mode():error=float(np.max(np.abs(model.linear(torch.from_numpy(x)).softmax(1).numpy()-z['probabilities'])))
            assert error<=3e-6
        weight,bias,statistics=linear.TRAIN_linear_fit(x,labels,counts)
        w=torch.from_numpy(weight.astype(np.float32));b=torch.from_numpy(bias.astype(np.float32))
        with torch.inference_mode():
            logits=F.linear(torch.from_numpy(x),w,b);correct=int((logits.argmax(1).numpy()==labels).sum());nll=float(F.cross_entropy(logits,torch.from_numpy(labels.astype(np.int64))))
        rows.append(dict(setting=name,alpha=value,TRAIN_capacity_correct=correct,TRAIN_capacity_NLL=nll,TRAIN_features_SHA256=m.sha(cache),estimator=statistics))
        if name=='rebalanced':proposal=(w,b,value)
    accepted=capacity_better(*rows);model.load_state_dict(original)
    if accepted:
        with torch.no_grad():model.linear.weight.copy_(proposal[0]);model.linear.bias.copy_(proposal[1]);model.fusion_logits[2].fill_(proposal[2])
    loop.validate_state(m,model,original,allowed);assert base.identities(m,paths)==lineage
    report=dict(passed=True,accepted=accepted,rows=rows,parent_PT=str(parent),parent_PT_SHA256=m.sha(parent),data_SHA256=dm['sha256'],prefix_SHA256=lineage,
        source_SHA256=m.sha(__file__),linear_engine_SHA256=m.sha(linear.__file__),TRAIN=1600,VALID=0,TEST_used_for_capacity=False,TEST_gradient=False,decision_data='TRAIN_only',
        allowed_initialization_keys=sorted(allowed),all_other_parameters_bitwise_frozen=True,measured_masks_and_input_ancestors_frozen=True,new_CCD=0,new_inference_layers=0,
        physical_calibration=False,capacity_not_generalization=True,all_nonhead_frozen=not accepted,native_parent_TRAIN_cache_max_probability_error=error)
    m.save(out/'report.json',report);return report
