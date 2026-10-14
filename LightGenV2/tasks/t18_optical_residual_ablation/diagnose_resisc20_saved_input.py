"""Bounded CPU input-detail and saved-confusion diagnosis; no model or test."""
import argparse,csv,hashlib,json,os,subprocess
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import resisc20 as s
from audit_caltech_repair234 import exited
from caltech_deploy234_model import rms_encode

HERE,ROOT=s.HERE,s.ROOT
BASE=HERE/'runs/simulation/resisc20_s17_mainL4_rho0rho03_fullres_continue120_micro8_gpu256_20261014'
SAMPLE_PER_CLASS=16

def describe(values):
    a=np.asarray(values,float)
    return dict(mean=float(a.mean()),median=float(np.median(a)),p10=float(np.quantile(a,.1)),p90=float(np.quantile(a,.9)))

def row_diagnostics(metric,classes):
    rows=[]
    cm=np.asarray(metric['confusion_matrix']);support=np.asarray(metric['support'])
    for k,name in enumerate(classes):
        wrong=cm[k].copy();wrong[k]=-1;j=int(wrong.argmax())
        rows.append(dict(class_name=name,support=int(support[k]),recall=metric['recall'][k],
            most_common_wrong_class=classes[j],most_common_wrong_count=int(cm[k,j]),
            true_class_count=int(cm[k,k]),diagonal_is_unique_row_max=bool(cm[k,k]>wrong.max())))
    return rows

def scores(path,fit,role,ids):
    rows=list(csv.DictReader(path.open(encoding='utf-8')))
    assert [v['sample_id'] for v in rows]==ids
    y=np.asarray([int(v['label_true']) for v in rows]);p=np.asarray([[float(v['score'+str(k)]) for k in range(20)] for v in rows])
    assert np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(1),1,atol=1e-6)
    pred=p.argmax(1);assert np.array_equal(pred,[int(v['label_pred']) for v in rows])
    cm=np.zeros((20,20),np.int64);np.add.at(cm,(y,pred),1)
    assert cm.tolist()==fit['metrics'][role]['confusion_matrix']
    return dict(labels=y,pred=pred,sha256=s.sha(path))

def run(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not a.out.exists()
    torch.set_num_threads(4)
    meta=s.read(BASE/'metadata.json');state=s.read(BASE/'status.json')
    assert exited(meta['pid']) and state['active']==state['pending']==0 and not state['failures']
    assert len(state['completed'])==2 and all(exited(j['pid']) and j['exit_code']==0 for j in state['completed'])
    assert meta['git_commit']=='bbdaf0a852aaf9001b0c82d74dcdd592ea9865ad' and len(meta['sources'])==144
    s.engine.verify_sources(meta['sources'],meta['git_commit'])
    audit=s.read(BASE/'cpu_report/audit.json');receipt=s.read(BASE/'cpu_saved_audit_dispatch_completed_20261014.json')
    assert audit['passed'] and receipt['audit_sha256']==s.sha(BASE/'cpu_report/audit.json') and audit['test_evaluations']==0
    source=s.sha(__file__);commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert hashlib.sha256(subprocess.check_output(['git','show',commit+':'+Path(__file__).relative_to(ROOT).as_posix()],cwd=ROOT)).hexdigest()==source
    data=Path(s.read(s.PROFILE)['data']);manifest=s.manifest(data);classes=manifest['classes']
    # NPZ materializes the train arrays only. No val/test image or test label access.
    with np.load(data,allow_pickle=False) as z:
        images=z['train_images'];labels=z['train_labels'];ids=z['train_ids']
    assert images.shape==(11198,150,150,3) and labels.shape==(11198,)
    assert ids.tolist()==manifest['split_ids']['train']
    rng=np.random.default_rng(20261014)
    index=np.concatenate([rng.choice(np.flatnonzero(labels==k),SAMPLE_PER_CLASS,replace=False) for k in range(20)])
    sample_ids=ids[index].tolist();stats=[];independent_error=0.;rms_errors=[];zero_fraction=[];quant_error=[]
    with torch.no_grad():
        for start in range(0,len(index),16):
            ii=index[start:start+16];raw=torch.from_numpy(images[ii].transpose(0,3,1,2).copy())
            x=raw.float()/255
            x100=F.interpolate(x,(100,100),mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
            x50=F.interpolate(x100,(50,50),mode='bilinear',align_corners=False,antialias=True)
            mosaic=torch.cat([torch.cat([x50[:,0:1],x50[:,1:2]],3),torch.cat([x50[:,2:3],x50.mean(1,keepdim=True)],3)],2)
            expected=rms_encode(F.interpolate(mosaic,(224,224),mode='bilinear',align_corners=False)[:,0])
            actual=s.r.input_amplitude(raw,None)
            independent_error=max(independent_error,float((actual-expected).abs().max()))
            restored=F.interpolate(x50,(150,150),mode='bilinear',align_corners=False,antialias=True)
            mse=(restored-x).square().mean((1,2,3))
            variance=(x-x.mean((2,3),keepdim=True)).square().mean((1,2,3)).clamp_min(1e-12)
            def gradient_energy(v):
                return .5*((v[:,:,1:]-v[:,:,:-1]).square().mean((1,2,3))+(v[:,:,:,1:]-v[:,:,:,:-1]).square().mean((1,2,3)))
            retention=gradient_energy(restored)/gradient_energy(x).clamp_min(1e-12)
            bounded=torch.tanh(actual.abs()/.5);q=torch.round(bounded*255)/255
            for j,n in enumerate(ii):
                stats.append(dict(sample_id=str(ids[n]),class_name=classes[int(labels[n])],rgb150_to_exact50_roundtrip_mse=float(mse[j]),
                    centered_variance_relative_error=float(mse[j]/variance[j]),gradient_energy_retention=float(retention[j])))
            rms_errors.extend((actual.square().mean((-2,-1)).sqrt()-.25).abs().tolist())
            zero_fraction.extend((q==0).float().mean((-2,-1)).tolist());quant_error.extend((q-bounded).abs().mean((-2,-1)).tolist())
    assert independent_error==0 and max(rms_errors)<1e-6
    input_summary={k:describe([v[k] for v in stats]) for k in ('rgb150_to_exact50_roundtrip_mse','centered_variance_relative_error','gradient_energy_retention')}
    class_input={name:{k:describe([v[k] for v in stats if v['class_name']==name]) for k in input_summary} for name in classes}
    predictions={};case_rows={};sha_rows={}
    for key,fit in audit['fits'].items():
        assert fit['epochs_completed']==120 and fit['selected_epoch']==120
        predictions[key]={};case_rows[key]={};sha_rows[key]={}
        for role in ('train','val'):
            result=scores(BASE/key/(role+'_predictions.csv'),fit,role,manifest['split_ids'][role])
            if role=='train':assert np.array_equal(result['labels'],labels)
            predictions[key][role]=result;case_rows[key][role]=row_diagnostics(fit['metrics'][role],classes);sha_rows[key][role]=result['sha256']
    paired={}
    for role in ('train','val'):
        p0,p3=(predictions[key][role] for key in ('L4_rho0','L4_rho03'))
        assert np.array_equal(p0['labels'],p3['labels']);y=p0['labels'];c0=p0['pred']==y;c3=p3['pred']==y
        paired[role]=dict(n=len(y),prediction_disagreements=int((p0['pred']!=p3['pred']).sum()),
            only_rho0_correct=int((c0&~c3).sum()),only_rho03_correct=int((c3&~c0).sum()),
            both_correct=int((c0&c3).sum()),both_wrong=int((~c0&~c3).sum()),
            net_rho03_accuracy_gain_pp=100*float(c3.mean()-c0.mean()))
    value=dict(passed=True,git_commit=commit,diagnostic_source_sha256=source,training_source_commit=meta['git_commit'],
        saved_audit_sha256=receipt['audit_sha256'],sources_current_original_match=True,cache_sha256=manifest['cache_sha256'],
        sample_seed=20261014,sampled_per_class=16,sampled_images=320,train_images_materialized=11198,
        sample_ids=sample_ids,encoder_path='uint8 RGB150 ->bicubicAA100/clamp ->bilinearAA50 ->R/G/B/mean mosaic100 ->bilinear224 ->RMS.25',
        independent_encoder_probability_not_computed=True,independent_encoder_max_abs_error=independent_error,
        rms_max_abs_error=max(rms_errors),input_quantization_zero_fraction=describe(zero_fraction),input_quantization_mean_abs_error=describe(quant_error),
        input_summary=input_summary,per_class_input=class_input,saved_prediction_sha256=sha_rows,saved_class_confusions=case_rows,
        paired_saved_predictions=paired,sample_rows=stats,models_constructed=0,model_forward_calls=0,
        val_image_arrays_read=0,test_arrays_read=0,test_evaluations=0,gpu_devices=0,calibration_fit=0,
        interpretation='Resampling detail loss and saved confusions are descriptive; not classifier-information loss, capacity/optimization causality, optical throughput, or proof a new encoder improves accuracy. No profile/labelmap/selection change.')
    a.out.mkdir(parents=True,exist_ok=False);s.save(a.out/'checks.json',value,exclusive=True)
    return dict(passed=True,input_summary=input_summary,paired_saved_predictions=paired,independent_encoder_max_abs_error=independent_error,output=str(a.out),model_forward_calls=0,test_arrays_read=0,gpu_devices=0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:print(json.dumps(run(a)))
    except BaseException as e:
        if a.out.exists():s.save(a.out/'failure.json',dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
