"""Independent saved-math and complete new TRAIN pixel replay; zero forwards."""
import argparse,hashlib,math,os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import caltech20_alt246_richsplit_viewreg as study
import audit_caltech20_alt246_richsplit as previous

def reference_parameters(n,epoch,batch,cfg):
    generator=np.random.default_rng(np.random.SeedSequence([cfg['seed'],epoch,batch,cfg['view_seed_tag'],3]));result=np.empty((n,7),dtype='<f8')
    for i in range(n):
        size=float(generator.uniform(cfg['crop_area'][0],cfg['crop_area'][1]))*22500;aspect=math.exp(float(generator.uniform(math.log(cfg['crop_aspect'][0]),math.log(cfg['crop_aspect'][1]))));width=int(round(math.sqrt(size*aspect)));height=int(round(math.sqrt(size/aspect)));width=max(1,min(150,width));height=max(1,min(150,height))
        result[i,0]=generator.integers(151-width);result[i,1]=generator.integers(151-height);result[i,2:4]=width,height
        result[i,4]=generator.uniform(*cfg['brightness']);result[i,5]=generator.uniform(*cfg['contrast']);result[i,6]=generator.uniform(*cfg['gamma'])
    return result

def reference_pixels(images,parameters):
    result=np.empty_like(images)
    for i in range(len(images)):
        left,top,width,height=[int(v) for v in parameters[i,:4]];brightness,contrast,gamma=parameters[i,4:];resized=np.array(Image.fromarray(images[i]).crop((left,top,left+width,top+height)).resize((150,150),Image.Resampling.BICUBIC),dtype=np.float64)
        for channel in range(3):
            values=resized[:,:,channel];average=values.mean();adjusted=((values-average)*contrast+average)*brightness;normalized=np.minimum(1,np.maximum(0,adjusted/255));result[i,:,:,channel]=np.rint(255*np.power(normalized,gamma)).astype(np.uint8)
    return result

def view_hashes(images,order,epoch,cfg):
    hashes={k:[] for k in ('view','pixels')}
    for batch,off in enumerate(range(0,len(order),16)):
        idx=order[off:off+16];parameters=reference_parameters(len(idx),epoch,batch,cfg);pixels=reference_pixels(images[idx],parameters)
        hashes['view'].append(hashlib.sha256(parameters.tobytes()).hexdigest());hashes['pixels'].append(hashlib.sha256(pixels.tobytes()).hexdigest())
    return {k:hashlib.sha256(''.join(v).encode()).hexdigest() for k,v in hashes.items()}

def audit(base,out,phase):
    s=study.s;assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not out.exists();meta=s.read(base/'metadata.json');depth=meta['depth'];parent_root=Path(meta['parent_root']);assert base==study.expected_root(depth,parent_root) and meta['sources']==study.sources() and meta['config']==study.config();study.engine.verify_sources(meta['sources'],meta['git_commit']);parent=study.parent_info(parent_root,depth);assert parent==meta['parent'];cfg=study.config();m=study.b.dataset.manifest();torch.set_num_threads(4);checks={};f=s.read(base/'fit/fit_result.json')
    ex=s.read(base/(phase+'_exit.json'));assert ex['returncode']==0 and ex['actual_exit'];proc=Path('/proc',str(ex['pid']),'cmdline');assert not proc.exists() or str(base/phase).encode() not in proc.read_bytes();best=torch.load(f['best_path'],map_location='cpu',weights_only=False);assert f['best_sha256']==s.sha(f['best_path'])
    if phase=='fit':
        metadata=s.read(base/'fit/metadata.json');assert metadata['sources']==meta['sources'] and metadata['config']==cfg and metadata['parent']==parent and metadata['versions']==study.versions()
        assert metadata['parent_last_sha256']==s.sha(parent['last']) and metadata['parent_history_sha256']==s.sha(parent['history_path'])
        old=torch.load(parent['last'],map_location='cpu',weights_only=False);last=torch.load(base/'fit/last.pt',map_location='cpu',weights_only=False);h=s.read(base/'fit/history.json');prefix=s.read(parent['history_path']);start=parent['actual_epochs'];end=start+20
        assert h[:start]==prefix and len(h)==end and f['actual_epochs']==end and f['additional_epochs']==end-100 and f['block_epochs']==20 and f['steps']==last['steps']==end*cfg['steps_per_epoch'] and f['stop_reason']=='registered20_epoch_method_review'
        assert f['fit_seconds']<=cfg['maximum_seconds']+max(q['epoch_seconds'] for q in h[start:])+60
        assert f['last_sha256']==s.sha(base/'fit/last.pt') and f['history_sha256']==s.sha(base/'fit/history.json') and last['epoch']==end and last['config']==cfg and last['sources']==study.sources()
        assert metadata['parent_raw_digest']==s.digest(old['model']) and metadata['parent_ema_digest']==s.digest(old['ema']) and metadata['initial_optimizer_moment_digest']==study.g.optimizer_moment_digest(old['optimizer'])
        model=study.RegularizedMoE(depth);assert best['contract']==model.contract();model.load_state_dict(last['model'],strict=True);model.load_state_dict(best['model'],strict=True)
        optimizer=last['optimizer'];assert len(optimizer['state'])==len(list(model.parameters())) and all(float(v['step'])==last['steps'] for v in optimizer['state'].values()) and [v['weight_decay'] for v in optimizer['param_groups']]==[0,cfg['electronic_weight_decay']]
        dummy=[torch.nn.Parameter(torch.zeros(())),torch.nn.Parameter(torch.zeros(()))];opt=torch.optim.AdamW([dict(params=[dummy[0]],weight_decay=0),dict(params=[dummy[1]],weight_decay=cfg['electronic_weight_decay'])],lr=cfg['lr']);scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=3,threshold=.002,threshold_mode='abs',min_lr=cfg['minimum_lr']);baseline=study.s.read(study.BASE/f'L{depth}_rho03/fit_result.json');scheduler.best=baseline['metrics']['val']['balanced_nll']
        with np.load(cfg['data'],allow_pickle=False) as z:train_images=z['train_images'].copy()
        for row in h[100:]:
            assert np.allclose(row['lr'],[q['lr'] for q in opt.param_groups],atol=1e-12,rtol=0);scheduler.step(row['val']['balanced_nll'])
        for name in ('best','num_bad_epochs','cooldown_counter','last_epoch','_last_lr'):assert np.allclose(last['scheduler'][name],scheduler.state_dict()[name],atol=1e-12,rtol=0),name
        for row in h[start:]:
            epoch=row['epoch'];assert row['adam_steps']==cfg['steps_per_epoch']*epoch and tuple(row[k] for k in ('order_sha256','affine_sha256','mix_sha256'))==previous.plans(epoch,cfg,len(train_images)) and tuple(row[k] for k in ('flip_sha256','mask_sha256'))==previous.mask_plans(epoch,cfg,len(train_images))
            rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],epoch,20261013]));order=rng.permutation(len(train_images)).astype('<i8');hashes=view_hashes(train_images,order,epoch,cfg)
            for kind,value in hashes.items():assert row[kind+'_sha256']==value,(epoch,kind)
            for role in ('train','val'):previous.history_metric(row[role],m['supports'][role])
        assert set(last['previous_final_ema'])==set(last['model'])==set(last['ema'])
        for name,earlier in last['previous_final_ema'].items():
            expected=earlier.numpy().astype(np.float64)*cfg['ema_decay']+last['model'][name].numpy().astype(np.float64)*(1-cfg['ema_decay']);assert np.allclose(expected,last['ema'][name].numpy(),atol=2e-7,rtol=1e-6),name
        selected=min(h,key=lambda q:q['val']['balanced_nll']);assert selected['epoch']==f['selected_epoch']==best['epoch'] and s.digest(best['model'])==selected['ema_digest'] and s.digest(last['model'])==h[-1]['raw_digest'] and s.digest(last['ema'])==h[-1]['ema_digest']
        assert best['sources']==(study.b.sources() if selected['epoch']<=100 else study.sources())
        for role in ('train','val'):
            actual,checks[role]=previous.verify_predictions(Path(f['selected_'+role+'_path']),best['model'],role,m);previous.agree(actual,f['metrics'][role]);previous.agree(actual,selected[role]);previous.agree(actual,best['metrics'][role])
        gate=s.read(base/'fit/numeric_gate.json');assert gate['passed'] and gate['initial_digest']==s.digest(old['model']) and gate['optimizer_steps']==0 and len(gate['checks'])==6
        for c in gate['checks']:assert c['maxabs']<=2e-6 and c['gradient_relative_l2']<=.02 and c['gradient_cosine']>=.999 and set(c['gradient_norms'])==set(dict(model.named_parameters())) and all(np.isfinite(v) and v>0 for v in c['gradient_norms'].values())
        checks['new_TRAIN_photos_replayed']=20*len(train_images);checks['new_epochs']=20;checks['parameter_count']=sum(v.numel() for v in model.parameters())
    else:
        study.receipt(base,'audit_completed.json','science');t=s.read(base/'test/result.json');started=s.read(base/'test/started.json');assert t['selected_epoch']==started['selected_epoch']==f['selected_epoch'] and t['model_digest']==started['model_digest']==s.digest(best['model']) and t['best_sha256']==started['best_sha256']==f['best_sha256'] and started['sources']==study.sources() and t['test_used_for_epoch_selection'] is False
        assert t['predictions_sha256']==s.sha(t['predictions_path']);actual,checks['test']=previous.verify_predictions(Path(t['predictions_path']),best['model'],'test',m);previous.agree(actual,t['metrics'])
        if t['reused']:
            assert t['new_test_evaluations']==0 and t['model_digest'] in parent['test_registry'];known=parent['test_registry'][t['model_digest']];assert t['predictions_path']==known['predictions_path'] and t['predictions_sha256']==known['predictions_sha256']
        else:assert t['new_test_evaluations']==1 and t['model_digest'] not in parent['test_registry'] and (base/('digest_'+t['model_digest']+'.claim.json')).exists()
    out.mkdir();s.save(out/'audit.json',dict(passed=True,phase=phase,run=str(base),sources=study.sources(),training_commit=meta['git_commit'],config=cfg,checks=checks,model_forwards=0,test_images_read=0),exclusive=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--phase',choices=['fit','test'],required=True);a=p.parse_args()
    try:audit(a.base,a.out,a.phase)
    except BaseException as e:
        mark=a.base/(a.phase+'_audit_failure.json')
        if not mark.exists():study.s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
